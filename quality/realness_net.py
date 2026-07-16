"""Time-domain realness_net: a phaseogram CNN late-fused with the 47 analytic features,
trained real-vs-synthetic to catch the near-real digital pianos the feature critic misses.

    python -m quality.realness_net [epochs]

Per note: cached pitch-synchronous phaseogram (quality/build_wave_cache.py) → 2D CNN →
note embedding.  Per file: mean+std pool → file embedding ⊕ 47 features → head → P(synth).
Channel invariance (the documented precondition) is taught by augmenting BOTH classes with
a random harmonic-EQ + phase roll + per-cycle gain jitter applied IN the phaseogram domain,
with a consistency loss between two augmented views.  Grouped CV by instrument identity.
"""

from __future__ import annotations
import os
import sys
import csv
import json
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

from quality.pipeline import FEATURE_COLS
from quality.phasefeat import C, P
from quality.build_wave_cache import safe_name

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, 'cache')
PHASE_DIR = os.path.join(CACHE, 'phase')
MANIFEST = os.path.join(CACHE, 'phase_manifest.json')
SEED = 0
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# --------------------------------------------------------------------------- data
def _analytic_table():
    tab = {}
    # v2 таблиці (features_maestro/negatives) — опційні; беруться, якщо є на диску
    names = ['features_cal.csv', 'synth_features.csv', 'samplelib_features.csv',
             'features_maestro.csv', 'features_negatives.csv']
    for name in names:
        path = os.path.join(CACHE, name)
        if not os.path.exists(path):
            continue
        with open(path, encoding='utf-8') as f:
            for r in csv.DictReader(f):
                vec = [float(r[c]) if r.get(c, '') not in ('', 'nan', None) else np.nan
                       for c in FEATURE_COLS]
                tab[r['file']] = np.array(vec, dtype=np.float32)
    return tab


def load_dataset(manifest_path=MANIFEST):
    manifest = json.load(open(manifest_path, encoding='utf-8'))
    ana = _analytic_table()
    nan = np.full(len(FEATURE_COLS), np.nan, dtype=np.float32)
    recs = []
    for m in manifest:
        npy = os.path.join(PHASE_DIR, safe_name(m['file']) + '.npy')
        if not os.path.exists(npy):
            continue
        img = np.load(npy).astype(np.float32)            # (n, C, P)
        recs.append({'file': m['file'], 'y': int(m['label']), 'stype': m['stype'],
                     'grp': m['group'], 'phase': img,
                     'analytic': ana.get(m['file'], nan)})
    return recs


# -------------------------------------------------------------------------- model
class PhaseCNN(nn.Module):
    def __init__(self, chans=(16, 32, 64, 128), emb=128):
        super().__init__()
        def blk(ci, co):
            return nn.Sequential(nn.Conv2d(ci, co, 3, padding=1), nn.BatchNorm2d(co),
                                 nn.ReLU(inplace=True), nn.MaxPool2d(2))
        layers, ci = [], 1
        for co in chans:
            layers.append(blk(ci, co)); ci = co
        layers.append(nn.AdaptiveAvgPool2d(1))
        self.net = nn.Sequential(*layers)
        self.fc = nn.Linear(ci, emb)

    def forward(self, x):                                 # (N,1,C,P)
        return self.fc(self.net(x).flatten(1))


class FileModel(nn.Module):
    def __init__(self, n_analytic, emb=128, hidden=128, pdrop=0.4, use_analytic=True):
        super().__init__()
        self.cnn = PhaseCNN(emb=emb)
        self.use_analytic = use_analytic
        fin = 2 * emb + (n_analytic if use_analytic else 0)
        self.head = nn.Sequential(nn.Linear(fin, hidden), nn.ReLU(inplace=True),
                                  nn.Dropout(pdrop), nn.Linear(hidden, 1))

    def forward(self, patches, counts, analytic):
        e = self.cnn(patches)
        pooled, idx = [], 0
        for c in counts:
            seg = e[idx:idx + c]; idx += c
            std = seg.std(0, unbiased=False) if c > 1 else torch.zeros_like(seg[0])
            pooled.append(torch.cat([seg.mean(0), std]))
        p = torch.stack(pooled)
        z = torch.cat([p, analytic], dim=1) if self.use_analytic else p
        return self.head(z).squeeze(1)


# ----------------------------------------------------------------------- channel aug
W_CONS = 1.0
EQ_TILT_DB = 6.0
EQ_PEAK_DB = 5.0


def _harmonic_eq(p, gen):
    """Random smooth EQ applied along the within-cycle axis (= reweight harmonics),
    shared across a note's cycles → emulates recording-channel spectral colour."""
    N = p.shape[0]
    Y = torch.fft.rfft(p, dim=-1)                         # (N,1,C,P/2+1)
    nb = Y.shape[-1]
    bands = torch.linspace(0, 1, nb, device=p.device)
    tilt = (torch.rand(N, 1, generator=gen) * 2 - 1).to(p.device) * EQ_TILT_DB
    curve = tilt * (bands[None] - 0.5)
    for _ in range(2):
        c = torch.rand(N, 1, generator=gen).to(p.device)
        w = (0.05 + torch.rand(N, 1, generator=gen) * 0.25).to(p.device)
        g = ((torch.rand(N, 1, generator=gen) * 2 - 1) * EQ_PEAK_DB).to(p.device)
        curve = curve + g * torch.exp(-0.5 * ((bands[None] - c) / w) ** 2)
    gain = torch.pow(10.0, curve.clamp(-10, 10) / 20.0)  # (N,nb) linear
    Y = Y * gain[:, None, None, :]
    return torch.fft.irfft(Y, n=p.shape[-1], dim=-1)


def _view(patches, gen, train):
    p = patches
    if train:
        N = p.shape[0]
        p = _harmonic_eq(p, gen)
        # random phase roll along the within-cycle axis (one shift per batch is enough —
        # the onset phase is arbitrary; varying it across batches teaches invariance)
        p = torch.roll(p, int(torch.randint(0, P, (1,), generator=gen).item()), dims=-1)
        # per-cycle gain jitter + small noise
        p = p * (1 + 0.08 * torch.randn(N, 1, C, 1, generator=gen).to(p.device))
        p = p + 0.02 * torch.randn_like(p)
    # peak-normalise each note
    pk = p.abs().amax(dim=(-1, -2), keepdim=True).clamp_min(1e-6)
    p = p / pk
    if train:                                            # row(time)-mask: drop a few cycles
        N = p.shape[0]
        for _ in range(2):
            w = int(torch.randint(0, 8, (1,), generator=gen).item())
            if w:
                r0 = int(torch.randint(0, max(1, C - w), (1,), generator=gen).item())
                p[:, :, r0:r0 + w, :] = 0.0
    return p


# ----------------------------------------------------------------------- train/eval
def _gather(recs, idxs):
    ph, counts, ana, ys = [], [], [], []
    for i in idxs:
        r = recs[i]
        ph.append(torch.from_numpy(r['phase']))
        counts.append(r['phase'].shape[0]); ana.append(r['analytic']); ys.append(r['y'])
    patches = torch.cat(ph).unsqueeze(1).float()         # (sumN,1,C,P)
    return patches, counts, np.stack(ana), np.array(ys, dtype=np.float32)


def _prep_ana(ana_np, si):
    imp, sc = si
    return torch.from_numpy(sc.transform(imp.transform(ana_np)).astype(np.float32))


def _predict(model, recs, idxs, si, bs=64):
    model.eval(); out = []
    with torch.no_grad():
        for b in range(0, len(idxs), bs):
            sub = idxs[b:b + bs]
            patches, counts, ana_np, _ = _gather(recs, sub)
            v = _view(patches.to(DEVICE), None, train=False)
            ana = _prep_ana(ana_np, si).to(DEVICE)
            out.append(torch.sigmoid(model(v, counts, ana)).cpu().numpy())
    return np.concatenate(out)


def _train_fold(recs, tr, use_analytic, epochs, bs=24, lr=1e-3, pair_w=0.0, pair_map=None):
    Xa = np.stack([recs[i]['analytic'] for i in tr])
    imp = SimpleImputer(strategy='median', keep_empty_features=True).fit(Xa)
    sc = StandardScaler().fit(imp.transform(Xa))
    si = (imp, sc)
    ytr = np.array([recs[i]['y'] for i in tr])
    pos_w = torch.tensor([(ytr == 0).sum() / max(1, (ytr == 1).sum())],
                         dtype=torch.float32, device=DEVICE)
    model = FileModel(len(FEATURE_COLS), use_analytic=use_analytic).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    g = torch.Generator().manual_seed(SEED)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(tr), generator=g).tolist()
        for b in range(0, len(tr), bs):
            idxs = [tr[perm[k]] for k in range(b, min(b + bs, len(tr)))]
            patches, counts, ana_np, ys_np = _gather(recs, idxs)
            patches = patches.to(DEVICE)
            ana = _prep_ana(ana_np, si).to(DEVICE)
            ys = torch.from_numpy(ys_np).to(DEVICE)
            v1 = _view(patches, g, True); v2 = _view(patches, g, True)
            l1 = model(v1, counts, ana); l2 = model(v2, counts, ana)
            opt.zero_grad()
            loss = (0.5 * (lossf(l1, ys) + lossf(l2, ys))
                    + W_CONS * F.mse_loss(torch.sigmoid(l1), torch.sigmoid(l2)))
            loss.backward()                    # звільняє основний граф ДО парного (пам'ять)
            # файлова консистентність clean<->#waug: той самий файл через інший
            # «канал» мусить діставати той самий вердикт. Аналітика — СПІЛЬНА
            # (базового item-а), щоб лосс тиснув лише на фазограмну гілку.
            # Окремий backward (градієнти акумулюються) + ліміт пар на батч — інакше
            # +2 повні форварди поверх двох в'ю дають CUDA OOM.
            if pair_w and pair_map:
                pairs = [(i, pair_map[i]) for i in idxs if i in pair_map][:8]
                if pairs:
                    del v1, v2, l1, l2, patches
                    bi = [a for a, _ in pairs]; pi = [p for _, p in pairs]
                    bp, bc, ba, _ = _gather(recs, bi)
                    pp, pc, _, _ = _gather(recs, pi)
                    ana_b = _prep_ana(ba, si).to(DEVICE)
                    lb = model(_view(bp.to(DEVICE), g, True), bc, ana_b)
                    lp = model(_view(pp.to(DEVICE), g, True), pc, ana_b)
                    (pair_w * F.mse_loss(torch.sigmoid(lb), torch.sigmoid(lp))).backward()
            opt.step()
    return model, si


# ----------------------------------------------------- deploy: full-data model + inference
# NET_CKPT — env-перемикач версії (default = старий; v2 через REALNESS_NET_CKPT або
# critic_eval --ckpt, який виставляє rn.NET_CKPT напряму). Старий чекпойнт НЕ перезаписуємо.
NET_CKPT = os.environ.get('REALNESS_NET_CKPT', os.path.join(CACHE, 'realness_net.pt'))
MANIFEST_V2 = os.path.join(CACHE, 'phase_manifest_v2.json')
TRENDS = os.path.join(CACHE, 'register_trends_cal.json')
_NET = None
_TRENDS = None


def train_full(epochs=40, out=NET_CKPT, manifest_path=MANIFEST, pair_cons=0.0):
    """Train ONE model on ALL data (real vs synth) and save it for inference.
    run() only produces CV OOF (no deployable model); this is the deployable checkpoint.
    pair_cons > 0 вмикає файлову консистентність clean<->#waug (потрібен manifest v2)."""
    torch.manual_seed(SEED); np.random.seed(SEED)
    recs = load_dataset(manifest_path)
    pair_map = None
    if pair_cons:
        by_file = {r['file']: i for i, r in enumerate(recs)}
        pair_map = {}
        for f, i in by_file.items():
            j = by_file.get(f + '#waug')
            if j is not None:
                pair_map[i] = j; pair_map[j] = i
        print(f"pair-consistency: w={pair_cons}, pairs={len(pair_map)//2}", flush=True)
    print(f"train_full on {len(recs)} files ({epochs} ep, device={DEVICE})", flush=True)
    model, si = _train_fold(recs, list(range(len(recs))), True, epochs,
                            pair_w=pair_cons, pair_map=pair_map)
    imp, sc = si
    torch.save({'state_dict': model.state_dict(), 'imputer': imp, 'scaler': sc,
                'n_analytic': len(FEATURE_COLS), 'feature_cols': list(FEATURE_COLS)}, out)
    print(f"saved -> {out}", flush=True)


def load_net():
    global _NET
    if _NET is None:
        d = torch.load(NET_CKPT, weights_only=False, map_location=DEVICE)
        model = FileModel(d['n_analytic'], use_analytic=True).to(DEVICE)
        model.load_state_dict(d['state_dict']); model.eval()
        _NET = (model, (d['imputer'], d['scaler']))
    return _NET


def _net_trends():
    global _TRENDS
    if _TRENDS is None:
        _TRENDS = json.load(open(TRENDS, encoding='utf-8'))
    return _TRENDS


def net_logit(path: str) -> float:
    """Raw synthetic-ness LOGIT from realness_net (higher = more synthetic).
    Prefer this over net_p_synth for OPTIMIZATION: the sigmoid saturates to 1.000 for all
    physical-model output (Otey is synthetic at the waveform level no matter the knobs), but
    the LOGIT keeps a usable gradient (real≈−9, Otey baseline≈+28, Otey with life≈+13).
    Builds the phaseogram + 47 analytic features on the fly.  Returns nan if unusable."""
    from detune.segment import load_mono, build_segments
    from quality.pipeline import (select_diverse_notes, note_raw_descriptors, aggregate_file)
    from quality.phasefeat import compute_phaseogram
    model, si = load_net()
    x, sr = load_mono(path)
    segs, _ = build_segments(x, sr)
    idx = select_diverse_notes(segs)
    imgs = [compute_phaseogram(segs[j].x, sr, segs[j].f0)
            for j in idx if 24.0 < segs[j].f0 < 4500.0]
    if not imgs:
        return float('nan')
    patches = _view(torch.from_numpy(np.stack(imgs).astype(np.float32)).unsqueeze(1).to(DEVICE),
                    None, train=False)
    _meta, notes, tuning = note_raw_descriptors(path)
    feats = aggregate_file(notes, tuning, _net_trends())
    ana_np = np.array([[feats.get(c, np.nan) for c in FEATURE_COLS]], dtype=np.float32)
    ana = _prep_ana(ana_np, si).to(DEVICE)
    with torch.no_grad():
        return float(model(patches, [len(imgs)], ana).item())


def net_p_synth(path: str) -> float:
    """P(synth) = sigmoid(net_logit).  NOTE: saturates to ~1.000 on physical-model output —
    use net_logit for optimization; this is for reporting/thresholding only."""
    lg = net_logit(path)
    return float(1.0 / (1.0 + np.exp(-lg))) if np.isfinite(lg) else float('nan')


def run(epochs=40):
    torch.manual_seed(SEED); np.random.seed(SEED)
    recs = load_dataset()
    y = np.array([r['y'] for r in recs])
    grp = np.array([r['grp'] for r in recs])
    stype = np.array([r['stype'] for r in recs])
    Xa = np.stack([r['analytic'] for r in recs])
    print(f"device={DEVICE}  files={len(recs)}  real={int((y==0).sum())} synth={int((y==1).sum())}")
    print(f"  synth types: {dict(zip(*np.unique(stype[y==1], return_counts=True)))}")

    n = len(recs)
    # ── resumable per-fold checkpoint (hibernate-safe between folds) ──
    ckpt = os.path.join(CACHE, 'realness_oof.npz')
    if os.path.exists(ckpt):
        d = np.load(ckpt)
        if int(d['n']) == n:
            oof_fuse, oof_neuro, oof_base = d['fuse'].copy(), d['neuro'].copy(), d['base'].copy()
            done = set(int(x) for x in d['done'])
            print(f"  RESUME: folds already done = {sorted(done)}", flush=True)
        else:
            print("  checkpoint stale (dataset changed) → starting fresh", flush=True)
            oof_fuse = np.zeros(n); oof_neuro = np.zeros(n); oof_base = np.zeros(n); done = set()
    else:
        oof_fuse = np.zeros(n); oof_neuro = np.zeros(n); oof_base = np.zeros(n); done = set()

    for k, (tr, va) in enumerate(GroupKFold(5).split(Xa, y, groups=grp)):
        if k in done:
            continue
        assert not (set(grp[tr]) & set(grp[va])), "leakage!"
        imp = SimpleImputer(strategy='median', keep_empty_features=True).fit(Xa[tr])
        sc = StandardScaler().fit(imp.transform(Xa[tr]))
        base = LogisticRegression(C=0.3, max_iter=20000, class_weight='balanced').fit(
            sc.transform(imp.transform(Xa[tr])), y[tr])
        oof_base[va] = base.predict_proba(sc.transform(imp.transform(Xa[va])))[:, 1]
        m, si = _train_fold(recs, list(tr), True, epochs)
        oof_fuse[va] = _predict(m, recs, list(va), si)
        m2, si2 = _train_fold(recs, list(tr), False, epochs)
        oof_neuro[va] = _predict(m2, recs, list(va), si2)
        done.add(k)
        np.savez(ckpt, fuse=oof_fuse, neuro=oof_neuro, base=oof_base,
                 done=np.array(sorted(done)), n=n)
        print(f"  fold {k} done (val={len(va)})  [checkpointed]", flush=True)

    if len(done) < 5:
        print(f"\nINCOMPLETE: {sorted(done)} of 5 folds done. Re-run to continue.", flush=True)
        return
    print(f"\nGrouped CV-AUC (real vs synth):")
    print(f"  analytic baseline (logreg) : {roc_auc_score(y, oof_base):.3f}")
    print(f"  phaseogram CNN only        : {roc_auc_score(y, oof_neuro):.3f}")
    print(f"  LATE FUSION (CNN+analytic) : {roc_auc_score(y, oof_fuse):.3f}")

    print("\nAliveness P(real)*100 — does the phaseogram net catch sample-based pianos?")
    for st in ['rendered', 'otey', 'm1', 'yt', 'sampled_digital']:
        msk = (y == 1) & (stype == st)
        if msk.sum():
            print(f"  {st:16s} n={int(msk.sum()):4d}  base alive={(1-oof_base[msk]).mean()*100:5.1f}"
                  f"  | fusion alive={(1-oof_fuse[msk]).mean()*100:5.1f}"
                  f"  caught@50={(oof_fuse[msk]>0.5).mean()*100:4.0f}%")
    print(f"  real             n={int((y==0).sum()):4d}  fusion alive={(1-oof_fuse[y==0]).mean()*100:5.1f}")

    # the ceiling for the TIME-DOMAIN method: AUC vs algorithmic vs sample-based
    algo = np.isin(stype, ['rendered', 'otey']); samp = np.isin(stype, ['sampled_digital', 'yt', 'm1'])
    for nm, mk in [('ALGORITHMIC (soundfont+Otey)', algo), ('SAMPLE-BASED (libs+digitals+m1)', samp)]:
        sel = (y == 0) | mk
        print(f"  fusion AUC real vs {nm:32s}: {roc_auc_score(y[sel], oof_fuse[sel]):.3f}")


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        rest = sys.argv[2:]
        if rest and rest[0].startswith('v') and rest[0] != 'v1':
            # розширений набір (manifest v2) -> versions/realness_net_<tag>.pt (v2, v2b, …)
            # опції: число = епохи; cons=<w> = файлова консистентність clean<->#waug
            os.makedirs(os.path.join(CACHE, 'versions'), exist_ok=True)
            ep, cons = 40, 0.0
            for tok in rest[1:]:
                if tok.startswith('cons='):
                    cons = float(tok[5:])
                else:
                    ep = int(tok)
            train_full(epochs=ep,
                       out=os.path.join(CACHE, 'versions', f'realness_net_{rest[0]}.pt'),
                       manifest_path=MANIFEST_V2, pair_cons=cons)
        else:
            train_full(epochs=int(rest[0]) if rest else 40)
    else:
        run(epochs=int(sys.argv[1]) if len(sys.argv) > 1 else 40)
