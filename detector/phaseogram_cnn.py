"""Phaseogram CNN, pooled file model and training helpers used by the paper."""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from detector.phaseogram import C, P

SEED = 0
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def safe_name(fn: str) -> str:
    return fn.replace('/', '__').replace('\\', '__')


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


def _view(patches, gen, train):
    """Phase roll, peak normalization and row masks; channel augmentation is on audio."""
    p = patches
    if train:
        p = torch.roll(p, int(torch.randint(0, P, (1,), generator=gen).item()), dims=-1)
    p = p / p.abs().amax(dim=(-1, -2), keepdim=True).clamp_min(1e-6)
    if train:
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


def _train_fold(recs, tr, use_analytic, epochs, bs=24, lr=1e-3):
    Xa = np.stack([recs[i]['analytic'] for i in tr])
    imp = SimpleImputer(strategy='median', keep_empty_features=True).fit(Xa)
    sc = StandardScaler().fit(imp.transform(Xa))
    si = (imp, sc)
    ytr = np.array([recs[i]['y'] for i in tr])
    pos_w = torch.tensor([(ytr == 0).sum() / max(1, (ytr == 1).sum())],
                         dtype=torch.float32, device=DEVICE)
    model = FileModel(Xa.shape[1], use_analytic=use_analytic).to(DEVICE)
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
            loss.backward()
            opt.step()
    return model, si
