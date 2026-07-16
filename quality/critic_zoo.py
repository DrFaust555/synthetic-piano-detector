"""Critic zoo — єдиний API над різнорідними real-vs-synth критиками + детектор незгоди
(Фаза 1 DETECTOR_IMPROVEMENT_PLAN.md).

Обдурити ОДИН критик легко; чотири різноархітектурні — важче, а їхня НЕЗГОДА — це
детектор «тут треба слухати» (needs_ear). Критики:

  NetCritic   realness_net.net_logit  (phaseogram-CNN + 47 фіч; ловить near-real digitals)
  XgbCritic   realness_xgb            (аналітичні 47 фіч; логіку винесено сюди з
                                       tuning_objective._realness_xgb — без зміни поведінки)
  PannsCritic frozen PANNs CNN14 embedding -> logreg-голова (артефакт
                                       cache/panns_realness.joblib; тренування — тут же)
  DiscCritic  GAN-дискримінатор (nn/disc_score) — advisory=True: у worst-case ансамблю
                                       НЕ входить, лише у звіт (ЖОРСТКЕ правило №4 плану)

Кожен критик: score(path) -> {"logit", "p_synth"} (p — калібрована Platt-ом, per-критик,
cache/critic_calib.json; якщо калібрування нема — native p / sigmoid(logit)).

    python -m quality.critic_zoo <wav|dir> ...        # таблиця p / worst / disagreement
    python -m quality.critic_zoo --calibrate           # перебудувати critic_calib.json
    python -m quality.critic_zoo --train-panns          # натренувати PANNs-голову
"""
from __future__ import annotations
import os
import sys
import csv
import json
import glob
import argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CACHE = os.path.join(HERE, 'cache')
AUDIO = os.path.join(ROOT, 'audio')
REALNESS_XGB = os.path.join(CACHE, 'realness_xgb.json')
TRENDS_PATH = os.path.join(CACHE, 'register_trends_cal.json')
CALIB_PATH = os.path.join(CACHE, 'critic_calib.json')
PANNS_REAL = os.path.join(CACHE, 'panns_realness.joblib')

NEEDS_EAR_THRESH = 0.35
REAL_MIN = 60.0                       # rating_cal поріг «реального-доброго» (як у loss.py)


# --------------------------------------------------------------------------- math
def _sigmoid(z):
    z = np.clip(z, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z))


def _logit(p, eps=1e-6):
    p = float(np.clip(p, eps, 1.0 - eps))
    return float(np.log(p / (1.0 - p)))


def platt_fit(logits, labels):
    """Одновимірний Platt (logistic) logit->P(synth). Повертає (A, B); A>0 = монотонно.
    C=1.0: на майже-сепарабельних logit-ах нерегуляризований нахил вибухає і калібрована
    p сатурує в 0/1 (disagreement вироджується у стрибок) — регуляризація тримає
    перехідну зону плавною."""
    from sklearn.linear_model import LogisticRegression
    lg = np.asarray(logits, float).reshape(-1, 1)
    y = np.asarray(labels, int)
    lr = LogisticRegression(C=1.0, solver='lbfgs', max_iter=2000).fit(lg, y)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


_CALIB = None


def _calib_all():
    global _CALIB
    if _CALIB is None:
        _CALIB = (json.load(open(CALIB_PATH, encoding='utf-8'))
                  if os.path.exists(CALIB_PATH) else {})
    return _CALIB


# --------------------------------------------------------------------------- critics
class Critic:
    name = 'base'
    advisory = False

    def _raw(self, path):
        """-> (logit, native_p_or_None). Вищий logit = синтетичніше."""
        raise NotImplementedError

    def calibrate(self, lg, native=None):
        pr = _calib_all().get(self.name)
        if pr:
            return float(_sigmoid(pr['A'] * lg + pr['B']))
        return float(native) if native is not None else float(_sigmoid(lg))

    def score(self, path) -> dict:
        lg, nat = self._raw(path)
        return {'logit': float(lg), 'p_synth': self.calibrate(float(lg), nat)}


class NetCritic(Critic):
    name = 'net'

    def _raw(self, path):
        from quality.realness_net import net_logit, NET_CKPT
        if not os.path.exists(NET_CKPT):
            raise FileNotFoundError(NET_CKPT)
        lg = net_logit(path)
        if not np.isfinite(lg):
            raise ValueError('net_logit nan')
        return float(lg), None


class XgbCritic(Critic):
    name = 'xgb'
    _clf = None
    _trends = None

    def _lazy(self):
        if XgbCritic._clf is None:
            import xgboost as xgb
            c = xgb.XGBClassifier()
            c.load_model(REALNESS_XGB)
            XgbCritic._clf = c
            XgbCritic._trends = json.load(open(TRENDS_PATH, encoding='utf-8'))
        return XgbCritic._clf, XgbCritic._trends

    def p_synth(self, path) -> float:
        """Сира P(synth) — точно як стара tuning_objective._realness_xgb."""
        from quality.pipeline import note_raw_descriptors, aggregate_file, FEATURE_COLS
        clf, trends = self._lazy()
        _m, notes, tuning = note_raw_descriptors(path)
        feats = aggregate_file(notes, tuning, trends)
        X = np.array([[feats.get(c, np.nan) for c in FEATURE_COLS]], dtype=float)
        return float(clf.predict_proba(X)[0, 1])

    def _raw(self, path):
        p = self.p_synth(path)
        return _logit(p), p


class PannsCritic(Critic):
    name = 'panns'
    _pack = None
    _tagger = None

    def _load(self):
        if PannsCritic._pack is None:
            import joblib
            PannsCritic._pack = joblib.load(PANNS_REAL)
        return PannsCritic._pack

    def _emb(self, path):
        from quality.pianoness import _ensure_assets, load_model, _to_panns
        from detune.segment import load_mono
        if PannsCritic._tagger is None:
            _ensure_assets()
            PannsCritic._tagger = load_model()
        y, sr = load_mono(path)
        _, emb = PannsCritic._tagger.inference(_to_panns(y, sr))
        return np.asarray(emb[0], dtype=np.float32)

    def _raw(self, path):
        pipe = self._load()['pipe']
        e = self._emb(path).reshape(1, -1)
        p = float(pipe.predict_proba(e)[0, 1])
        try:
            lg = float(pipe.decision_function(e)[0])
        except Exception:
            lg = _logit(p)
        return lg, p


class DiscCritic(Critic):
    name = 'disc'
    advisory = True                    # ЖОРСТКЕ правило №4: лише звіт, ніколи не в J / worst
    _sc = None

    def _scorer(self):
        if DiscCritic._sc is None:
            from nn.disc_score import DiscScorer
            DiscCritic._sc = DiscScorer()
        return DiscCritic._sc

    def _raw(self, path):
        s = self._scorer().score_file(path)   # real ~1, synth ~0
        return -float(s), None                 # синтетичність = − realness


# --------------------------------------------------------------------------- ensemble
_DEFAULT = None


def default_critics():
    """Доступні критики (ті, чиї артефакти є). DiscCritic — advisory."""
    global _DEFAULT
    if _DEFAULT is None:
        cs = [NetCritic(), XgbCritic()]
        if os.path.exists(PANNS_REAL):
            cs.append(PannsCritic())
        try:
            from nn.disc_score import DEFAULT_D
            if os.path.exists(DEFAULT_D):
                cs.append(DiscCritic())
        except Exception:
            pass
        _DEFAULT = cs
    return _DEFAULT


def ensemble(path, critics=None) -> dict:
    """{per_critic, p_mean, p_worst, disagreement, needs_ear}.
    p_worst / p_mean / disagreement — лише по НЕ-advisory критиках (advisory — у звіт)."""
    critics = critics if critics is not None else default_critics()
    per = {}
    for c in critics:
        try:
            per[c.name] = {**c.score(path), 'advisory': c.advisory}
        except Exception as e:
            per[c.name] = {'error': str(e), 'advisory': c.advisory}
    dec = [v['p_synth'] for v in per.values()
           if 'p_synth' in v and not v['advisory']]
    p_mean = float(np.mean(dec)) if dec else float('nan')
    p_worst = float(np.max(dec)) if dec else float('nan')
    disagreement = float(np.max(dec) - np.min(dec)) if len(dec) >= 2 else 0.0
    return {'per_critic': per, 'p_mean': p_mean, 'p_worst': p_worst,
            'disagreement': disagreement, 'needs_ear': disagreement > NEEDS_EAR_THRESH}


# --------------------------------------------------------------------------- labelled corpus
def _resolve(fn: str) -> str | None:
    for cand in (os.path.join(ROOT, fn), os.path.join(AUDIO, fn), fn):
        if os.path.exists(cand):
            return cand
    return None


def labelled_files(real_cap=60, synth_cap=120, seed=0):
    """(paths, labels, groups) — real (rating_cal>=REAL_MIN) vs synth (renders+libs+m1+otey)."""
    from quality.grouping import group_key
    rng = np.random.default_rng(seed)
    reals = []
    with open(os.path.join(CACHE, 'features_cal.csv'), encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if int(r.get('is_anchor', 0) or 0):
                continue
            try:
                if float(r['rating_cal']) < REAL_MIN:
                    continue
            except (ValueError, KeyError):
                continue
            p = _resolve(r['file'])
            if p and 'm1_patches' not in r['file']:
                reals.append((p, 0, group_key(r['file'], r.get('brand', ''))))
    synth = []
    for name, grp_prefix in (('synth_features.csv', 'syn'),
                             ('samplelib_features.csv', 'lib')):
        fp = os.path.join(CACHE, name)
        if not os.path.exists(fp):
            continue
        for r in csv.DictReader(open(fp, encoding='utf-8')):
            p = _resolve(r['file'])
            if p:
                g = f"{grp_prefix}:{r.get('synth_type', r.get('instrument', '?'))}"
                synth.append((p, 1, g))

    def _sub(items, cap):
        if cap and len(items) > cap:
            idx = rng.choice(len(items), cap, replace=False)
            return [items[i] for i in idx]
        return items
    reals, synth = _sub(reals, real_cap), _sub(synth, synth_cap)
    both = reals + synth
    return ([b[0] for b in both], np.array([b[1] for b in both]),
            np.array([b[2] for b in both]))


# --------------------------------------------------------------------------- calibration
def _xgb_oof_logits():
    """ЧЕСНІ OOF-логіти для xgb: перерахунок GroupKFold(5) з кешованих таблиць фіч
    (без аудіо, секунди) — той самий протокол, що train_realness. -> (logits, labels)."""
    from quality.pipeline import FEATURE_COLS
    from quality import train_realness as TR
    from sklearn.model_selection import GroupKFold
    import xgboost as xgb
    df = TR._load()
    X = df[FEATURE_COLS].values
    y = df['y'].values
    grp = df['grp'].values
    spw = float(np.sum(y == 0)) / max(1, np.sum(y == 1))
    oof = np.full(len(y), np.nan)
    for tr, te in GroupKFold(5).split(X, y, groups=grp):
        clf = xgb.XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05,
                                subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
                                eval_metric='auc', tree_method='hist', n_jobs=4)
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    ok = np.isfinite(oof)
    return [_logit(p) for p in oof[ok]], [int(v) for v in y[ok]]


def build_calibration(real_cap=60, synth_cap=120, seed=0):
    """Per-критик Platt logit->P(synth) -> critic_calib.json. Де можливо — на ЧЕСНИХ
    OOF-скорах (xgb: перерахунок з таблиць фіч; panns: OOF, збережений train_panns_realness);
    для решти (net/disc) — in-sample на вибірці файлів (позначається oof=false: Platt
    фітиться на скорах моделі по її ж тренувальних даних — брати з обережністю)."""
    from sklearn.metrics import roc_auc_score
    out = {}
    oof_src = {}
    try:
        oof_src['xgb'] = _xgb_oof_logits()
        print("  xgb: OOF з таблиць фіч", flush=True)
    except Exception as e:
        print(f"  xgb: OOF не вдався ({e}) — впаде на in-sample")
    if os.path.exists(PANNS_REAL):
        try:
            import joblib
            pack = joblib.load(PANNS_REAL)
            if 'oof' in pack and 'y' in pack:
                pp = np.asarray(pack['oof'], float)
                oof_src['panns'] = ([_logit(v) for v in pp],
                                    [int(v) for v in np.asarray(pack['y'])])
                print("  panns: збережений OOF", flush=True)
        except Exception:
            pass
    paths = y_files = None
    for c in default_critics():
        if c.name in oof_src:
            lg, yy = oof_src[c.name]
            oof = True
        else:
            if paths is None:
                paths, y_files, _ = labelled_files(real_cap, synth_cap, seed)
                print(f"calib-вибірка (in-sample) {len(paths)} файлів "
                      f"(real {(y_files==0).sum()}, synth {(y_files==1).sum()})", flush=True)
            lg, yy, oof = [], [], False
            for p, yi in zip(paths, y_files):
                try:
                    lg.append(c._raw(p)[0]); yy.append(int(yi))
                except Exception:
                    continue
        if len(set(yy)) < 2:
            print(f"  {c.name:6s}: замало даних, пропуск"); continue
        A, B = platt_fit(lg, yy)
        p_cal = _sigmoid(A * np.array(lg) + B)
        auc = float(roc_auc_score(yy, p_cal))
        out[c.name] = {'A': A, 'B': B, 'auc': round(auc, 4), 'n': len(yy),
                       'advisory': c.advisory, 'oof': bool(oof)}
        print(f"  {c.name:6s}: A={A:+.3f} B={B:+.3f}  AUC={auc:.3f}  n={len(yy)}"
              f"  {'OOF' if oof else 'in-sample'}", flush=True)
    json.dump(out, open(CALIB_PATH, 'w'), indent=2)
    global _CALIB
    _CALIB = out
    print(f"-> {os.path.relpath(CALIB_PATH, ROOT)}")


# --------------------------------------------------------------------------- panns head training
def train_panns_realness(real_cap=400, synth_cap=600, seed=0, n_pca=48):
    """Натренувати PANNs-embedding -> logreg real-vs-synth голову; GroupKFold по інструменту.
    Артефакт cache/panns_realness.joblib. ВАЖКО (PANNs-embedding по кожному файлу)."""
    import joblib
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score
    from quality.pianoness import _ensure_assets, load_model, _to_panns
    from detune.segment import load_mono

    paths, y, grp = labelled_files(real_cap, synth_cap, seed)
    _ensure_assets()
    tagger = load_model()
    E, yy, gg = [], [], []
    for i, (p, yi, g) in enumerate(zip(paths, y, grp)):
        try:
            x, sr = load_mono(p)
            _, emb = tagger.inference(_to_panns(x, sr))
            E.append(np.asarray(emb[0], dtype=np.float32)); yy.append(int(yi)); gg.append(g)
        except Exception as e:
            print(f"  ERR {os.path.basename(p)}: {e}")
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(paths)} embedded", flush=True)
    E = np.stack(E); yy = np.array(yy); gg = np.array(gg)
    print(f"embedded {len(E)} (real {(yy==0).sum()}, synth {(yy==1).sum()}, "
          f"groups {len(set(gg))})", flush=True)

    def _pipe():
        return Pipeline([('sc', StandardScaler()),
                         ('pca', PCA(n_components=min(n_pca, E.shape[1]), random_state=seed)),
                         ('clf', LogisticRegression(C=0.5, class_weight='balanced',
                                                    max_iter=5000, random_state=seed))])
    oof = np.zeros(len(yy))
    for tr, va in GroupKFold(5).split(E, yy, groups=gg):
        oof[va] = _pipe().fit(E[tr], yy[tr]).predict_proba(E[va])[:, 1]
    auc = float(roc_auc_score(yy, oof))
    print(f"GroupKFold CV-AUC (PANNs real-vs-synth): {auc:.3f}", flush=True)
    pipe = _pipe().fit(E, yy)                          # deploy on all data
    # oof/y зберігаємо для ЧЕСНОГО Platt-калібрування у build_calibration
    joblib.dump({'pipe': pipe, 'cv_auc': round(auc, 4), 'n': len(yy),
                 'oof': oof, 'y': yy}, PANNS_REAL)
    print(f"-> {os.path.relpath(PANNS_REAL, ROOT)}")


# --------------------------------------------------------------------------- CLI
def _iter_files(args):
    files = []
    for a in args:
        p = a if os.path.isabs(a) else os.path.join(ROOT, a)
        if os.path.isdir(p):
            files += sorted(glob.glob(os.path.join(p, '*.wav')))
        else:
            files += sorted(glob.glob(p))
    return files


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('paths', nargs='*')
    ap.add_argument('--calibrate', action='store_true')
    ap.add_argument('--train-panns', action='store_true')
    args = ap.parse_args()
    if args.calibrate:
        build_calibration(); return
    if args.train_panns:
        train_panns_realness(); return
    files = _iter_files(args.paths)
    if not files:
        print("Вкажи <wav|dir> …, або --calibrate / --train-panns."); return
    critics = default_critics()
    names = [c.name for c in critics]
    print("critics: " + ", ".join(f"{n}{'*' if c.advisory else ''}" for n, c
                                   in zip(names, critics)) + "   (* = advisory)")
    hdr = f"{'file':30s}" + ''.join(f'{n[:8]:>9s}' for n in names) + \
          f"{'worst':>8s}{'disag':>8s}  flag"
    print(hdr); print('-' * len(hdr))
    for f in files:
        e = ensemble(f, critics)
        cells = ''
        for n in names:
            v = e['per_critic'].get(n, {})
            cells += (f"{v['p_synth']:9.3f}" if 'p_synth' in v else f"{'ERR':>9s}")
        flag = 'NEEDS-EAR' if e['needs_ear'] else ''
        print(f"{os.path.basename(f)[:30]:30s}{cells}{e['p_worst']:8.3f}"
              f"{e['disagreement']:8.3f}  {flag}")


if __name__ == '__main__':
    main()
