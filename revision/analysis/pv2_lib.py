"""Protocol v2 -- shared helpers (paths, models, folds, AUC + grouped bootstrap, feature extraction).

Everything here is deterministic:
  * fold assignment = GroupKFold's greedy algorithm with a FIXED tie order (groups sorted by name,
    permuted with numpy default_rng(0), then stable-sorted by size);
  * XGBoost n_jobs=1, random_state=0 (paper hyper-parameters from quality/train_realness.py);
  * logistic regression = lbfgs (deterministic);
  * bootstrap = numpy default_rng(0), re-seeded for every comparison.
Nothing in the repository is modified; all outputs go to Stattia/revision/analysis/out/.
"""
from __future__ import annotations

import os
import sys
import json
import hashlib
import warnings
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))          # project root
OUT = os.path.join(HERE, 'out')
FEAT_CACHE = os.path.join(OUT, 'cache')
QCACHE = os.path.join(ROOT, 'quality', 'cache')
DATA_V2 = os.path.join(ROOT, 'Stattia', 'revision', 'data_v2')
KIT = os.path.join(ROOT, 'Stattia', 'listening_kit')

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from quality.pipeline import FEATURE_COLS, DESCRIPTORS          # noqa: E402

MED16 = [f'{d}_med' for d in DESCRIPTORS]                          # Group A medians (features 1-16)
FEATSETS = {'all47': list(FEATURE_COLS), 'med16': MED16}

SEED = 0
N_FOLDS = 5
N_BOOT = 2000

# ----------------------------------------------------------------------------- kinds
REAL_KINDS = ['real_isolated', 'real_performance']
ALGO_KINDS = ['algo_single', 'algo_performance']
# excluded from TRAINING under T2 (scored zero-shot by every fold model)
SB_KINDS = ['sampled_library', 'sampled_digital_performance', 'modelled_digital', 'm1', 'digital_other', 'sound_module']


def norm_token(s) -> str:
    """lower-case alphanumerics only ('Casio_CA-110 Piano (FM).sf2' -> 'casioca110pianofm')."""
    s = str(s).lower()
    if s.endswith('.sf2'):
        s = s[:-4]
    return ''.join(ch for ch in s if ch.isalnum())


# ----------------------------------------------------------------------------- models
def check_xgboost():
    try:
        import xgboost as xgb
    except ImportError:
        raise SystemExit('xgboost is not importable. Install xgboost==3.2.0 (pip install xgboost==3.2.0) '
                         'or put a folder containing it on PYTHONPATH.')
    return xgb.__version__


def make_model(kind: str, spw: float):
    if kind == 'xgb':
        import xgboost as xgb
        # quality/train_realness.py hyper-parameters; n_jobs=1 + random_state=0 for determinism
        return xgb.XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
                                 colsample_bytree=0.8, scale_pos_weight=spw, eval_metric='auc',
                                 tree_method='hist', n_jobs=1, random_state=SEED)
    if kind == 'lr':
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        # quality/realness_net.py analytic-baseline settings
        return Pipeline([('i', SimpleImputer(strategy='median', keep_empty_features=True)),
                         ('s', StandardScaler()),
                         ('m', LogisticRegression(C=0.3, max_iter=20000, class_weight='balanced'))])
    raise ValueError(kind)


def fit_predict(kind, Xtr, ytr, Xte_list, spw=None):
    """spw = scale_pos_weight; train_realness.py uses ONE global n_real/n_synth ratio over the
    whole training population (not per fold) -- callers pass that; default = ratio of ytr."""
    if spw is None:
        spw = float(np.sum(ytr == 0)) / max(1, int(np.sum(ytr == 1)))
    m = make_model(kind, spw)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        m.fit(Xtr, ytr)
        return m, [m.predict_proba(X)[:, 1] if len(X) else np.zeros(0) for X in Xte_list]


# ----------------------------------------------------------------------------- folds
def _greedy_assign(names, sizes, fold_load, seed, out_map):
    names = np.asarray(names, dtype=object)
    sizes = np.asarray(sizes, dtype=float)
    o = np.argsort(names.astype(str), kind='stable')
    names, sizes = names[o], sizes[o]
    perm = np.random.default_rng(seed).permutation(len(names))
    names, sizes = names[perm], sizes[perm]
    order = np.argsort(-sizes, kind='stable')
    for i in order:
        f = int(np.argmin(fold_load))                 # first lightest fold
        out_map[names[i]] = f
        fold_load[f] += sizes[i]
    return out_map


def assign_folds(grp_base: pd.Series, grp_new: pd.Series | None = None, k=N_FOLDS, seed=SEED):
    """group -> fold.  Base-corpus groups first (pure function of the base corpus); groups that
    only occur in newly added data are then assigned with the same greedy rule, continuing
    from the base fold loads, so the base assignment never changes when data_v2 arrives."""
    fmap, load = {}, np.zeros(k)
    vc = grp_base.value_counts()
    _greedy_assign(vc.index.values, vc.values, load, seed, fmap)
    if grp_new is not None and len(grp_new):
        vn = grp_new[~grp_new.isin(list(fmap))].value_counts()
        if len(vn):
            _greedy_assign(vn.index.values, vn.values, load, seed, fmap)
    return fmap


# ----------------------------------------------------------------------------- AUC + bootstrap
def auc_fast(y, p, w=None):
    """Weighted Mann-Whitney AUC (ties = 1/2).  y in {0,1}; w = non-negative weights."""
    y = np.asarray(y); p = np.asarray(p, dtype=float)
    if w is None:
        w = np.ones(len(y))
    u, inv = np.unique(p, return_inverse=True)
    wp = np.bincount(inv, weights=w * (y == 1), minlength=len(u))
    wn = np.bincount(inv, weights=w * (y == 0), minlength=len(u))
    Wp, Wn = wp.sum(), wn.sum()
    if Wp <= 0 or Wn <= 0:
        return np.nan
    cn = np.cumsum(wn) - wn
    return float(np.sum(wp * (cn + 0.5 * wn)) / (Wp * Wn))


def grouped_boot_ci(y, p, grp, n=N_BOOT, seed=SEED):
    """Stratified grouped bootstrap: real groups and synthetic groups are resampled with
    replacement separately (each class keeps its number of groups), AUC recomputed with the
    multiplicities as weights.  Returns (lo, hi) = 2.5 / 97.5 percentiles."""
    y = np.asarray(y); p = np.asarray(p, dtype=float); grp = np.asarray(grp).astype(str)
    rng = np.random.default_rng(seed)
    u, inv = np.unique(p, return_inverse=True)
    ycls = {}
    for c in (0, 1):
        g = np.unique(grp[y == c])
        ycls[c] = g
    gidx = {}
    for c in (0, 1):
        gi = {g: i for i, g in enumerate(ycls[c])}
        gidx[c] = np.array([gi.get(g, -1) if yy == c else -1 for g, yy in zip(grp, y)])
    pos, neg = (y == 1), (y == 0)
    aucs = np.empty(n)
    for b in range(n):
        w = np.zeros(len(y))
        for c, m in ((0, neg), (1, pos)):
            G = len(ycls[c])
            cnt = np.bincount(rng.integers(0, G, G), minlength=G).astype(float)
            w[m] = cnt[gidx[c][m]]
        wp = np.bincount(inv, weights=w * pos, minlength=len(u))
        wn = np.bincount(inv, weights=w * neg, minlength=len(u))
        cn = np.cumsum(wn) - wn
        aucs[b] = np.sum(wp * (cn + 0.5 * wn)) / (wp.sum() * wn.sum())
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def compare(y, p, grp, real_mask, syn_mask, boot=True):
    """Metrics for one real-vs-synthetic comparison."""
    sel = real_mask | syn_mask
    out = {'n_real': int(real_mask.sum()), 'g_real': int(len(np.unique(grp[real_mask]))),
           'n_syn': int(syn_mask.sum()), 'g_syn': int(len(np.unique(grp[syn_mask])))}
    if out['n_real'] == 0 or out['n_syn'] == 0:
        return {**out, 'auc': np.nan, 'ci_lo': np.nan, 'ci_hi': np.nan, 'det_syn': np.nan,
                'fa_real': np.nan, 'meanP_syn': np.nan, 'meanP_real': np.nan}
    yy = np.where(syn_mask[sel], 1, 0)
    pp, gg = p[sel], grp[sel]
    a = auc_fast(yy, pp)
    lo, hi = grouped_boot_ci(yy, pp, gg) if boot else (np.nan, np.nan)
    return {**out, 'auc': a, 'ci_lo': lo, 'ci_hi': hi,
            'det_syn': float(np.mean(p[syn_mask] > 0.5)), 'fa_real': float(np.mean(p[real_mask] > 0.5)),
            'meanP_syn': float(np.mean(p[syn_mask])), 'meanP_real': float(np.mean(p[real_mask]))}


# ----------------------------------------------------------------------------- feature extraction
_TRENDS = None


def _trends():
    global _TRENDS
    if _TRENDS is None:
        _TRENDS = json.load(open(os.path.join(QCACHE, 'register_trends_cal.json'), encoding='utf-8'))
    return _TRENDS


def extract_one(path: str) -> dict:
    """The single-file pipeline used for every isolated-note file in the corpus
    (quality.pipeline.note_raw_descriptors + aggregate_file, register_trends_cal.json)."""
    from quality.pipeline import note_raw_descriptors, aggregate_file
    try:
        meta, notes, tuning = note_raw_descriptors(path)
        feats = aggregate_file(notes, tuning, _trends())
        return {'n_notes': meta['n_notes'], 'duration_s': meta['duration_s'], 'mode': meta['mode'],
                'error': '', **feats}
    except Exception as e:                              # recorded, never silently dropped
        return {'n_notes': 0, 'duration_s': np.nan, 'mode': '', 'error': repr(e)[:200]}


def _sig(path):
    st = os.stat(path)
    return f'{st.st_size}:{int(st.st_mtime)}'


def extract_cached(paths: list[str], cache_name: str, workers: int = 8) -> pd.DataFrame:
    """Extract features for absolute `paths`; cache rows in out/cache/<cache_name>.csv keyed by
    (relative path, size, mtime).  Returns a DataFrame in the order of `paths`."""
    os.makedirs(FEAT_CACHE, exist_ok=True)
    cpath = os.path.join(FEAT_CACHE, cache_name + '.csv')
    rel = [os.path.relpath(p, ROOT).replace('\\', '/') for p in paths]
    sig = [_sig(p) for p in paths]
    cache = pd.read_csv(cpath, float_precision='round_trip') if os.path.exists(cpath) else pd.DataFrame()
    have = {}
    if len(cache):
        for _, r in cache.iterrows():
            have[(r['file'], str(r['sig']))] = r.to_dict()
    todo = [i for i, (f, s) in enumerate(zip(rel, sig)) if (f, s) not in have]
    if todo:
        print(f'  [extract] {cache_name}: {len(todo)} new file(s) (cached {len(paths) - len(todo)})', flush=True)
        if workers > 1 and len(todo) > 4:
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=workers) as ex:
                res = list(ex.map(extract_one, [paths[i] for i in todo], chunksize=4))
        else:
            res = [extract_one(paths[i]) for i in todo]
        for i, r in zip(todo, res):
            have[(rel[i], sig[i])] = {'file': rel[i], 'sig': sig[i], **r}
    rows = [have[(f, s)] for f, s in zip(rel, sig)]
    df = pd.DataFrame(rows)
    for c in FEATURE_COLS:
        if c not in df:
            df[c] = np.nan
    df = df[['file', 'sig', 'n_notes', 'duration_s', 'mode', 'error'] + FEATURE_COLS]
    allrows = pd.concat([cache, df], ignore_index=True) if len(cache) else df
    allrows = allrows.drop_duplicates(['file', 'sig'], keep='last')
    allrows.to_csv(cpath, index=False)
    # always return the values AS STORED in the cache (round-trip parse), so a fresh extraction and a
    # cached re-run feed bit-identical numbers to the models
    back = pd.read_csv(cpath, float_precision='round_trip')
    back['sig'] = back['sig'].astype(str)
    back = back.set_index(['file', 'sig'])
    df = back.loc[list(zip(rel, sig))].reset_index()
    df = df[['file', 'sig', 'n_notes', 'duration_s', 'mode', 'error'] + FEATURE_COLS]
    df[FEATURE_COLS] = df[FEATURE_COLS].apply(pd.to_numeric, errors='coerce')
    return df.reset_index(drop=True)


# ----------------------------------------------------------------------------- nominal pitch + f0 audit
_PC = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}


def sci_to_midi(name):
    """'C4', 'D#4', 'Ds4', 'Ab1', 'Bb0' -> MIDI (C4 = 60).  Same rule as figures/phaseo_lib.py."""
    import re
    m = re.fullmatch(r'([A-G])(#|s|b)?(-?\d)', str(name))
    if not m:
        return None
    return 12 * (int(m.group(3)) + 1) + _PC[m.group(1)] + {'#': 1, 's': 1, 'b': -1, None: 0}[m.group(2)]


def helmholtz_to_midi(tok):
    """CCMusic Helmholtz token -> MIDI ('c1' = C4, 'C' = C2, 'C1' = C1, 'A2' = A0, 'c1s-d1b' = C#4).
    Same rule as figures/phaseo_lib.py (used for the figure's f0 check)."""
    import re
    t = str(tok).split('-')[0]
    m = re.fullmatch(r'([A-Ga-g])(\d)?(s)?', t)
    if not m:
        return None
    letter, num, sharp = m.group(1), m.group(2), m.group(3)
    octv = (2 - int(num) if num else 2) if letter.isupper() else (3 + int(num) if num else 3)
    return 12 * (octv + 1) + _PC[letter.upper()] + (1 if sharp else 0)


def nominal_midi(path):
    """Nominal pitch of an isolated-note file from its name (None if not parseable)."""
    import re
    b = os.path.basename(str(path))
    p = str(path).replace('\\', '/')
    if 'ccmusic_pianos' in p:
        m = re.fullmatch(r'(.+?)_([A-Ga-g]\d?s?(?:-[A-Ga-g]\d?b)?)_(train|validation|test)_\d+\.wav', b)
        return helmholtz_to_midi(m.group(2)) if m else None
    for rx in (r'([A-G]s?\d)_v\d+\.wav',                  # SoundFont renders, Otey
               r'([A-G]#?\d)v\d+\.ogg',                    # Salamander
               r'Piano\.(?:pp|mf|ff)\.([A-G]b?\d)\.aiff?',  # Iowa
               r'([A-G]#?\d)\.(?:pp|p|mf|f|ff|sf)\.wav',   # K18
               r'([A-G]#?\d)\.wav'):                       # FreePats
        m = re.fullmatch(rx, b)
        if m:
            return sci_to_midi(m.group(1))
    return None


def f0_one(args) -> dict:
    """f0 audit of one isolated-note file + 'f0-clean' features.
    Segments = detune.segment.build_segments (the 47-feature pipeline's own note windows and f0).
    A segment is f0-valid if its f0 is within 1 semitone of the nominal pitch.  f0-clean features =
    aggregate_file over the f0-valid per-note descriptor sets only (tuning features unchanged)."""
    path, nom = args
    from detune.segment import load_mono, build_segments, freq_to_midi
    from quality.pipeline import note_raw_descriptors, aggregate_file
    out = {'nominal_midi': np.nan if nom is None else float(nom), 'n_seg': 0, 'n_seg_ok': 0,
           'primary_midi_est': np.nan, 'primary_err': np.nan, 'aligned': False, 'n_notes_clean': 0, 'error': ''}
    try:
        x, sr = load_mono(path)
        segs, _ = build_segments(x, sr)
        segs = [s for s in segs if not (s.f0 < 24.0 or s.f0 > 4500.0)]
        out['n_seg'] = len(segs)
        if nom is not None and segs:
            pr = max(segs, key=lambda s: s.energy)
            out['primary_midi_est'] = float(freq_to_midi(pr.f0))
            out['primary_err'] = out['primary_midi_est'] - nom
        err = [freq_to_midi(s.f0) - nom for s in segs] if nom is not None else []
        out['n_seg_ok'] = int(sum(abs(e) < 1.0 for e in err))
        _, notes, tuning = note_raw_descriptors(path)
        aligned = len(notes) == len(segs) and all(n['midi'] == s.midi for n, s in zip(notes, segs))
        out['aligned'] = bool(aligned)
        if nom is None:
            keep = []
        elif aligned:
            keep = [n for n, e in zip(notes, err) if abs(e) < 1.0]
        else:                                           # fall back to the rounded MIDI of each note
            keep = [n for n in notes if abs(n['midi'] - nom) < 1.0]
        out['n_notes_clean'] = len(keep)
        if keep:
            out.update(aggregate_file(keep, tuning, _trends()))
    except Exception as e:
        out['error'] = repr(e)[:200]
    return out


def f0_cached(paths, noms, cache_name, workers=8) -> pd.DataFrame:
    """Cached f0_one over files (key = relative path + size:mtime).  Returns rows in input order,
    values as stored (round-trip CSV parse)."""
    os.makedirs(FEAT_CACHE, exist_ok=True)
    cpath = os.path.join(FEAT_CACHE, cache_name + '.csv')
    rel = [os.path.relpath(p, ROOT).replace('\\', '/') for p in paths]
    sig = [_sig(p) for p in paths]
    cache = pd.read_csv(cpath, float_precision='round_trip') if os.path.exists(cpath) else pd.DataFrame()
    have = set()
    if len(cache):
        cache['sig'] = cache['sig'].astype(str)
        have = set(zip(cache['file'], cache['sig']))
    todo = [i for i, k in enumerate(zip(rel, sig)) if k not in have]
    new = []
    if todo:
        print(f'  [f0 audit] {cache_name}: {len(todo)} new file(s) (cached {len(paths) - len(todo)})', flush=True)
        args = [(paths[i], noms[i]) for i in todo]
        if workers > 1 and len(todo) > 4:
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=workers) as ex:
                res = list(ex.map(f0_one, args, chunksize=4))
        else:
            res = [f0_one(a) for a in args]
        new = [{'file': rel[i], 'sig': sig[i], **r} for i, r in zip(todo, res)]
    cols = ['file', 'sig', 'nominal_midi', 'n_seg', 'n_seg_ok', 'primary_midi_est', 'primary_err', 'aligned',
            'n_notes_clean', 'error'] + FEATURE_COLS
    nd = pd.DataFrame(new)
    for c in cols:
        if len(nd) and c not in nd:
            nd[c] = np.nan
    allrows = pd.concat([cache, nd[cols] if len(nd) else nd], ignore_index=True) if len(cache) else nd[cols]
    allrows = allrows.drop_duplicates(['file', 'sig'], keep='last')
    allrows.to_csv(cpath, index=False)
    back = pd.read_csv(cpath, float_precision='round_trip')
    back['sig'] = back['sig'].astype(str)
    back = back.set_index(['file', 'sig'])
    df = back.loc[list(zip(rel, sig))].reset_index()
    df[FEATURE_COLS] = df[FEATURE_COLS].apply(pd.to_numeric, errors='coerce')
    return df[cols].reset_index(drop=True)


def md5_of(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


# ----------------------------------------------------------------------------- formatting
def f3(x):
    return 'n/a' if x is None or not np.isfinite(x) else f'{x:.3f}'


def fci(r):
    if not np.isfinite(r.get('auc', np.nan)):
        return 'n/a'
    return f"{r['auc']:.3f} [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]"


def fpct(x):
    return 'n/a' if x is None or not np.isfinite(x) else f'{100 * x:.1f}%'


def md_table(header, rows):
    s = '| ' + ' | '.join(header) + ' |\n|' + '|'.join(['---'] * len(header)) + '|\n'
    for r in rows:
        s += '| ' + ' | '.join(str(c) for c in r) + ' |\n'
    return s
