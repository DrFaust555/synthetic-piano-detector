"""Shared helpers of the evaluation protocol: paths, feature sets, models, fold assignment, AUC and
grouped bootstrap, and cached feature extraction.

Everything here is deterministic:
  * fold assignment = the greedy GroupKFold rule with a fixed tie order (groups sorted by name,
    permuted with numpy default_rng(seed), then stable-sorted by size);
  * XGBoost n_jobs=1, random_state=0;
  * bootstrap = numpy default_rng(0), re-seeded for every comparison.
Inputs are read from data/.  Feature tables in data/features/ are extended only when features of an
audio file that is not listed there are extracted.
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
from detector.paths import DATA, RESULTS, AUDIO_ROOT   # noqa: E402
ROOT = AUDIO_ROOT        # audio files are named relative to this folder (see detector/paths.py)
OUT = os.path.join(DATA, 'protocol', 'v2')   # fold map of the original corpus (read only)
FEAT_CACHE = os.path.join(DATA, 'features')  # per-file feature tables, keyed by file and size:mtime
QCACHE = os.path.join(DATA, 'corpus')
DATA_V2 = os.path.join(DATA, 'performances')
KIT = os.path.join(DATA, 'listening')


from detector.pipeline import FEATURE_COLS, DESCRIPTORS, BASS_FEATURE_COLS          # noqa: E402

MED16 = [f'{d}_med' for d in DESCRIPTORS]                          # Group A medians (features 1-16)

SEED = 0
N_FOLDS = 5
N_BOOT = 2000

# Isolated notes: one strike-anchored window at the nominal pitch (Sec. 3.3).
# Keep all 47 columns in feature tables, select these 27 for isolated-note models.
UNISON3 = ['unison_index', 'med_spread_cents', 'frac_split']
ISO27 = list(MED16) + list(BASS_FEATURE_COLS) + UNISON3


def iso_cache(name: str) -> str:
    """Name of a strike-anchored, nominal-pitch isolated-note feature table."""
    return name + '_iso1'


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


def nominal_f0(path):
    """Nominal fundamental (Hz, A4 = 440) of an isolated-note file from its name; None if not parseable."""
    m = nominal_midi(path)
    return None if m is None else 440.0 * 2.0 ** ((m - 69) / 12.0)


def note_raw_iso(QP, path: str, name: str | None = None, midi=None):
    """QP.note_raw_descriptors for an ISOLATED-NOTE file with one strike-anchored window.  Nominal pitch: `midi` when given
    (e.g. a listening clip, whose name carries no note), else parsed from `name` -- the file's original path
    when `path` is a transformed copy -- or from `path`.  QP is detector.pipeline (passed in so that callers
    which wrap its load_mono keep their wrapper)."""
    f0 = 440.0 * 2.0 ** ((midi - 69) / 12.0) if midi is not None and np.isfinite(midi) else nominal_f0(name or path)
    return QP.note_raw_descriptors(path, isolated=True, f0_hint=f0)


def extract_one(path: str, isolated: bool = False, midi=None) -> dict:
    """The single-file pipeline (detector.pipeline.note_raw_descriptors + aggregate_file,
    register_trends_cal.json).  isolated=True: the file holds one isolated note (see note_raw_iso)."""
    import detector.pipeline as QP
    from detector.pipeline import aggregate_file
    try:
        meta, notes, tuning = note_raw_iso(QP, path, midi=midi) if isolated else QP.note_raw_descriptors(path)
        feats = aggregate_file(notes, tuning, _trends())
        return {'n_notes': meta['n_notes'], 'duration_s': meta['duration_s'], 'mode': meta['mode'],
                'error': '', **feats}
    except Exception as e:                              # recorded, never silently dropped
        return {'n_notes': 0, 'duration_s': np.nan, 'mode': '', 'error': repr(e)[:200]}


_SIG_TABLE: dict = {}


def _sig(path):
    """Cache key of an audio file in the feature tables.  A file listed in the shipped tables
    (data/features/*.csv) keeps the key stored there (the size:mtime of the authors' copy), so every analysis
    runs from the tables, with or without the audio.  Other files, or all files when SPD_REEXTRACT=1, are keyed
    by their own size:mtime, which makes extract_cached() compute their features from the audio."""
    if not _SIG_TABLE:
        import glob
        for t in glob.glob(os.path.join(FEAT_CACHE, '*.csv')):
            c = pd.read_csv(t, usecols=lambda c: c in ('file', 'sig'), dtype=str)
            if {'file', 'sig'} <= set(c.columns):
                _SIG_TABLE.update(zip(c['file'], c['sig']))
    rel = os.path.relpath(path, ROOT).replace('\\', '/')
    if rel in _SIG_TABLE and os.environ.get('SPD_REEXTRACT') != '1':
        return _SIG_TABLE[rel]
    if os.path.exists(path):
        st = os.stat(path)
        return f'{st.st_size}:{int(st.st_mtime)}'
    return 'missing'


def _extract_one_iso(args) -> dict:
    path, midi = args
    return extract_one(path, isolated=True, midi=midi)


def extract_cached(paths: list[str], cache_name: str, workers: int = 8, isolated: bool = False,
                   midis=None) -> pd.DataFrame:
    """Extract features for absolute `paths`; cache rows in out/cache/<cache_name>.csv keyed by
    (relative path, size, mtime).  Returns a DataFrame in the order of `paths`.
    isolated=True: the files are isolated notes -- extracted with one strike-anchored window, cache name via iso_cache();
    `midis` = their nominal pitches when the file names do not carry them (listening clips)."""
    if isolated:
        cache_name = iso_cache(cache_name)
        mids = list(midis) if midis is not None else [None] * len(paths)
        fn, args = _extract_one_iso, list(zip(paths, mids))
    else:
        fn, args = extract_one, list(paths)
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
                res = list(ex.map(fn, [args[i] for i in todo], chunksize=4))
        else:
            res = [fn(args[i]) for i in todo]
        for i, r in zip(todo, res):
            have[(rel[i], sig[i])] = {'file': rel[i], 'sig': sig[i], **r}
    rows = [have[(f, s)] for f, s in zip(rel, sig)]
    df = pd.DataFrame(rows)
    fcols = FEATURE_COLS
    for c in fcols:
        if c not in df:
            df[c] = np.nan
    df = df[['file', 'sig', 'n_notes', 'duration_s', 'mode', 'error'] + fcols]
    allrows = pd.concat([cache, df], ignore_index=True) if len(cache) else df
    allrows = allrows.drop_duplicates(['file', 'sig'], keep='last')
    if todo:
        allrows.to_csv(cpath, index=False)
    # always return the values AS STORED in the cache (round-trip parse), so a fresh extraction and a
    # cached re-run feed bit-identical numbers to the models
    back = pd.read_csv(cpath, float_precision='round_trip')
    back['sig'] = back['sig'].astype(str)
    back = back.set_index(['file', 'sig'])
    df = back.loc[list(zip(rel, sig))].reset_index()
    df = df[['file', 'sig', 'n_notes', 'duration_s', 'mode', 'error'] + fcols]
    df[fcols] = df[fcols].apply(pd.to_numeric, errors='coerce')
    return df.reset_index(drop=True)


# ----------------------------------------------------------------------------- nominal pitch
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
    Same rule as figures/phaseo_lib.py (used for the figure agent's f0 check)."""
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
    m = re.fullmatch(r'MAPS_ISOL_NO_[PMF]_S0_M(\d+)_ENSTDk(?:Cl|Am)\.wav', b)    # MAPS (as in v4_lib)
    if m:
        return int(m.group(1))
    for rx in (r'([A-G]s?\d)_v\d+\.wav',                  # SoundFont renders, Otey
               r'([A-G]#?\d)v\d+\.ogg',                    # Salamander
               r'Piano\.(?:pp|mf|ff)\.([A-G]b?\d)\.aiff?',  # Iowa
               r'([A-G]#?\d)\.(?:pp|p|mf|f|ff|sf)\.wav',   # K18
               r'([A-G]#?\d)\.wav'):                       # FreePats
        m = re.fullmatch(rx, b)
        if m:
            return sci_to_midi(m.group(1))
    return None


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
