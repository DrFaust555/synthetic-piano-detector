"""Corpus of the final protocol and cross-validation with own-fold scoring.

  * every row never used in training (sample-based sources, library notes) is scored by the single model
    of its own fold, like every real row; the mean of the five fold models is kept as a second column;
  * the real class is cleaned: non-acoustic files, the files listed in
    data/protocol/real_class_exclusions.csv and byte-identical duplicates (md5) are removed; real
    performances are grouped by YouTube video id as well as by instrument (connected components), so a
    video with several pianos sits in one fold;
  * SoundFont single-note renders with identical feature vectors (velocity copies) are flagged;
  * when the synthetic side of a comparison is a single group, the interval resamples the real groups only.
Folds: the greedy grouped rule (seed 0) on these groups (data/protocol/folds.csv).
"""
from __future__ import annotations

import os
import re
import zlib
import hashlib
import numpy as np
import pandas as pd

import protocol.common as L
from protocol.common import FEATURE_COLS, MED16, ROOT

from detector.paths import DATA, RESULTS   # noqa: E402
HERE = os.path.dirname(os.path.abspath(__file__))
OUT3 = os.path.join(RESULTS, 'protocol')
CACHE3 = os.path.join(DATA, 'protocol')
AUDIT = os.path.join(DATA, 'protocol')
IR_DIR = os.path.join(ROOT, 'audio', '_ir', 'mit_survey', 'Audio')

IQR16 = [c for c in FEATURE_COLS if c.endswith('_iqr')]
from detector.pipeline import BASS_FEATURE_COLS, TUNING   # noqa: E402
FS3 = {'all47': list(FEATURE_COLS), 'med16': list(MED16), 'only_A_med': list(MED16), 'only_A_iqr': IQR16,
       'only_B': list(BASS_FEATURE_COLS), 'only_C': list(TUNING)}
# Isolated-note models use 27 features; performances use 47.
FS3_ISO = {**FS3, 'all47': list(L.ISO27)}

# Real-class screen: 19 non-acoustic sources identified by title and 2 probable Synthesia videos.
AUDIT_REAL_EXCLUDE = {
    'becker_PiNUjUzvoho': 'digital piano (Becker HP 66 Digital)',
    'clementi_g__pcziWU2A': 'digital piano (Kurzweil CUP2)',
    'kawai_KrK2nRrbxQQ': 'digital piano (Kawai DG30 Digital Grand)',
    'chase_dZI2xPuaxCo': 'digital piano (Chase P-55)',
    'bell_H_guQeAgQHo': 'digital piano (Dexibell VIVO H10MG)',
    'parsons_zjN9w6jmEfY': 'sample library (IK SampleTank Alan Parsons Imperial Grand)',
    'parsons_LoLX5RLFTlk': 'sample library (IK SampleTank Alan Parsons Imperial Grand)',
    'parsons_4d1ZFfnUwDY': 'sample library (IK SampleTank Alan Parsons Imperial Grand)',
    'parsons_qU9Fi1aqPl4': 'sample library (IK SampleTank Alan Parsons Imperial Grand)',
    'parsons_M5efrI6CbvY': 'sample library (IK SampleTank Alan Parsons Imperial Grand)',
    'ravenscroft_HKypM-VAVug': 'virtual instrument (Ravenscroft 275 iOS app)',
    'ravenscroft_0xCITWRZFk8': 'virtual instrument (Ravenscroft 275 iOS app)',
    'ravenscroft_AB4XlMedsM0': 'virtual instrument (Ravenscroft 275 iOS app)',
    'ravenscroft_pBoc0fyojRg': 'virtual instrument (Ravenscroft 275 iOS app)',
    'ravenscroft_jCZuTryAV58': 'virtual instrument (Ravenscroft 275 iOS app)',
    'ravenscroft_rpquLAhuJIg': 'virtual instrument (Ravenscroft 275 VI at NAMM)',
    'ravenscroft_lwAmNGzgqZo': 'virtual instrument (Ravenscroft 275 VI at NAMM)',
    'ravenscroft_wtY1go01gfo': 'virtual instrument (Ravenscroft 275 VI at NAMM)',
    'stuart_mVYCneSwxWg': 'electric piano ("Electric Piano Solo")',
    'parsons_vQNTtbrMZAQ': 'probable Synthesia video',
    'butler_Fw9OxOghZVo': 'probable Synthesia video',
}


def audio_path(f):
    if f.startswith('ccmusic_pianos/') or f.startswith('m1_patches/') or '_extra/' in f or '/' not in f:
        return os.path.join(ROOT, 'audio', f)
    return os.path.join(ROOT, f)


def _stem(f):
    return os.path.splitext(str(f).replace('\\', '/').split('/')[-1])[0]


_REAL_STAT = None


def _size_sig(f):
    """(byte size, size:mtime signature) of a real-performance file.  Where the audio is absent (the remote
    recompute machine holds only the isolated-note audio) both come from out_v3/cache/real_stat.csv, written
    by write_real_stat() on the machine that holds the corpus."""
    global _REAL_STAT
    p = audio_path(f)
    if os.path.exists(p):
        return os.path.getsize(p), L._sig(p)
    if _REAL_STAT is None:
        _REAL_STAT = pd.read_csv(os.path.join(CACHE3, 'real_stat.csv'), dtype=str).set_index('file')
    r = _REAL_STAT.loc[f]
    return int(r['size']), str(r['sig'])


def write_real_stat(files):
    """out_v3/cache/real_stat.csv: size, signature and duration (s) of every real-performance file
    (see _size_sig; level_test.sample reads the duration where the audio is absent)."""
    import soundfile as sf
    os.makedirs(CACHE3, exist_ok=True)
    rows = [{'file': f, 'size': os.path.getsize(audio_path(f)), 'sig': L._sig(audio_path(f)),
             'duration': sf.info(audio_path(f)).duration} for f in files]
    pd.DataFrame(rows).to_csv(os.path.join(CACHE3, 'real_stat.csv'), index=False)


def _md5_real(files):
    """md5 of every real-performance file that shares its byte size with another one (cached)."""
    os.makedirs(CACHE3, exist_ok=True)
    cp = os.path.join(CACHE3, 'real_md5.csv')
    cache = pd.read_csv(cp, dtype=str) if os.path.exists(cp) else pd.DataFrame(columns=['file', 'sig', 'md5'])
    have = {(r.file, r.sig): r.md5 for r in cache.itertuples()}
    stat = {f: _size_sig(f) for f in files}
    by = pd.Series({f: s[0] for f, s in stat.items()})
    cand = by[by.duplicated(keep=False)].index.tolist()
    out, new = {}, []
    for f in cand:
        sig = stat[f][1]
        if (f, sig) not in have:
            have[(f, sig)] = L.md5_of(audio_path(f))
            new.append((f, sig))
        out[f] = have[(f, sig)]
    if new:
        pd.DataFrame([{'file': f, 'sig': s, 'md5': m} for (f, s), m in have.items()]).to_csv(cp, index=False)
    return out


def _video_ids(files):
    man = pd.read_csv(os.path.join(DATA, 'protocol', 'manifest.csv'))
    mid = man.set_index('file')['url'].astype(str).str.extract(r'(?:v=|youtu\.be/|shorts/)([A-Za-z0-9_-]{11})')[0]
    out = {}
    for f in files:
        if '/' in f:                                   # *_extra: the file stem ends with the 11-char id
            out[f] = _stem(f)[-11:]
        else:
            v = mid.get(f)
            out[f] = v if isinstance(v, str) else None
    return out


def _uf_groups(df, mask):
    """connected components over (v2 group, video id) for the rows in mask."""
    parent = {}

    def find(a):
        while parent.setdefault(a, a) != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    files = df.loc[mask, 'file'].tolist()
    vids = _video_ids(files)
    gid = _screen_groups()
    for f, g in zip(files, df.loc[mask, 'grp']):
        find('g:' + g)
        if vids.get(f):
            union('g:' + g, 'v:' + vids[f])
        if gid.get(f):
            union('g:' + g, 'i:' + gid[f])
    return np.array([find('g:' + g) for g in df.loc[mask, 'grp']]), vids


def _screen_groups():
    p = os.path.join(AUDIT, 'real_class_exclusions.csv')
    if not os.path.exists(p):
        return {}
    t = pd.read_csv(p, dtype=str).fillna('')
    if 'group_id' not in t:
        return {}
    return {f: g for f, g in zip(t['file'], t['group_id']) if g}


def read_exclusion_csv(include_review=False):
    p = os.path.join(AUDIT, 'real_class_exclusions.csv')
    if not os.path.exists(p):
        return {}, 'audit/real_class_exclusions.csv not present at run time'
    t = pd.read_csv(p, dtype=str).fillna('')
    fcol = next((c for c in t.columns if c.lower() in ('file', 'path', 'relpath', 'filename')), t.columns[0])
    dcol = next((c for c in t.columns if c.lower() in ('exclude', 'decision', 'action', 'verdict', 'status')), None)
    if dcol is not None:
        v = t[dcol].str.strip().str.lower()
        vals = ['1', 'true', 'yes', 'y', 'exclude', 'excluded', 'drop', 'remove', 'non-acoustic', 'nonacoustic']
        keep = v.isin(vals + (['review'] if include_review else []))
        t = t[keep]
    reason_col = next((c for c in t.columns if c.lower() in ('category', 'reason', 'why', 'note')), None)
    return ({_stem(f): (t[reason_col].iloc[i] if reason_col else 'listed in real_class_exclusions.csv')
             for i, f in enumerate(t[fcol])},
            f'audit/real_class_exclusions.csv: {len(t)} file(s) with decision exclude' + (' or review' if include_review else ''))


def build_corpus_v3(log=print, include_review=False):
    """v2 corpus (incl. data_v2) + A2/A3 cleaning; returns (df, info)."""
    import protocol.corpus_base as P
    df, info = P.build_corpus(log=log)
    rep = {}
    real_perf = (df.kind == 'real_performance').values
    stems = df['file'].map(_stem)
    # A2a audit list
    hit = real_perf & stems.isin(list(AUDIT_REAL_EXCLUDE)).values
    rep['audit_B1'] = [(f, AUDIT_REAL_EXCLUDE[_stem(f)]) for f in df.loc[hit, 'file']]
    rep['audit_B1_missing'] = sorted(set(AUDIT_REAL_EXCLUDE) - set(stems[real_perf]))
    # A2b systematic screen
    exl, exl_note = read_exclusion_csv(include_review)
    hit2 = (df.y == 0).values & stems.isin(list(exl)).values & ~hit
    rep['screen'] = [(f, exl[_stem(f)]) for f in df.loc[hit2, 'file']]
    rep['screen_note'] = exl_note
    drop = hit | hit2
    # A2c byte-identical duplicates among real performances
    rp_files = df.loc[real_perf & ~drop, 'file'].tolist()
    md5 = _md5_real(rp_files)
    dups = pd.Series(md5).groupby(pd.Series(md5)).apply(lambda s: sorted(s.index)).tolist()
    dup_drop = [f for grp in dups if len(grp) > 1 for f in grp[1:]]
    rep['duplicates'] = [grp for grp in dups if len(grp) > 1]
    drop |= df['file'].isin(dup_drop).values
    # A3 SoundFont velocity copies
    sf = df[(df.subkind == 'soundfont_single') & ~drop].sort_values('file')
    keyv = sf[FEATURE_COLS].round(5).fillna(-9e9).astype(str).agg('|'.join, axis=1) + '#' + sf['grp']
    sf_dup = sf.index[keyv.duplicated(keep='first').values]
    rep['sf_dups'] = int(len(sf_dup))
    df['sf_dup'] = False
    df.loc[sf_dup, 'sf_dup'] = True
    df = df[~drop].copy()
    # A2d real performances grouped by video id (connected components with the v2 groups)
    rpm = (df.kind == 'real_performance').values
    comp, vids = _uf_groups(df, rpm)
    rep['n_real_groups_v2'] = int(df.loc[rpm, 'grp'].nunique())
    df.loc[rpm, 'grp'] = ['rp:' + c[2:] for c in comp]
    rep['n_real_groups_v3'] = int(df.loc[rpm, 'grp'].nunique())
    df = df.reset_index(drop=True)
    fmap = L.assign_folds(df.loc[df.base, 'grp'], df.loc[~df.base, 'grp'])
    df['fold'] = df['grp'].map(fmap).astype(int)
    info['v3'] = rep
    return df, info


# ----------------------------------------------------------------------------- CV with own-fold scoring
def run_cv3(df, cols, cond, ext=None, ext_fold=None, model='xgb'):
    """Out-of-fold scores; held-out sources are scored by their assigned fold model."""
    X = df[cols].values.astype(float)
    y = df['y'].values
    fold = df['fold'].values
    trainable = np.ones(len(df), bool) if cond == 'T1' else ~df['kind'].isin(L.SB_KINDS).values
    if 'trainable' in df:
        trainable &= df['trainable'].values.astype(bool)
    spw = float(np.sum(y[trainable] == 0)) / max(1, int(np.sum(y[trainable] == 1)))
    own = np.full(len(df), np.nan)
    ext = ext or {}
    ep = {k: np.zeros((L.N_FOLDS, len(v))) for k, v in ext.items()}
    for k in range(L.N_FOLDS):
        tr = trainable & (fold != k)
        sets = [X] + [v[cols].values.astype(float) for v in ext.values()]
        _, preds = L.fit_predict(model, X[tr], y[tr], sets, spw=spw)
        te = fold == k
        own[te] = preds[0][te]
        for j, name in enumerate(ext):
            ep[name][k] = preds[1 + j]
    out = {'own': own, 'trainable': trainable}
    for name in ext:
        fo = np.asarray(ext_fold[name]).astype(int)
        out.setdefault('ext_own', {})[name] = ep[name][fo, np.arange(len(fo))]
    return out


def cmp3(y, p, grp, rm, sm):
    r = L.compare(y, p, grp, rm, sm)
    r['one_syn_group'] = bool(r['g_syn'] == 1)
    return r


def fmt3(r, det=True, meanp=False, fa=False):
    if not np.isfinite(r.get('auc', np.nan)):
        return 'n/a'
    if r.get('one_syn_group'):
        s = f"{r['auc']:.3f} (real-side only [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}])"
    else:
        s = L.fci(r)
    if det:
        s += f"; det {L.fpct(r['det_syn'])}"
    if meanp:
        s += f"; P {r['meanP_syn']:.2f}"
    if fa:
        s += f"; FA {L.fpct(r['fa_real'])}"
    return s


# ----------------------------------------------------------------------------- audio transforms
def onset_index(x, sr, thresh_db=-50.0):
    win = max(1, sr // 1000)
    n = len(x) // win
    if n < 2:
        return 0
    rms = np.sqrt(np.mean(x[:n * win].reshape(n, win) ** 2, axis=1) + 1e-30)
    above = np.where(rms > rms.max() * 10 ** (thresh_db / 20))[0]
    return int(above[0] * win) if len(above) else 0


_IRS = {}


def _ir_list():
    import glob
    return sorted(glob.glob(os.path.join(IR_DIR, '*.wav')))


def _ir(i, sr):
    key = (i, sr)
    if key not in _IRS:
        import soundfile as sf
        from scipy.signal import resample_poly
        from math import gcd
        h, s0 = sf.read(_ir_list()[i])
        if h.ndim > 1:
            h = h.mean(axis=1)
        g = gcd(int(sr), int(s0))
        if s0 != sr:
            h = resample_poly(h, sr // g, s0 // g)
        _IRS[key] = h / (np.max(np.abs(h)) + 1e-12)
    return _IRS[key]


# source kinds of the performance format (used by every performance analysis)
PERF_KINDS = ['real_performance', 'algo_performance', 'sampled_digital_performance', 'modelled_digital', 'm1',
              'sound_module', 'digital_other']
