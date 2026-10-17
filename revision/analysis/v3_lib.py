"""Protocol v3 -- shared helpers (after the independent code audit, Stattia/revision/audit/CODE_AUDIT.md).

Changes against v2 (protocol_v2.py / pv2_lib.py are not modified; v2 outputs stay in out/):
  A1  own-fold scoring: rows never seen in training (T2 sample-based kinds, library notes) are scored by
      the single model of their own fold (their group's fold in the fixed fold map), like every real row;
      the v2 rule (mean of the 5 fold models) is kept as a secondary column.
  A2  real class cleaned: the non-acoustic files of CODE_AUDIT B1 (19 clear + 2 probable), every file
      listed in audit/real_class_exclusions.csv (read at run time if present), byte-identical duplicates
      (md5; first copy in sorted order kept); real performances grouped by YouTube video id as well as by
      the v2 group (connected components), so a multi-piano comparison video sits in one fold.
  A3  SoundFont single-note renders with identical 47-feature vectors (velocity copies) deduplicated.
  A4  a comparison whose synthetic side is ONE group gets no general CI: the interval is printed as
      "real-side only".
Folds: the v2 greedy GroupKFold rule (seed 0) re-applied to the v3 groups; stored in out_v3/folds.csv.
"""
from __future__ import annotations

import os
import re
import zlib
import hashlib
import numpy as np
import pandas as pd

import pv2_lib as L
from pv2_lib import FEATURE_COLS, MED16, ROOT

HERE = os.path.dirname(os.path.abspath(__file__))
OUT3 = os.path.join(HERE, 'out_v3')
CACHE3 = os.path.join(OUT3, 'cache')
AUDIT = os.path.join(ROOT, 'Stattia', 'revision', 'audit')
IR_DIR = os.path.join(ROOT, 'audio', '_ir', 'mit_survey', 'Audio')

IQR16 = [c for c in FEATURE_COLS if c.endswith('_iqr')]
from quality.pipeline import BASS_FEATURE_COLS, TUNING   # noqa: E402
FS3 = {'all47': list(FEATURE_COLS), 'med16': list(MED16), 'only_A_med': list(MED16), 'only_A_iqr': IQR16,
       'only_B': list(BASS_FEATURE_COLS), 'only_C': list(TUNING)}

# CODE_AUDIT.md B1: clear (title says so) 19 + probable (Synthesia) 2
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
    if f.startswith('ccmusic_pianos/') or '_extra/' in f or '/' not in f:
        return os.path.join(ROOT, 'audio', f)
    return os.path.join(ROOT, f)


def _stem(f):
    return os.path.splitext(str(f).replace('\\', '/').split('/')[-1])[0]


def _md5_real(files):
    """md5 of every real-performance file that shares its byte size with another one (cached)."""
    os.makedirs(CACHE3, exist_ok=True)
    cp = os.path.join(CACHE3, 'real_md5.csv')
    cache = pd.read_csv(cp, dtype=str) if os.path.exists(cp) else pd.DataFrame(columns=['file', 'sig', 'md5'])
    have = {(r.file, r.sig): r.md5 for r in cache.itertuples()}
    sizes = {f: os.path.getsize(audio_path(f)) for f in files}
    by = pd.Series(sizes)
    cand = by[by.duplicated(keep=False)].index.tolist()
    out, new = {}, []
    for f in cand:
        sig = L._sig(audio_path(f))
        if (f, sig) not in have:
            have[(f, sig)] = L.md5_of(audio_path(f))
            new.append((f, sig))
        out[f] = have[(f, sig)]
    if new:
        pd.DataFrame([{'file': f, 'sig': s, 'md5': m} for (f, s), m in have.items()]).to_csv(cp, index=False)
    return out


def _video_ids(files):
    man = pd.read_csv(os.path.join(ROOT, 'audio', 'manifest.csv'))
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
    import protocol_v2 as P
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
    """Returns dict: own (out-of-fold for EVERY row incl. rows excluded from training = own-fold scoring),
    mean5 (v2 rule: rows excluded from training get the mean of the 5 fold models), and for each ext set
    'ext_own' (row scored by the model of ext_fold) and 'ext_mean5'."""
    X = df[cols].values.astype(float)
    y = df['y'].values
    fold = df['fold'].values
    trainable = np.ones(len(df), bool) if cond == 'T1' else ~df['kind'].isin(L.SB_KINDS).values
    if 'trainable' in df:
        trainable &= df['trainable'].values.astype(bool)
    spw = float(np.sum(y[trainable] == 0)) / max(1, int(np.sum(y[trainable] == 1)))
    own = np.full(len(df), np.nan)
    m5 = np.zeros(len(df))
    ext = ext or {}
    ep = {k: np.zeros((L.N_FOLDS, len(v))) for k, v in ext.items()}
    for k in range(L.N_FOLDS):
        tr = trainable & (fold != k)
        sets = [X] + [v[cols].values.astype(float) for v in ext.values()]
        _, preds = L.fit_predict(model, X[tr], y[tr], sets, spw=spw)
        te = fold == k
        own[te] = preds[0][te]
        m5 += preds[0] / L.N_FOLDS
        for j, name in enumerate(ext):
            ep[name][k] = preds[1 + j]
    out = {'own': own, 'mean5': np.where(trainable, own, m5), 'trainable': trainable}
    for name in ext:
        fo = np.asarray(ext_fold[name]).astype(int)
        out.setdefault('ext_own', {})[name] = ep[name][fo, np.arange(len(fo))]
        out.setdefault('ext_mean5', {})[name] = ep[name].mean(axis=0)
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


def cut(x, sr, keep_s, fade_s=0.1, pre_s=0.01):
    o = onset_index(x, sr)
    a = max(0, o - int(pre_s * sr))
    e = o + int((keep_s + fade_s) * sr)
    y = np.array(x[a:e], dtype=np.float64)
    f0 = o + int(keep_s * sr) - a                    # fade from onset + keep_s
    if f0 < len(y):
        n = len(y) - f0
        t = np.arange(n) / max(1, int(fade_s * sr))
        y[f0:] *= 0.5 * (1.0 + np.cos(np.pi * np.clip(t, 0, 1)))
    return y


def level_at(x, sr, t_after_onset, win_s=0.02):
    """frame RMS (dB re the loudest 20-ms frame) at t seconds after onset; -inf if beyond the file."""
    o = onset_index(x, sr)
    w = int(win_s * sr)
    n = len(x) // w
    rms = np.sqrt(np.mean(x[:n * w].reshape(n, w) ** 2, axis=1) + 1e-30)
    i = (o + int(t_after_onset * sr)) // w
    if i >= n:
        return -np.inf
    return float(20 * np.log10(rms[i] / rms.max()))


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


def c3(x, sr, relpath, snr_db=40.0):
    """measured room IR (MIT IR Survey) drawn per file + white noise at snr_db re the note RMS (first
    1.0 s after onset of the reverberant note).  Deterministic: RNG seeded by crc32(relpath)."""
    from scipy.signal import fftconvolve
    rng = np.random.default_rng(zlib.crc32(relpath.encode('utf-8')))
    irs = _ir_list()
    i = int(rng.integers(len(irs)))
    y = fftconvolve(x, _ir(i, sr))[:len(x)]
    y *= (np.max(np.abs(x)) + 1e-30) / (np.max(np.abs(y)) + 1e-30)
    o = onset_index(y, sr)
    seg = y[o:o + int(1.0 * sr)]
    rms = float(np.sqrt(np.mean(seg ** 2))) if len(seg) else float(np.sqrt(np.mean(y ** 2)))
    y = y + rng.standard_normal(len(y)) * rms * 10 ** (-snr_db / 20)
    return y, os.path.basename(irs[i])


def transform(x, sr, variant, relpath):
    info = {}
    if variant == 'orig':
        return x, info
    parts = variant.split('+')
    if 'c3' in parts:
        x, info['ir'] = c3(x, sr, relpath)
    if 'c1a' in parts:
        x = cut(x, sr, 1.2)
    if 'c1b' in parts:
        x = cut(x, sr, 4.0)
    return x, info


def extract_variant(args):
    """features of one audio file under one transform (load_mono wrapped as in level_test.py)."""
    path, variant, relpath = args
    import detune.segment as S
    import quality.pipeline as QP
    orig = S.__dict__.get('_v3_orig_load') or S.load_mono
    S._v3_orig_load = orig
    out = {'error': ''}
    try:
        x0, sr = orig(path)
        out['level_4s_db'] = level_at(x0, sr, 4.0)
        out['dur_after_onset'] = (len(x0) - onset_index(x0, sr)) / sr
        xt, inf = transform(x0, sr, variant, relpath)
        out.update(inf)

        def patched(p):
            return xt, sr
        S.load_mono = patched
        QP.load_mono = patched
        meta, notes, tuning = QP.note_raw_descriptors(path)
        out.update({'n_notes': meta['n_notes'], 'mode': meta['mode'],
                    **QP.aggregate_file(notes, tuning, L._trends())})
    except Exception as e:
        out['error'] = repr(e)[:200]
    finally:
        S.load_mono = orig
        QP.load_mono = orig
    return out


def extract_cached3(jobs, name, workers=8, log=print):
    """jobs: list of (path, variant, relpath); cache out_v3/cache/<name>.csv keyed by (relpath, sig, variant)."""
    os.makedirs(CACHE3, exist_ok=True)
    cp = os.path.join(CACHE3, name + '.csv')
    cache = pd.read_csv(cp, float_precision='round_trip') if os.path.exists(cp) else pd.DataFrame()
    if len(cache):
        cache['sig'] = cache['sig'].astype(str)
    have = set(zip(cache['file'], cache['sig'], cache['variant'])) if len(cache) else set()
    sigs = [L._sig(p) for p, _, _ in jobs]
    todo = [(j, s) for j, s in zip(jobs, sigs) if (j[2], s, j[1]) not in have]
    if todo:
        log(f'  [v3 extract] {name}: {len(todo)} new (cached {len(jobs) - len(todo)})')
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers) as ex:
            res = list(ex.map(extract_variant, [j for j, _ in todo], chunksize=4))
        new = pd.DataFrame([{'file': j[2], 'sig': s, 'variant': j[1], **r} for (j, s), r in zip(todo, res)])
        cache = pd.concat([cache, new], ignore_index=True) if len(cache) else new
        cache = cache.drop_duplicates(['file', 'sig', 'variant'], keep='last')
        cache.to_csv(cp, index=False)
        cache = pd.read_csv(cp, float_precision='round_trip')
        cache['sig'] = cache['sig'].astype(str)
    ci = cache.set_index(['file', 'sig', 'variant'])
    out = ci.loc[[(j[2], s, j[1]) for j, s in zip(jobs, sigs)]].reset_index()
    for c in FEATURE_COLS:
        if c not in out:
            out[c] = np.nan
    out[FEATURE_COLS] = out[FEATURE_COLS].apply(pd.to_numeric, errors='coerce')
    return out


def file_md5_list(paths):
    h = hashlib.md5()
    for p in paths:
        h.update(open(p, 'rb').read())
    return h.hexdigest()
