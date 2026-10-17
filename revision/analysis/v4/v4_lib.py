"""Protocol v4 (isolated regime) -- shared helpers.

Recorded-note class = CCMusic (8 pianos) + 4 free sample libraries (Salamander, Iowa, K18, FreePats)
+ MAPS ENSTDkCl / ENSTDkAm ISOL/NO notes (real Yamaha Disklavier, CC BY-NC-SA 2.0, see get_maps_isol.py).
Synthetic class = 10 SoundFont instruments (velocity copies removed, as v3) + Otey.
Nothing outside Stattia/revision/analysis/v4/ is written (feature caches go to v4/cache/).
"""
from __future__ import annotations

import os
import re
import sys
import zlib
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ANA = os.path.dirname(HERE)
if ANA not in sys.path:
    sys.path.insert(0, ANA)

import pv2_lib as L                               # noqa: E402
import v3_lib as V                                # noqa: E402
from pv2_lib import FEATURE_COLS, MED16, ROOT     # noqa: E402

CACHE4 = os.path.join(HERE, 'cache')
OUT4 = HERE
MAPS_DIR = os.path.join(ROOT, 'audio', '_maps_isol')
FS = {'all47': list(FEATURE_COLS), 'med16': list(MED16)}
N_AUG_COPIES = 2                                   # augmented copies per note (different IR / SNR each)
SNR_RANGE = (30.0, 50.0)


# ----------------------------------------------------------------------------- corpus (read-only inputs)
def _rel(p):
    return os.path.relpath(p, ROOT).replace('\\', '/')


def load_base():
    """v3 isolated rows (CCMusic, SoundFont w/o velocity copies, Otey) with the stored features, the
    four libraries (v2 cache, read only) and MAPS (v4 cache).  Returns one DataFrame with columns
    file (relative to ROOT), path, src, inst, grp, y, layer_mf, midi + features."""
    folds = pd.read_csv(os.path.join(ANA, 'out_v3', 'folds.csv'))
    iso = folds[folds.kind.isin(['real_isolated', 'algo_single']) & ~folds.sf_dup.astype(bool)].copy()
    rdf = pd.read_csv(os.path.join(ROOT, 'quality', 'cache', 'features_cal.csv'), float_precision='round_trip')
    sdf = pd.read_csv(os.path.join(ROOT, 'quality', 'cache', 'synth_features.csv'), float_precision='round_trip')
    feat = pd.concat([rdf[['file'] + FEATURE_COLS], sdf[['file'] + FEATURE_COLS]]).drop_duplicates('file', keep='first')
    iso = iso.merge(feat, on='file', how='left', validate='one_to_one')
    rows = []
    for _, r in iso.iterrows():
        p = V.audio_path(r.file)
        if r.subkind == 'ccmusic':
            inst, src = r.file.split('/')[-1].split('_')[0], 'CCMusic'
            grp = 'rec:cc:' + inst
        elif r.subkind == 'soundfont_single':
            inst, src, grp = r.file.split('/')[-2], 'SoundFont', r.grp
        else:
            inst, src, grp = 'Otey', 'Otey', r.grp
        rows.append({'file': _rel(p), 'path': p, 'src': src, 'inst': inst, 'grp': grp, 'grp_v3': r.grp,
                     'y': int(r.y), 'layer_mf': True, 'midi': L.nominal_midi(p), **{c: r[c] for c in FEATURE_COLS}})
    base = pd.DataFrame(rows)
    # libraries: the cached v2 extraction (read only), validity rule of followup_v2._libv
    import protocol_v2 as P
    libs = P.library_note_files()
    lc = pd.read_csv(os.path.join(ANA, 'out', 'cache', 'library_notes.csv'), float_precision='round_trip')
    lc = lc.drop_duplicates('file', keep='last').set_index('file')
    lr = []
    for _, r in libs.iterrows():
        f = _rel(r['path'])
        if f not in lc.index:
            continue
        x = lc.loc[f]
        if x[FEATURE_COLS].notna().sum() < 30 or str(x.get('error', '') if pd.notna(x.get('error', '')) else '') != '':
            continue
        lr.append({'file': f, 'path': r['path'], 'src': 'library', 'inst': r['library'], 'grp': 'rec:lib:' + r['library'],
                   'grp_v3': 'syn:lib:' + r['library'], 'y': 0, 'layer_mf': bool(r['mf_layer']),
                   'midi': L.nominal_midi(r['path']), **{c: float(x[c]) for c in FEATURE_COLS}})
    lib = pd.DataFrame(lr)
    maps = maps_features()
    return pd.concat([base, lib, maps], ignore_index=True)


def maps_files():
    rows = []
    for sub in ('ENSTDkCl', 'ENSTDkAm'):
        d = os.path.join(MAPS_DIR, sub)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            m = re.fullmatch(r'MAPS_ISOL_NO_([PMF])_S0_M(\d+)_(ENSTDk(?:Cl|Am))\.wav', f)
            if m:
                rows.append({'path': os.path.join(d, f), 'room': sub, 'dyn': m.group(1), 'midi': int(m.group(2))})
    return pd.DataFrame(rows)


def maps_features():
    mf = maps_files()
    if not len(mf):
        return pd.DataFrame()
    jobs = [(p, 'orig', _rel(p)) for p in mf['path']]
    fe = extract_cached4(jobs, 'maps_orig')
    out = pd.DataFrame({'file': [_rel(p) for p in mf['path']], 'path': mf['path'], 'src': 'MAPS',
                        'inst': 'MAPS ' + mf['room'], 'grp': 'rec:maps:disklavier', 'grp_v3': 'rec:maps:disklavier',
                        'y': 0, 'layer_mf': (mf['dyn'] == 'M').values, 'midi': mf['midi']})
    for c in FEATURE_COLS:
        out[c] = fe[c].values
    ok = (out[FEATURE_COLS].notna().sum(axis=1) >= 30) & (fe['error'].fillna('').astype(str) == '').values
    return out[ok].reset_index(drop=True)


# ----------------------------------------------------------------------------- augmentation + extraction
def channel(x, sr, relpath, k):
    """random measured room IR (MIT IR Survey) + white noise at SNR ~ U(30, 50) dB re the note RMS
    (first 1 s after onset of the reverberant note).  Deterministic per (note, copy k)."""
    from scipy.signal import fftconvolve
    rng = np.random.default_rng(zlib.crc32(f'v4aug{k}:{relpath}'.encode('utf-8')))
    irs = V._ir_list()
    i = int(rng.integers(len(irs)))
    snr = float(rng.uniform(*SNR_RANGE))
    y = fftconvolve(x, V._ir(i, sr))[:len(x)]
    y *= (np.max(np.abs(x)) + 1e-30) / (np.max(np.abs(y)) + 1e-30)
    o = V.onset_index(y, sr)
    seg = y[o:o + int(1.0 * sr)]
    rms = float(np.sqrt(np.mean(seg ** 2))) if len(seg) else float(np.sqrt(np.mean(y ** 2)))
    y = y + rng.standard_normal(len(y)) * rms * 10 ** (-snr / 20)
    return y, os.path.basename(irs[i]), snr


def extract4(args):
    path, variant, relpath = args
    import detune.segment as S
    import quality.pipeline as QP
    orig = S.__dict__.get('_v4_orig_load') or S.load_mono
    S._v4_orig_load = orig
    out = {'error': '', 'ir': '', 'snr_db': np.nan}
    try:
        x0, sr = orig(path)
        if variant == 'orig':
            xt = x0
        else:
            xt, out['ir'], out['snr_db'] = channel(x0, sr, relpath, int(variant[3:]))

        def patched(p):
            return xt, sr
        S.load_mono = patched
        QP.load_mono = patched
        meta, notes, tuning = QP.note_raw_descriptors(path)
        out.update({'n_notes': meta['n_notes'], **QP.aggregate_file(notes, tuning, L._trends())})
    except Exception as e:
        out['error'] = repr(e)[:200]
    finally:
        S.load_mono = orig
        QP.load_mono = orig
    return out


def extract_cached4(jobs, name, workers=14, log=print):
    """jobs: (path, variant, relpath); variant 'orig' or 'aug<k>'.  Cache v4/cache/<name>.csv keyed by
    (relpath, size:mtime, variant)."""
    os.makedirs(CACHE4, exist_ok=True)
    cp = os.path.join(CACHE4, name + '.csv')
    cache = pd.read_csv(cp, float_precision='round_trip') if os.path.exists(cp) else pd.DataFrame()
    if len(cache):
        cache['sig'] = cache['sig'].astype(str)
    have = set(zip(cache['file'], cache['sig'], cache['variant'])) if len(cache) else set()
    sigs = [L._sig(p) for p, _, _ in jobs]
    todo = [(j, s) for j, s in zip(jobs, sigs) if (j[2], s, j[1]) not in have]
    if todo:
        log(f'  [v4 extract] {name}: {len(todo)} new (cached {len(jobs) - len(todo)})', flush=True)
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers) as ex:
            res = list(ex.map(extract4, [j for j, _ in todo], chunksize=4))
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


def aug_features(df, log=print):
    """{k: DataFrame aligned with df} for the N_AUG_COPIES augmented copies of every note."""
    out = {}
    for k in range(N_AUG_COPIES):
        jobs = [(p, f'aug{k}', f) for p, f in zip(df['path'], df['file'])]
        out[k] = extract_cached4(jobs, 'iso_aug', log=log)
    return out
