"""Protocol v2 -- detector analyses for the JAES revision.

    python Stattia/revision/analysis/run_all.py          (runs this + listening_v2, writes RESULTS_v2.md)

Corpus  = the 47-feature XGBoost corpus behind the paper (quality/cache/features_cal.csv,
          synth_features.csv, samplelib_features.csv -- identical to the released CSVs) with
          corrections: 25 non-acoustic files removed from the real class; the 14 old digital-piano
          demos regrouped by brand (KORG = Korg; Roland SC-55 = sound module, own group);
          + data_v2/*.csv picked up automatically when present.
Models  = XGBoost (train_realness.py hyper-parameters, n_jobs=1) and the logistic-regression
          baseline (realness_net.py settings); feature sets all47 and med16 (Group-A medians).
CV      = 5-fold grouped by instrument, fixed fold map (pv2_lib.assign_folds), stored in out/folds.csv.
T1      = paper protocol (all synthetic kinds labelled synthetic, in training).
T2      = sample-based kinds excluded from training; scored zero-shot by every fold model (mean).

Expected data_v2 schemas (all feature CSVs must be produced with the same single-file pipeline,
quality.pipeline.note_raw_descriptors + aggregate_file + quality/cache/register_trends_cal.json;
extract_features_v2.py does exactly that):
  digital_v2_features.csv : file, <47 FEATURE_COLS>, brand, model, sound_method[, group]
                            sound_method containing 'model' -> modelled_digital;
                            'sampl'/'pcm'/'rom' -> sampled_digital_performance; anything else -> digital_other
  sf2_perf_features.csv   : file, <47>, instrument[, group]   (instrument = SoundFont name; the
                            same SoundFont as a single-note render shares its CV group)
  otey_perf_features.csv  : file, <47>[, group]               (group defaults to the Otey engine group)
  old_demos_regroup.csv   : file, group[, brand, model, device_type, sound_method]
"""
from __future__ import annotations

import os
import re
import glob
import json
import time
import numpy as np
import pandas as pd

import pv2_lib as L
from pv2_lib import FEATURE_COLS, MED16, FEATSETS, OUT, ROOT, QCACHE, DATA_V2, KIT

from quality.grouping import group_key

# ============================================================================ corpus
OLD_DEMO_DEFAULT = [
    # filename prefix, brand, model, device_type, sound_method, note
    ('KORG_B2_', 'Korg', 'B2', 'digital_piano', 'sampling', ''),
    ('KORG_Liano_', 'Korg', 'Liano', 'digital_piano', 'sampling', ''),
    ('Korg_D1_', 'Korg', 'D1', 'digital_piano', 'sampling', ''),
    ('Korg_SP-280_', 'Korg', 'SP-280', 'digital_piano', 'sampling', ''),
    ('Kurzweil_MP10_', 'Kurzweil', 'MP10', 'digital_piano', 'sampling', ''),
    ('PRIVIA_PX-S3000_', 'Casio', 'Privia PX-S3000', 'digital_piano', 'sampling', ''),
    ('Roland_GP-3_', 'Roland', 'GP-3', 'digital_piano', 'sampling',
     'UNVERIFIED [general knowledge]: the 2022 Roland GP series is marketed with modelled piano sound; '
     'check and set sound_method in data_v2/old_demos_regroup.csv'),
    ('Roland_GP-6_', 'Roland', 'GP-6', 'digital_piano', 'sampling', 'as GP-3'),
    ('Roland_GP9M_', 'Roland', 'GP-9M', 'digital_piano', 'sampling', 'as GP-3'),
    ('sc55_gs', 'Roland', 'SC-55 Sound Canvas', 'sound_module', 'sampling', 'GM/GS sound module, not a digital piano'),
    ('sc55_short', 'Roland', 'SC-55 Sound Canvas', 'sound_module', 'sampling', 'GM/GS sound module, not a digital piano'),
    ('Yamaha_Clavinova_CLP-695', 'Yamaha', 'Clavinova CLP-695GP', 'digital_piano', 'sampling', ''),
    ('Yamaha_Clavinova_CLP-795', 'Yamaha', 'Clavinova CLP-795GP', 'digital_piano', 'sampling', ''),
    ('Yamaha_Clavinova_CLP-865', 'Yamaha', 'Clavinova CLP-865GP', 'digital_piano', 'sampling', ''),
]


def digital_kind(sound_method) -> str:
    s = str(sound_method).strip().lower()
    if 'model' in s:
        return 'modelled_digital'
    if any(k in s for k in ('sampl', 'pcm', 'rom')):
        return 'sampled_digital_performance'
    return 'digital_other'


def _old_demo_table(yt_files: list[str]) -> pd.DataFrame:
    user = os.path.join(DATA_V2, 'old_demos_regroup.csv')
    rows = []
    for f in yt_files:
        base = f.split('/')[-1]
        hit = next((d for d in OLD_DEMO_DEFAULT if base.startswith(d[0])), None)
        if hit is None:
            raise SystemExit(f'old demo without regroup entry: {f}')
        _, brand, model, dev, sm, note = hit
        grp = 'dig:' + L.norm_token(brand) + ('_sc55' if dev == 'sound_module' else '')
        rows.append({'file': f, 'brand': brand, 'model': model, 'device_type': dev,
                     'sound_method': sm, 'group': grp, 'note': note, 'source_of_entry': 'default (protocol_v2.py)'})
    t = pd.DataFrame(rows)
    t['model_level_id'] = ''
    if os.path.exists(user):
        # accepted schema: file (required) + any of group, brand, model, device_type, sound_method,
        # proposed_instrument, note.  Without a `group` column the CV group is the BRAND (models of one
        # brand share a sound engine, e.g. the three Clavinovas all play the Yamaha CFX/Boesendorfer
        # samples), with the SC-55 sound module as its own group; proposed_instrument is kept for reference.
        u = pd.read_csv(user)
        if 'file' not in u:
            raise SystemExit('data_v2/old_demos_regroup.csv needs a `file` column')
        u['_b'] = u['file'].astype(str).str.replace('\\', '/', regex=False).str.split('/').str[-1]
        for i, r in t.iterrows():
            m = u[u['_b'] == r['file'].split('/')[-1]]
            if len(m):
                m = m.iloc[0]
                for c in ('brand', 'model', 'device_type', 'sound_method'):
                    if c in u and pd.notna(m.get(c)):
                        t.loc[i, c] = m[c]
                if 'proposed_instrument' in u and pd.notna(m.get('proposed_instrument')):
                    t.loc[i, 'model_level_id'] = m['proposed_instrument']
                if 'note' in u and pd.notna(m.get('note')):
                    t.loc[i, 'note'] = str(m['note'])[:300]
                if 'group' in u and pd.notna(m.get('group')):
                    g = str(m['group'])
                    t.loc[i, 'group'] = g if g.startswith('dig:') else 'dig:' + L.norm_token(g)
                else:
                    sc = 'module' in str(t.loc[i, 'device_type']).lower()
                    t.loc[i, 'group'] = 'dig:' + L.norm_token(t.loc[i, 'brand']) + ('_sc55' if sc else '')
                t.loc[i, 'source_of_entry'] = 'data_v2/old_demos_regroup.csv'
    t['device_type'] = ['sound_module' if 'module' in str(d).lower() else 'digital_piano' for d in t['device_type']]
    return t


def _filter_valid(df):
    df[FEATURE_COLS] = df[FEATURE_COLS].apply(pd.to_numeric, errors='coerce')
    ok = df[FEATURE_COLS].notna().sum(axis=1) >= 30          # same rule as train_realness._load
    return df[ok].copy(), int((~ok).sum())


def build_corpus(corrected=True, log=print, digital_features='raw'):
    """Returns (df, info).  df columns: file, y, kind, subkind, grp, device_type, brand_model, model_key,
    capture, flags, base, + features.  digital_features='peaknorm' swaps the features of every digital-piano
    row (old demos + digital_v2) for the versions extracted from the -1 dBFS PCM-16 peak-normalised audio
    (data_v2/*_peaknorm.csv), i.e. the level convention of the real corpus."""
    info = {'digital_features': digital_features}
    rdf = pd.read_csv(os.path.join(QCACHE, 'features_cal.csv'))
    sdf = pd.read_csv(os.path.join(QCACHE, 'synth_features.csv'))
    ldf = pd.read_csv(os.path.join(QCACHE, 'samplelib_features.csv'))
    man = pd.read_csv(os.path.join(ROOT, 'audio', 'manifest.csv'))
    info['md5'] = {n: L.md5_of(os.path.join(QCACHE, n)) for n in
                   ('features_cal.csv', 'synth_features.csv', 'samplelib_features.csv', 'register_trends_cal.json')}

    is_m1 = rdf['file'].str.contains('m1_patches', na=False)
    real = rdf[~is_m1].copy()
    real['y'] = 0
    real['grp'] = [group_key(f, b) for f, b in zip(real['file'], real.get('brand', ''))]
    real['kind'] = np.where(real['file'].str.startswith('ccmusic_pianos/'), 'real_isolated', 'real_performance')
    real['subkind'] = np.where(real['file'].str.startswith('ccmusic_pianos/'), 'ccmusic',
                               np.where(real['file'].str.contains('/'), 'youtube_extra', 'manifest'))
    real['device_type'] = ''
    real['brand_model'] = ''
    real['model_key'] = ''

    nonac = man[(man['source'] != 'acoustic') & ~man['file'].str.startswith('m1_patches')].copy()
    info['nonacoustic'] = nonac
    if corrected:
        drop = real['file'].isin(set(nonac['file']))
        info['n_nonacoustic_dropped'] = int(drop.sum())
        real = real[~drop]

    parts = [real]
    # algorithmic single-note renders / Otey
    s = sdf.copy()
    s['y'] = 1
    s['device_type'] = ''
    s['brand_model'] = ''
    ren = s[s.synth_type == 'rendered'].copy()
    ren['kind'], ren['subkind'] = 'algo_single', 'soundfont_single'
    ren['grp'] = 'syn:sf2:' + ren['instrument'].map(L.norm_token) if corrected else 'syn:rendered:' + ren['instrument']
    ot = s[s.synth_type == 'otey'].copy()
    ot['kind'], ot['subkind'] = 'algo_single', 'otey_single'
    ot['grp'] = 'syn:otey' if corrected else 'syn:otey:Otey'
    yt = s[s.synth_type == 'yt'].copy()
    yt['kind'], yt['subkind'] = 'sampled_digital_performance', 'old_demo'
    if corrected:
        reg = _old_demo_table(list(yt['file']))
        info['old_demo_regroup'] = reg
        reg = reg.set_index('file')
        yt['grp'] = yt['file'].map(reg['group'])
        yt['device_type'] = yt['file'].map(reg['device_type'])
        yt['brand_model'] = yt['file'].map(reg['brand'] + ' ' + reg['model'])
        yt['model_key'] = yt['file'].map(reg['model_level_id'].where(reg['model_level_id'] != '', reg['brand'] + '_' + reg['model']))
        yt['kind'] = ['sound_module' if reg.loc[f, 'device_type'] == 'sound_module' else digital_kind(reg.loc[f, 'sound_method'])
                      for f in yt['file']]
        if digital_features == 'peaknorm':
            yt = _swap_peaknorm(yt, os.path.join(DATA_V2, 'old_demos_features_peaknorm.csv'), info, 'old_demos')
    else:
        yt['grp'] = 'syn:yt:' + yt['instrument']
    parts += [ren, ot, yt]
    m1 = rdf[is_m1].copy()
    m1['y'], m1['kind'], m1['subkind'], m1['grp'] = 1, 'm1', 'm1', 'syn:m1'
    m1['device_type'], m1['brand_model'] = '', 'Korg M1'
    lib = ldf.copy()
    lib['y'], lib['kind'] = 1, 'sampled_library'
    lib['subkind'] = 'library:' + lib['instrument']
    lib['grp'] = 'syn:lib:' + lib['instrument'] if corrected else 'syn:samplelib:' + lib['instrument']
    lib['device_type'], lib['brand_model'] = '', ''
    parts += [m1, lib]

    keep = ['file', 'y', 'kind', 'subkind', 'grp', 'device_type', 'brand_model', 'model_key', 'capture', 'flags'] + FEATURE_COLS
    for p_ in parts:
        p_['capture'] = ''
        p_['flags'] = ''
        if 'model_key' not in p_:
            p_['model_key'] = ''
    df = pd.concat([p[keep] for p in parts], ignore_index=True)
    df, nbad = _filter_valid(df)
    info['n_invalid_base'] = nbad
    df['base'] = True

    # ---------------- data_v2 (only for the corrected corpus)
    info['v2'] = {}
    if corrected:
        v2parts = []
        p = os.path.join(DATA_V2, 'digital_v2_features.csv' if digital_features == 'raw' else 'digital_v2_features_peaknorm.csv')
        if os.path.exists(p):
            d = pd.read_csv(p)
            _need(d, p)
            sm = d['sound_method'] if 'sound_method' in d else pd.Series([''] * len(d))
            if 'sound_method' not in d:
                log(f'  WARNING {p}: no sound_method column -> all rows go to digital_other')
            d['y'] = 1
            d['kind'] = [digital_kind(x) for x in sm]
            d['subkind'] = np.where(d['kind'] == 'modelled_digital', 'modelled_v2', 'digital_v2')
            if 'group' in d:
                d['grp'] = ['dig:' + L.norm_token(g) if not str(g).startswith('dig:') else g for g in d['group']]
            elif 'brand' in d:
                d['grp'] = 'dig:' + d['brand'].map(L.norm_token)
            else:
                raise SystemExit(f'{p}: needs a brand or group column')
            d['device_type'] = d['device_type'] if 'device_type' in d else 'digital_piano'
            d['brand_model'] = (d['brand'].astype(str) + ' ' + d['model'].astype(str)) if 'model' in d and 'brand' in d else ''
            d['model_key'] = d['instrument'].astype(str) if 'instrument' in d else d['brand_model']
            d, cap_info = _digital_v2_capture(d, log)
            info['v2']['digital_v2_features.csv'] = {'rows': len(d), 'sound_method': sm.astype(str).value_counts().to_dict(),
                                                     **cap_info}
            v2parts.append(d)
        p = os.path.join(DATA_V2, 'sf2_perf_features.csv')
        if os.path.exists(p):
            d = pd.read_csv(p)
            _need(d, p)
            d['y'], d['kind'], d['subkind'] = 1, 'algo_performance', 'sf2_perf'
            if 'group' in d:
                d['grp'] = d['group'].astype(str)
            elif 'instrument' in d:
                d['grp'] = 'syn:sf2:' + d['instrument'].map(L.norm_token)
            else:
                raise SystemExit(f'{p}: needs an instrument or group column')
            d['device_type'], d['brand_model'] = '', d.get('instrument', '')
            d['capture'] = ''
            d['flags'], d['model_key'] = '', ''
            known = set(df.loc[df.subkind == 'soundfont_single', 'grp'])
            info['v2']['sf2_perf_features.csv'] = {'rows': len(d), 'groups': int(d['grp'].nunique()),
                                                   'groups_shared_with_single_note_renders': int(d['grp'].isin(known).sum())}
            v2parts.append(d)
        p = os.path.join(DATA_V2, 'otey_perf_features.csv')
        if os.path.exists(p):
            d = pd.read_csv(p)
            _need(d, p)
            d['y'], d['kind'], d['subkind'] = 1, 'algo_performance', 'otey_perf'
            d['grp'] = d['group'].astype(str) if 'group' in d else 'syn:otey'
            d['device_type'], d['brand_model'] = '', ''
            d['capture'] = ''
            d['flags'], d['model_key'] = '', ''
            info['v2']['otey_perf_features.csv'] = {'rows': len(d)}
            v2parts.append(d)
        if v2parts:
            v2 = pd.concat([x[keep] for x in v2parts], ignore_index=True)
            v2, nbad2 = _filter_valid(v2)
            info['n_invalid_v2'] = nbad2
            v2['base'] = False
            df = pd.concat([df, v2], ignore_index=True)
    df = df.reset_index(drop=True)
    fmap = L.assign_folds(df.loc[df.base, 'grp'], df.loc[~df.base, 'grp'])
    df['fold'] = df['grp'].map(fmap).astype(int)
    return df, info


EXCLUDE_V2 = {'0Si1TBz9b6I': 'Casio PX-S1000 demo with EXTERNAL convolution reverb mixed in (manifest flag external_reverb_added)'}


def _swap_peaknorm(rows, path, info, tag):
    """replace the 47 features of `rows` by those of the peak-normalised extraction (matched on file)."""
    if not os.path.exists(path):
        raise SystemExit(f'peaknorm variant requested but {path} is missing')
    pn = pd.read_csv(path).drop_duplicates('file').set_index('file')
    rows = rows.copy()
    hit = rows['file'].isin(pn.index)
    for c in FEATURE_COLS:
        rows.loc[hit, c] = pd.to_numeric(rows.loc[hit, 'file'].map(pn[c]), errors='coerce').values
    info[f'peaknorm_{tag}'] = {'matched': int(hit.sum()), 'unmatched': int((~hit).sum())}
    return rows


def _digital_v2_capture(d, log=print):
    """capture (line / mic / unknown) for digital_v2 rows: the feature CSV's own `capture` column, else
    data_v2/digital_v2_manifest.csv joined on file (full relative path, then basename).  Rows the manifest
    marks excluded are dropped (counted in the returned info)."""
    info = {}
    d = d.copy()
    d['_b'] = d['file'].astype(str).str.replace('\\', '/', regex=False).str.split('/').str[-1]
    mp = os.path.join(DATA_V2, 'digital_v2_manifest.csv')
    man = pd.read_csv(mp, dtype=str) if os.path.exists(mp) else None
    if man is not None and 'file' in man:
        man['_b'] = man['file'].astype(str).str.replace('\\', '/', regex=False).str.split('/').str[-1]
        mb = man.drop_duplicates('_b').set_index('_b')
        if 'excluded' in man:
            exc = d['_b'].map(mb['excluded']).fillna('').astype(str).str.strip().str.lower()
            drop = exc.isin(['1', 'true', 'yes', 'y'])
            info['dropped_excluded_by_manifest'] = int(drop.sum())
            d = d[~drop.values].copy()
        if 'capture' not in d and 'capture' in man:
            d['capture'] = d['_b'].map(mb['capture'])
            info['capture_source'] = 'digital_v2_manifest.csv'
        fl = d['_b'].map(mb['flags']).fillna('') if 'flags' in man else pd.Series([''] * len(d), index=d.index)
        en = d['_b'].map(mb['effects_note']).fillna('') if 'effects_note' in man else pd.Series([''] * len(d), index=d.index)
        # reverb marker = onboard/external reverb flag OR the video description mentions reverb
        def _desc_reverb(e):
            parts = [q.strip().lower() for q in str(e).split('|')]
            return any(q.startswith('description') and ('reverb' in q or 'hall' in q) for q in parts)
        d['flags'] = [';'.join(x for x in (str(f), 'reverb_in_description' if _desc_reverb(e) else '') if x)
                      for f, e in zip(fl, en)]
        vid = d['_b'].map(mb['video_id']).fillna('') if 'video_id' in man else pd.Series([''] * len(d), index=d.index)
        ex = d['flags'].str.contains('external_reverb_added') | vid.isin(list(EXCLUDE_V2))
        info['excluded_protocol'] = {b: EXCLUDE_V2.get(v, 'external_reverb_added flag') for b, v in zip(d.loc[ex, '_b'], vid[ex])}
        d = d[~ex.values].copy()
    elif 'capture' in d:
        info['capture_source'] = 'digital_v2_features.csv'
    if 'flags' not in d:
        d['flags'] = ''
    if 'capture' not in d:
        d['capture'] = 'unknown'
        info['capture_source'] = 'none (all unknown)'
    d['capture'] = d['capture'].fillna('unknown').astype(str).str.strip().str.lower().replace({'': 'unknown'})
    info['capture'] = d['capture'].value_counts().to_dict()
    return d.drop(columns=['_b']), info


def _need(d, p):
    miss = [c for c in ['file'] + FEATURE_COLS if c not in d.columns]
    if miss:
        raise SystemExit(f'{p}: missing columns {miss[:6]}{"..." if len(miss) > 6 else ""}')


# ============================================================================ CV
def run_cv(df, model_kind, featset, cond, ext=None):
    """Returns (scores, ext_fold_preds).  scores: OOF for trainable rows; for rows excluded from
    training (T2 sample-based) the mean over all 5 fold models.  ext: {name: X} extra sets
    scored by every fold model -> {name: array (5, n)}."""
    cols = FEATSETS[featset]
    X = df[cols].values.astype(float)
    y = df['y'].values
    fold = df['fold'].values
    trainable = np.ones(len(df), bool) if cond == 'T1' else ~df['kind'].isin(L.SB_KINDS).values
    spw = float(np.sum(y[trainable] == 0)) / max(1, int(np.sum(y[trainable] == 1)))
    oof = np.full(len(df), np.nan)
    zs = np.zeros(len(df))
    ext = ext or {}
    ext_pred = {k: np.zeros((L.N_FOLDS, len(v))) for k, v in ext.items()}
    for k in range(L.N_FOLDS):
        tr = trainable & (fold != k)
        te = fold == k
        sets = [X[te], X] + [v[cols].values.astype(float) for v in ext.values()]
        _, preds = L.fit_predict(model_kind, X[tr], y[tr], sets, spw=spw)
        oof[te] = preds[0]
        zs += preds[1] / L.N_FOLDS
        for j, name in enumerate(ext):
            ext_pred[name][k] = preds[2 + j]
    scores = np.where(trainable, oof, zs)
    return scores, ext_pred


# ============================================================================ comparisons
def _comparisons(df):
    k, sk = df['kind'].values, df['subkind'].values
    dev = np.array(['sound_module' if 'module' in str(d).lower() else str(d) for d in df['device_type'].fillna('')])
    bm = df['brand_model'].fillna('').astype(str).values
    cap = df['capture'].fillna('').astype(str).values if 'capture' in df else np.array([''] * len(df))
    real = np.isin(k, L.REAL_KINDS)
    RI, RP = k == 'real_isolated', k == 'real_performance'
    old = sk == 'old_demo'
    v2 = np.isin(sk, ['digital_v2', 'modelled_v2'])
    SD = k == 'sampled_digital_performance'
    fl = df['flags'].fillna('').astype(str).values if 'flags' in df else np.array([''] * len(df))
    lt = np.array(['long_tails' in f for f in fl])
    rv = np.array(['reverb' in f for f in fl])
    C = [
        ('isolated', 'I1', 'real_isolated (CCMusic) vs algo_single (SoundFont renders + Otey)', RI, k == 'algo_single'),
        ('isolated', 'I1a', '... vs SoundFont single-note renders', RI, sk == 'soundfont_single'),
        ('isolated', 'I1b', '... vs Otey single notes', RI, sk == 'otey_single'),
        ('isolated', 'I2', 'real_isolated (CCMusic) vs sampled_library (14-note takes; unit mismatch, see per-note table)', RI, k == 'sampled_library'),
        ('performance', 'P1', 'real_performance vs algo_performance', RP, k == 'algo_performance'),
        ('performance', 'P1a', '... vs SoundFont performance renders (sf2_perf)', RP, sk == 'sf2_perf'),
        ('performance', 'P1b', '... vs Otey performance renders (otey_perf)', RP, sk == 'otey_perf'),
        ('performance', 'P2', 'real_performance vs sampled digital pianos (old demos + digital_v2)', RP, SD),
        ('performance', 'P2a', '... old demos only (SC-55 module excluded)', RP, old & SD),
        ('performance', 'P2d', '... digital_v2 only', RP, v2 & SD),
        ('performance', 'P2d-line', '... digital_v2, line capture', RP, v2 & SD & (cap == 'line')),
        ('performance', 'P2d-mic', '... digital_v2, microphone capture', RP, v2 & SD & (cap == 'mic')),
        ('performance', 'P2d-unk', '... digital_v2, capture unknown', RP, v2 & SD & (cap == 'unknown')),
        ('performance', 'P2d-lt', '... digital_v2, long_tails flag', RP, v2 & SD & lt),
        ('performance', 'P2d-nolt', '... digital_v2, no long_tails flag', RP, v2 & SD & ~lt),
        ('performance', 'P2d-rv', '... digital_v2, reverb stated (flag or description)', RP, v2 & SD & rv),
        ('performance', 'P2d-norv', '... digital_v2, no reverb statement', RP, v2 & SD & ~rv),
        ('performance', 'P3', 'real_performance vs modelled digital pianos (old GP-6/9M + digital_v2 Roland)', RP, k == 'modelled_digital'),
        ('performance', 'P3d', '... digital_v2 modelled only', RP, (k == 'modelled_digital') & (sk == 'modelled_v2')),
        ('performance', 'P4', 'real_performance vs Korg M1', RP, k == 'm1'),
        ('performance', 'P5', 'real_performance vs digital_other (Roland GP-3, hybrid / unclear)', RP, k == 'digital_other'),
        ('performance', 'P6', 'real_performance vs Roland SC-55 sound module (not a digital piano)', RP, k == 'sound_module'),
        ('pooled', 'X1', 'all real vs all algorithmic', real, np.isin(k, L.ALGO_KINDS)),
        ('pooled', 'X2', 'all real vs sample-based (libraries + sampled digital + M1)', real,
         np.isin(k, ['sampled_library', 'sampled_digital_performance', 'm1'])),
        ('pooled', 'X3', 'all real vs all synthetic', real, df['y'].values == 1),
        ('crossed', 'C1', 'real_performance vs algo_single (format-crossed)', RP, k == 'algo_single'),
        ('crossed', 'C2', 'real_isolated vs sampled_digital_performance (format-crossed)', RI, k == 'sampled_digital_performance'),
        ('crossed', 'C3', 'real_isolated vs Korg M1 (format-crossed)', RI, k == 'm1'),
    ]
    return C


def eval_comparisons(df, scores, tag):
    y, grp = df['y'].values, df['grp'].values
    rows = []
    for regime, cid, label, rm, sm in _comparisons(df):
        r = L.compare(y, scores, grp, rm, sm)
        rows.append({'run': tag, 'regime': regime, 'id': cid, 'comparison': label, **r})
    return rows


# ============================================================================ per-note library set
LIB_SPECS = {
    'Salamander': ('audio/ref_bright_yamaha_salamander/SalamanderGrandPianoV3_OggVorbis/ogg',
                   r'^([A-G]#?)(\d)v(\d+)\.ogg$', lambda m: m.group(3), '8'),
    'Iowa': ('audio/ref_top_grand_iowa_steinway', r'^Piano\.(pp|mf|ff)\.([A-G]b?)(\d)\.aiff?$',
             lambda m: m.group(1), 'mf'),
    'K18': ('audio/ref_dull_upright/K18', r'^([A-G]#?)(\d)\.(pp|p|mf|f|ff|sf)\.wav$', lambda m: m.group(3), 'mf'),
    'FreePats': ('audio/ref_bad_old_piano/PianoFB SFZ+WAV-20200401/samples', r'^([A-G]#?)(\d)\.wav$',
                 lambda m: 'single', 'single'),
}


def library_note_files():
    rows = []
    for lib, (d, rx, layer_of, mf) in LIB_SPECS.items():
        for f in sorted(os.listdir(os.path.join(ROOT, d))):
            m = re.match(rx, f)
            if not m:
                continue
            layer = layer_of(m)
            rows.append({'library': lib, 'path': os.path.join(ROOT, d, f), 'layer': layer, 'mf_layer': layer == mf})
    return pd.DataFrame(rows)


# ============================================================================ main
def main(log=print):
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    xgbv = L.check_xgboost()
    md = []
    key = {}

    df, info = build_corpus(corrected=True, log=log)
    log(f'corpus: {len(df)} rows, {df.grp.nunique()} groups')

    # ---------------- bookkeeping outputs
    nonac = info['nonacoustic'].copy()
    old_grp = {f: group_key(f, '') for f in nonac['file']}
    rdf_files = set(pd.read_csv(os.path.join(QCACHE, 'features_cal.csv'), usecols=['file'])['file'])
    nonac['in_features_cal'] = nonac['file'].isin(rdf_files)
    nonac[['file', 'source', 'rating', 'in_features_cal', 'url', 'description_uk']].to_csv(
        os.path.join(OUT, 'excluded_non_acoustic.csv'), index=False, encoding='utf-8')
    info['old_demo_regroup'].to_csv(os.path.join(OUT, 'old_demos_regroup_used.csv'), index=False)
    folds_csv = os.path.join(OUT, 'folds.csv')
    prev = pd.read_csv(folds_csv) if os.path.exists(folds_csv) else None
    df[['file', 'y', 'kind', 'subkind', 'grp', 'fold']].to_csv(folds_csv, index=False)
    fold_note = 'first run (no previous out/folds.csv)'
    if prev is not None:
        mm = prev.merge(df[['file', 'fold']], on='file', how='inner', suffixes=('_prev', ''))
        nd = int((mm['fold_prev'] != mm['fold']).sum())
        fold_note = f'compared with previous out/folds.csv: {len(mm)} common files, {nd} changed fold'

    # ---------------- composition table
    comp = []
    for (kd, sk), g in df.groupby(['kind', 'subkind'], sort=False):
        comp.append([kd, sk, len(g), g.grp.nunique(), int(g.y.iloc[0])])
    md.append('## 1. Corpus (protocol v2)\n\n')
    md.append(f"Built from `quality/cache/features_cal.csv`, `synth_features.csv`, `samplelib_features.csv` "
              f"(md5 {info['md5']['features_cal.csv'][:8]}, {info['md5']['synth_features.csv'][:8]}, "
              f"{info['md5']['samplelib_features.csv'][:8]}; identical to the released CSVs per facts_A). "
              f"Rows with fewer than 30 non-missing features are dropped, as in `train_realness._load` "
              f"({info['n_invalid_base']} dropped).\n\n")
    reg = info['old_demo_regroup']
    reg_txt = '; '.join(f"{g.split(':', 1)[1]} {len(x)} file(s) [{', '.join(f'{k}: {v}' for k, v in x['sound_method'].value_counts().items())}]"
                        for g, x in reg.groupby('group'))
    reg_src = ', '.join(sorted(set(reg['source_of_entry'])))
    md.append(f"Corrections: **{info['n_nonacoustic_dropped']} non-acoustic files removed from the real class** "
              f"(list: `out/excluded_non_acoustic.csv`); the 14 old digital-piano demos regrouped by brand, the Roland "
              f"SC-55 sound module as its own group (`out/old_demos_regroup_used.csv`; source: {reg_src}): {reg_txt}. "
              f"sound_method decides the kind: sample -> sampled_digital_performance, modelling -> modelled_digital, "
              f"anything else (e.g. hybrid_unclear) -> digital_other; all three are excluded from training under T2. data_v2: "
              f"{json.dumps(info['v2']) if info['v2'] else 'no feature files present in Stattia/revision/data_v2/ (rows marked n/a)'}.\n\n")
    md.append(L.md_table(['kind', 'subkind', 'files', 'CV groups', 'label'], comp))
    md.append(f"\nTotal {len(df)} files: real {int((df.y == 0).sum())} ({df[df.y == 0].grp.nunique()} groups), "
              f"synthetic {int((df.y == 1).sum())} ({df[df.y == 1].grp.nunique()} groups).\n\n")

    fl = []
    for k in range(L.N_FOLDS):
        g = df[df.fold == k]
        syn = sorted(set(g[g.y == 1].grp))
        fl.append([k, len(g), int((g.y == 0).sum()), int((g.kind == 'real_isolated').sum()),
                   g[g.kind == 'real_isolated'].grp.nunique(), ', '.join(s.split(':', 1)[1] for s in syn)])
    md.append('**Fixed fold assignment** (`out/folds.csv`; greedy GroupKFold rule, tie order = groups sorted by name '
              f'and permuted with seed 0; {fold_note}):\n\n')
    md.append(L.md_table(['fold', 'files', 'real files', 'CCMusic files', 'CCMusic groups', 'synthetic groups'], fl))

    # ---------------- format diagnostics
    iqr_cols = [c for c in FEATURE_COLS if c.endswith('_iqr')]
    bc_cols = [c for c in FEATURE_COLS if c not in MED16 and c not in iqr_cols]          # Groups B + C (15)
    fd, sig = [], {}
    for (kd, sk), g in df.groupby(['kind', 'subkind'], sort=False):
        all0 = float(np.mean((g[iqr_cols].values == 0).all(axis=1)))
        nanbc = float(g[bc_cols].isna().values.mean())
        sig[sk] = (all0, nanbc)
        fd.append([sk, len(g), f"{np.mean((g[iqr_cols].values == 0)):.2f}", f'{all0:.2f}', f'{nanbc:.2f}',
                   f"{g['global_scatter'].isna().mean():.2f}", f"{g['bass_fund_strength'].isna().mean():.2f}",
                   f"{g[FEATURE_COLS].isna().values.mean():.2f}"])
    md.append('\n**Format diagnostics** (why file format is visible to the model): share of the 16 IQR features '
              'exactly 0, share of files with ALL 16 IQR features = 0, NaN rate of the 15 Group B+C features, NaN rate of '
              '`global_scatter` and `bass_fund_strength`, overall NaN rate.\n\n')
    md.append(L.md_table(['subkind', 'files', 'IQR==0 (features)', 'all 16 IQR = 0 (files)', 'NaN Group B+C',
                          'NaN global_scatter', 'NaN bass_fund_strength', 'NaN all'], fd))
    md.append('\nFormat signature in one line: files with all 16 IQR features = 0 / NaN rate of Group B+C features -- ' +
              '; '.join(f'{sk} {100 * a:.0f}% / {100 * b:.0f}%' for sk, (a, b) in sig.items()) +
              '. A model can read "single-note file" from these two signatures alone; the 16 Group-A medians '
              '(feature set med16) contain neither, which is why every comparison is also reported for med16.\n')
    key['format_signature'] = sig

    # ---------------- per-note library set (extracted once, cached)
    libs = library_note_files()
    log(f'library note files: {len(libs)}')
    lf = L.extract_cached(list(libs['path']), 'library_notes')
    libs = pd.concat([libs.reset_index(drop=True), lf], axis=1)
    libs['grp'] = 'syn:lib:' + libs['library']
    libs['valid'] = (libs[FEATURE_COLS].notna().sum(axis=1) >= 30) & (libs['error'].fillna('') == '')

    # ---------------- re-extract every isolated-note corpus file: reproducibility + segmentation diagnostics
    iso_rows = df[df.kind.isin(['real_isolated', 'algo_single'])]
    iso_paths = [os.path.join(ROOT, 'audio', f) if f.startswith('ccmusic_pianos/') else os.path.join(ROOT, f)
                 for f in iso_rows['file']]
    rex = L.extract_cached(iso_paths, 'corpus_isolated_reextract')
    a = iso_rows[FEATURE_COLS].values.astype(float)
    b = rex[FEATURE_COLS].values.astype(float)
    both = np.isfinite(a) & np.isfinite(b)
    repro = {'files': len(iso_rows), 'max_abs_diff': float(np.max(np.abs(a[both] - b[both]))),
             'nan_mismatch': int((np.isfinite(a) != np.isfinite(b)).sum()),
             'errors': int((rex['error'].fillna('') != '').sum())}
    key['reextract'] = repro
    seg = []
    for name, modes, nn in ([(sk, rex.loc[(iso_rows.subkind == sk).values, 'mode'], rex.loc[(iso_rows.subkind == sk).values, 'n_notes'])
                             for sk in ('ccmusic', 'soundfont_single', 'otey_single')]
                            + [(f'library {lb} (all layers)', libs.loc[libs.library == lb, 'mode'], libs.loc[libs.library == lb, 'n_notes'])
                               for lb in LIB_SPECS]):
        seg.append([name, len(modes), f"{(modes == 'single/chord').mean():.2f}", f'{np.median(nn):.0f}',
                    f'{np.mean(nn):.1f}', f'{np.max(nn):.0f}', f"{(nn <= 1).mean():.2f}"])

    # ---------------- stimuli (features of the 30 clips and of their source files)
    key_df = pd.read_csv(os.path.join(KIT, 'answer_key.csv'))
    clip_paths = [os.path.join(KIT, 'web', c + '.wav') for c in key_df['clip']]
    src_paths = [os.path.join(ROOT, s) for s in key_df['src']]
    stim_clip = L.extract_cached(clip_paths, 'stimuli_clips')
    stim_src = L.extract_cached(src_paths, 'stimuli_sources')

    # ---------------- CV runs
    runs = {}
    rows = []
    libv = libs[libs['valid']].reset_index(drop=True)
    for mk in ('xgb', 'lr'):
        for fs in ('all47', 'med16'):
            for cond in ('T1', 'T2'):
                tag = f'{mk}_{fs}_{cond}'
                t1 = time.time()
                sc, ext = run_cv(df, mk, fs, cond, ext={'libnotes': libv})
                runs[tag] = (sc, ext)
                rows += eval_comparisons(df, sc, tag)
                log(f'  {tag}: {time.time() - t1:.1f}s')
    comp_df = pd.DataFrame(rows)
    comp_df.to_csv(os.path.join(OUT, 'comparisons.csv'), index=False)
    oof = df[['file', 'y', 'kind', 'subkind', 'grp', 'fold']].copy()
    for tag, (sc, _) in runs.items():
        oof[tag] = sc
    oof.to_csv(os.path.join(OUT, 'oof_scores.csv'), index=False, float_format='%.6f')

    def table_for(mk, fs):
        out = []
        for cid in comp_df[comp_df.run == f'{mk}_{fs}_T1']['id']:
            a = comp_df[(comp_df.run == f'{mk}_{fs}_T1') & (comp_df.id == cid)].iloc[0]
            b = comp_df[(comp_df.run == f'{mk}_{fs}_T2') & (comp_df.id == cid)].iloc[0]
            if a['n_syn'] == 0 or a['n_real'] == 0:
                out.append([cid, a['comparison'], '0 / 0', 'n/a (no data)', '', '', 'n/a', '', ''])
                continue
            single = ' (1 syn. group)' if a['g_syn'] == 1 else ''
            out.append([cid, a['comparison'],
                        f"{a['n_real']}/{a['g_real']} vs {a['n_syn']}/{a['g_syn']}{single}",
                        L.fci(a), L.fpct(a['det_syn']), L.f3(a['meanP_syn']),
                        L.fci(b), L.fpct(b['det_syn']), L.f3(b['meanP_syn'])])
        return out

    hdr = ['id', 'comparison', 'files/groups real vs syn', 'T1 AUC [95% CI]', 'T1 det.', 'T1 mean P',
           'T2 AUC [95% CI]', 'T2 det.', 'T2 mean P']
    md.append('\n## 2. Detector results, corpus level (out-of-fold)\n\n')
    md.append('T1 = paper protocol (every synthetic kind labelled synthetic and in training). '
              'T2 = sample-based kinds (libraries, digital demos, modelled digital, M1) excluded from training; '
              'their scores are the mean over the 5 fold models (zero-shot). Algorithmic and real rows are out-of-fold in both. '
              'AUC CI = stratified grouped bootstrap (real groups and synthetic groups resampled separately, B = 2000, seed 0). '
              '"det." = share of synthetic files with P(synthetic) > 0.5; "mean P" = mean P(synthetic) of the synthetic files. '
              'Real false-alarm rates are in `out/comparisons.csv` (`fa_real`).\n\n')
    if (df.kind == 'algo_performance').any():
        md.append('**Caveat for P1 (algorithmic performances):** the SoundFont and Otey performance renders are dry digital '
                  'audio (no room, microphone or codec), while the real performances are YouTube recordings. P1 removes the '
                  'single-note-vs-performance FORMAT confound but not the CHANNEL confound. Each performance render shares '
                  'the CV group of the same SoundFont\'s (or Otey\'s) single-note renders, so no engine is in both the '
                  'training and the test folds.\n\n')
    titles = {('xgb', 'all47'): '2.1 XGBoost, all 47 features (primary)',
              ('xgb', 'med16'): '2.2 XGBoost, 16 Group-A medians only (per-note evidence)',
              ('lr', 'all47'): '2.3 Logistic regression, all 47 features',
              ('lr', 'med16'): '2.4 Logistic regression, 16 Group-A medians only'}
    for (mk, fs), title in titles.items():
        md.append(f'### {title}\n\n')
        md.append(L.md_table(hdr, table_for(mk, fs)))
        md.append('\n')
    # false alarm summary
    fa = []
    for tag in runs:
        sc = runs[tag][0]
        for kd in L.REAL_KINDS:
            m = (df.kind == kd).values
            fa.append([tag, kd, int(m.sum()), L.fpct(np.mean(sc[m] > 0.5)), L.f3(np.mean(sc[m]))])
    md.append('**Real-class false alarms** (share of real files with P > 0.5, mean P):\n\n')
    md.append(L.md_table(['run', 'real kind', 'files', 'false alarms', 'mean P'], fa))
    # per-instrument / per-velocity detection of the algorithmic single notes
    alg = df[df.kind == 'algo_single'].copy()
    alg['unit'] = np.where(alg.subkind == 'otey_single',
                           'Otey v' + alg['file'].str.extract(r'_v(\d+)\.wav$')[0].fillna('?'),
                           alg['grp'].str.replace('syn:sf2:', 'SoundFont ', regex=False))
    pu = []
    for u, g in alg.groupby('unit'):
        ix = g.index.values
        pu.append([u, len(g), int(g.fold.iloc[0])] + [L.fpct(np.mean(runs[t][0][ix] > 0.5)) for t in
                                                       ('xgb_all47_T1', 'xgb_all47_T2', 'xgb_med16_T2')])
    dmd, dkey = digital_section(df, runs, info, log)
    md.append(dmd)
    key.update(dkey)
    md.append('\n**Algorithmic single notes, detection by SoundFont and by Otey velocity** (out-of-fold, P > 0.5; '
              'the listening test used Casio CA-110 at v100 and Otey at v127):\n\n')
    md.append(L.md_table(['unit', 'files', 'fold', 'XGB all47 T1', 'XGB all47 T2', 'XGB med16 T2'], pu))

    def kv(run, cid):
        r = comp_df[(comp_df.run == run) & (comp_df.id == cid)].iloc[0]
        return r
    for run in runs:
        for cid in comp_df[comp_df.run == run]['id']:
            r = kv(run, cid)
            if r['n_syn'] > 0 and r['n_real'] > 0:
                key[f'{run}:{cid}'] = r.to_dict()

    # ---------------- per-note isolated regime
    md.append('\n## 3. Per-note isolated regime (note-for-note comparison)\n\n')
    md.append('Feasibility: per-note descriptors are cached only for real files (`quality/cache/notes_cal/`, incl. the 668 CCMusic files); '
              'single-note renders, Otey notes and library notes have no per-note cache, but every one of them is a single-note '
              'audio file present locally, and the single-file pipeline (`note_raw_descriptors` + `aggregate_file`, '
              '`register_trends_cal.json`) re-extracts the stored corpus rows exactly (check below). '
              'So the evaluation is feasible: each library note file is featurised exactly like a CCMusic note file, a SoundFont '
              'render or an Otey render (one file = one note). CCMusic, SoundFont and Otey rows are the corpus rows (OOF scores '
              'from section 2); library notes are new rows, never seen in training: under **T1** each library note is scored by '
              'the fold model whose test fold holds that library (it was trained with the other three libraries\' 14-note takes '
              'labelled synthetic, as in the paper); under **T2** by the mean of the 5 fold models (no library in training). '
              '**I-only** = a detector trained only on isolated notes (CCMusic vs SoundFont renders + Otey) with the same fixed '
              'folds; library notes scored by the mean of its 5 fold models.\n\n')
    nlib = libs.groupby('library').agg(files=('path', 'size'), valid=('valid', 'sum'),
                                        mf=('mf_layer', 'sum'), dur=('duration_s', 'median'),
                                        notes=('n_notes', 'median')).reset_index()
    md.append(L.md_table(['library', 'note files', 'valid (>=30 features)', 'mf-equivalent layer files', 'median duration (s)',
                          'median segments found'],
                         [[r.library, r.files, int(r.valid), int(r.mf), f'{r.dur:.1f}', f'{r.notes:.0f}'] for r in nlib.itertuples()]))
    md.append('\nmf-equivalent layer = Salamander v8 (of 16), Iowa mf, K18 mf, FreePats single layer (the layers used in the '
              'listening test). CCMusic note files: median 4.3 s; SoundFont renders 1.8 s; Otey 3.0 s.\n\n')
    md.append(f"**Reproducibility of the single-file pipeline:** all {repro['files']} isolated-note corpus files (CCMusic, "
              f"SoundFont renders, Otey) were re-extracted from audio with the current code: max |difference| to the stored "
              f"CSV rows = {repro['max_abs_diff']:.1e} (CSV rounding), {repro['nan_mismatch']} missing-value mismatches, "
              f"{repro['errors']} errors.\n\n")
    seg.append(['listening clips (3 s)', len(stim_clip), f"{(stim_clip['mode'] == 'single/chord').mean():.2f}",
                f"{np.median(stim_clip['n_notes']):.0f}", f"{np.mean(stim_clip['n_notes']):.1f}",
                f"{np.max(stim_clip['n_notes']):.0f}", f"{(stim_clip['n_notes'] <= 1).mean():.2f}"])
    md.append('**Segmentation of single-note files** (the onset detector of `detune/segment.py` decides between "single/chord" '
              'mode, one window from the onset, and "piece" mode, several windows of at most 2.5 s; "segments" = per-note '
              'descriptor sets that enter the median/IQR). One audio file = one struck note in every row, but long library '
              'notes are often split into several windows, so their IQR features are less often 0:\n\n')
    md.append(L.md_table(['source', 'files', 'share single/chord mode', 'median segments', 'mean segments', 'max segments',
                          'share with <= 1 segment'], seg))
    md.append('\n')

    # I-only CV
    iso = df[df.kind.isin(['real_isolated', 'algo_single'])].reset_index(drop=True)
    iso_runs = {}
    for mk in ('xgb', 'lr'):
        for fs in ('all47', 'med16'):
            sc, ext = run_cv(iso, mk, fs, 'T1', ext={'libnotes': libv})
            iso_runs[f'{mk}_{fs}_Ionly'] = (sc, ext)

    lib_fold = {g: int(df.loc[df.grp == g, 'fold'].iloc[0]) for g in libv['grp'].unique()}
    pn_rows, pn_scores = [], libv[['library', 'path', 'layer', 'mf_layer', 'grp', 'n_notes', 'duration_s']].copy()
    pn_scores['path'] = [os.path.relpath(p, ROOT).replace('\\', '/') for p in pn_scores['path']]
    variants = []
    for mk in ('xgb', 'lr'):
        for fs in ('all47', 'med16'):
            variants += [(f'{mk}_{fs}_T1', 'T1'), (f'{mk}_{fs}_T2', 'T2'), (f'{mk}_{fs}_Ionly', 'Ionly')]
    for tag, mode in variants:
        if mode == 'Ionly':
            sc_c, ext = iso_runs[tag]
            base_df = iso
            lib_sc = ext['libnotes'].mean(axis=0)
        else:
            sc_c, ext = runs[tag]
            base_df = df
            if mode == 'T1':
                fi = libv['grp'].map(lib_fold).values
                lib_sc = ext['libnotes'][fi, np.arange(len(libv))]
            else:
                lib_sc = ext['libnotes'].mean(axis=0)
        pn_scores[tag] = lib_sc
        cc = (base_df.kind == 'real_isolated').values
        sets = [('CCMusic notes vs SoundFont renders', base_df.subkind.values == 'soundfont_single', None),
                ('CCMusic notes vs Otey notes', base_df.subkind.values == 'otey_single', None)]
        y0 = base_df['y'].values
        for name, msk, _ in sets:
            r = L.compare(y0, sc_c, base_df['grp'].values, cc, msk)
            pn_rows.append({'run': tag, 'comparison': name, **r})
        for lname, lmask in ([('all 4 libraries, mf layer', libv['mf_layer'].values),
                              ('all 4 libraries, all layers', np.ones(len(libv), bool))]
                             + [(f'{lb}, mf layer', (libv['library'] == lb).values & libv['mf_layer'].values)
                                for lb in LIB_SPECS]
                             + [(f'{lb}, all layers', (libv['library'] == lb).values) for lb in LIB_SPECS]):
            yy = np.concatenate([np.zeros(cc.sum()), np.ones(lmask.sum())])
            pp = np.concatenate([sc_c[cc], lib_sc[lmask]])
            gg = np.concatenate([base_df['grp'].values[cc], libv['grp'].values[lmask]])
            r = L.compare(yy, pp, gg, yy == 0, yy == 1)
            pn_rows.append({'run': tag, 'comparison': f'CCMusic notes vs library notes: {lname}', **r})
    pn = pd.DataFrame(pn_rows)
    pn.to_csv(os.path.join(OUT, 'pernote_comparisons.csv'), index=False)
    pn_scores.to_csv(os.path.join(OUT, 'pernote_library_scores.csv'), index=False, float_format='%.6f')

    order = ['CCMusic notes vs SoundFont renders', 'CCMusic notes vs Otey notes',
             'CCMusic notes vs library notes: all 4 libraries, mf layer',
             'CCMusic notes vs library notes: all 4 libraries, all layers'] + \
            [f'CCMusic notes vs library notes: {lb}, mf layer' for lb in LIB_SPECS] + \
            [f'CCMusic notes vs library notes: {lb}, all layers' for lb in LIB_SPECS]
    for mk, fs in (('xgb', 'all47'), ('xgb', 'med16'), ('lr', 'all47'), ('lr', 'med16')):
        md.append(f'### 3.{["xgb_all47", "xgb_med16", "lr_all47", "lr_med16"].index(mk + "_" + fs) + 1} '
                  f'{"XGBoost" if mk == "xgb" else "Logistic regression"}, {"all 47 features" if fs == "all47" else "16 Group-A medians"}\n\n')
        tb = []
        for name in order:
            cells = [name.replace('CCMusic notes vs ', '')]
            n_done = False
            for mode in ('T1', 'T2', 'Ionly'):
                r = pn[(pn.run == f'{mk}_{fs}_{mode}') & (pn.comparison == name)].iloc[0]
                if not n_done:
                    cells.append(f"{r['n_real']}/{r['g_real']} vs {r['n_syn']}/{r['g_syn']}")
                    n_done = True
                cells += [L.fci(r), L.fpct(r['det_syn'])]
            tb.append(cells)
        md.append(L.md_table(['CCMusic notes vs', 'files/groups', 'T1 AUC [CI]', 'T1 det.', 'T2 AUC [CI]', 'T2 det.',
                              'I-only AUC [CI]', 'I-only det.'], tb))
        md.append('\n')
        for name in order:
            for mode in ('T1', 'T2', 'Ionly'):
                key[f'pernote:{mk}_{fs}_{mode}:{name}'] = pn[(pn.run == f'{mk}_{fs}_{mode}') & (pn.comparison == name)].iloc[0].to_dict()
    # does the verdict depend on how many segments the segmenter found? (format / segmentation shortcut)
    nseg_iso = rex['n_notes'].values
    sub_iso = iso_rows['subkind'].values
    sc_iso = {tag: runs[tag][0][iso_rows.index.values] for tag in ('xgb_all47_T1', 'xgb_all47_T2', 'xgb_med16_T2')}
    sd = []
    srcs = [('CCMusic notes (real; share = false alarms)', sub_iso == 'ccmusic', nseg_iso, sc_iso),
            ('SoundFont renders', sub_iso == 'soundfont_single', nseg_iso, sc_iso),
            ('Otey notes', sub_iso == 'otey_single', nseg_iso, sc_iso),
            ('library notes, mf layer (zero-shot / held-out)', libv['mf_layer'].values, libv['n_notes'].values,
             {t: pn_scores[t].values for t in sc_iso}),
            ('library notes, all layers', np.ones(len(libv), bool), libv['n_notes'].values,
             {t: pn_scores[t].values for t in sc_iso})]
    for name, msk, nseg, scd in srcs:
        one = msk & (nseg <= 1)
        many = msk & (nseg >= 2)
        row = [name, f'{int(one.sum())} / {int(many.sum())}']
        for t in sc_iso:
            p = scd[t]
            row.append(f"{L.fpct(np.mean(p[one] > 0.5)) if one.any() else 'n/a'} / {L.fpct(np.mean(p[many] > 0.5)) if many.any() else 'n/a'}")
        sd.append(row)
        key[f'segdep:{name}'] = row
    md.append('**Does the verdict depend on segmentation?** Share of files with P(synthetic) > 0.5 among files in which '
              'the segmenter found 1 segment vs >= 2 segments (one segment => all 16 IQR features are 0):\n\n')
    md.append(L.md_table(['source', 'files: 1 seg / >=2 seg', 'XGB all47 T1: 1 seg / >=2', 'XGB all47 T2: 1 seg / >=2',
                          'XGB med16 T2: 1 seg / >=2'], sd))
    # segment-stratified AUC: CCMusic vs X with both sides restricted to the same segment stratum
    cc_m = sub_iso == 'ccmusic'
    strat = []
    for name, msk, nseg, scd in srcs[1:]:
        row = [name]
        for t in ('xgb_all47_T2', 'xgb_med16_T2'):
            for lo_, hi_ in ((0, 1), (2, 10 ** 6)):
                rm = cc_m & (nseg_iso >= lo_) & (nseg_iso <= hi_)
                sm_ = msk & (nseg >= lo_) & (nseg <= hi_)
                yy = np.r_[np.zeros(rm.sum()), np.ones(sm_.sum())]
                pp = np.r_[sc_iso[t][rm], scd[t][sm_]]
                gg = np.r_[iso_rows['grp'].values[rm],
                           (iso_rows['grp'].values if scd is sc_iso else libv['grp'].values)[sm_]]
                r = L.compare(yy, pp, gg, yy == 0, yy == 1)
                row.append(L.fci(r))
                key[f'segstrat:{name}:{t}:{lo_}'] = r
        strat.append(row)
    md.append('\nSegment-stratified AUC (CCMusic notes vs the source, both restricted to the same segment stratum; '
              f'CCMusic: {int((cc_m & (nseg_iso <= 1)).sum())} single-segment, {int((cc_m & (nseg_iso >= 2)).sum())} multi-segment files):\n\n')
    md.append(L.md_table(['CCMusic vs', 'XGB all47 T2, 1 seg', 'XGB all47 T2, >=2 seg', 'XGB med16 T2, 1 seg',
                          'XGB med16 T2, >=2 seg'], strat))
    md.append('\n')

    # I-only own real false alarms
    fa2 = []
    iso_out = iso[['file', 'y', 'subkind', 'grp', 'fold']].copy()
    for tag, (sc, _) in iso_runs.items():
        m = (iso.kind == 'real_isolated').values
        fa2.append([tag, int(m.sum()), L.fpct(np.mean(sc[m] > 0.5))])
        iso_out[tag] = sc
    iso_out.to_csv(os.path.join(OUT, 'oof_scores_isolated_only.csv'), index=False, float_format='%.6f')
    md.append('I-only detector, CCMusic false alarms (P > 0.5): ' + '; '.join(f'{a} {c}' for a, _, c in fa2) + '.\n')
    ccf = df[df.kind == 'real_isolated'].groupby('fold').grp.nunique().to_dict()
    md.append(f'CCMusic groups per fold: {ccf} (folds without CCMusic contribute only synthetic OOF scores).\n')

    # ---------------- f0 audit + register / f0 sensitivity of the isolated regime
    f0_md, f0_key = f0_section(df, runs, iso_rows, iso_paths, libv, pn_scores, lib_fold, key_df, clip_paths)
    md.append(f0_md)
    key.update(f0_key)

    # ---------------- stimuli, out-of-sample
    stim_md, stim_key = score_stimuli(df, key_df, stim_clip, stim_src)
    md.append(stim_md)
    key.update(stim_key)

    # ---------------- effect of data_v2: same protocol on the base corpus only
    has_v2 = bool((~df.base).any())
    if has_v2:
        dfb = df[df.base].reset_index(drop=True)
        base_rows, base_lib = [], {}
        for fs in ('all47', 'med16'):
            for cond in ('T1', 'T2'):
                sc_b, ext_b = run_cv(dfb, 'xgb', fs, cond, ext={'libnotes': libv})
                base_rows += eval_comparisons(dfb, sc_b, f'xgb_{fs}_{cond}_baseonly')
                if cond == 'T1':
                    base_lib[f'xgb_{fs}_{cond}'] = (sc_b, ext_b['libnotes'][libv['grp'].map(lib_fold).values, np.arange(len(libv))])
                else:
                    base_lib[f'xgb_{fs}_{cond}'] = (sc_b, ext_b['libnotes'].mean(axis=0))
        bdf = pd.DataFrame(base_rows)
        bdf.to_csv(os.path.join(OUT, 'comparisons_base_only.csv'), index=False)
        md.append('\n## 4b. Effect of adding data_v2 (XGBoost; same folds)\n\n')
        md.append('Sections 2-4 train on the base corpus PLUS the data_v2 rows present (' +
                  ', '.join(f'{k}: {v.get("rows")} rows' for k, v in info['v2'].items()) +
                  '). Below, the same protocol on the base corpus only (data_v2 rows absent from training and evaluation).\n\n')
        tb = []
        mf = libv['mf_layer'].values
        for cid in ('I1a', 'I1b', 'I2', 'P2a', 'P4', 'X1', 'X2', 'C1', 'libnotes_mf'):
            row = [cid]
            for fs in ('all47', 'med16'):
                for cond in ('T1', 'T2'):
                    if cid == 'libnotes_mf':
                        sc_b, lsc = base_lib[f'xgb_{fs}_{cond}']
                        ccb = (dfb.kind == 'real_isolated').values
                        yy = np.r_[np.zeros(ccb.sum()), np.ones(mf.sum())]
                        rb = L.compare(yy, np.r_[sc_b[ccb], lsc[mf]], np.r_[dfb['grp'].values[ccb], libv['grp'].values[mf]],
                                       yy == 0, yy == 1, boot=False)
                        rv = key.get(f'pernote:xgb_{fs}_{cond}:CCMusic notes vs library notes: all 4 libraries, mf layer', {})
                    else:
                        rb = bdf[(bdf.run == f'xgb_{fs}_{cond}_baseonly') & (bdf.id == cid)].iloc[0]
                        rv = comp_df[(comp_df.run == f'xgb_{fs}_{cond}') & (comp_df.id == cid)].iloc[0]
                    row.append(f"{L.f3(rb['auc'])} -> {L.f3(rv['auc'])}")
            tb.append(row)
        md.append(L.md_table(['id', 'all47 T1 (base -> +v2)', 'all47 T2', 'med16 T1', 'med16 T2'], tb))
        for r in base_rows:
            key[f"baseonly:{r['run']}:{r['id']}"] = r

    # ---------------- continuity: uncorrected corpus, paper protocol
    dfu, _ = build_corpus(corrected=False)
    sc_u, _ = run_cv(dfu, 'xgb', 'all47', 'T1')
    ru = eval_comparisons(dfu, sc_u, 'xgb_all47_T1_uncorrected')
    rc = bdf[bdf.run == 'xgb_all47_T1_baseonly'] if has_v2 else comp_df[comp_df.run == 'xgb_all47_T1']
    md.append('\n## 5. Continuity with the submitted paper (XGBoost, all 47, T1)\n\n')
    md.append('"Uncorrected" = the paper\'s 4227-file corpus and groups (25 non-acoustic files kept as real, demos grouped by '
              'case-sensitive filename token), run with this protocol\'s fixed folds and n_jobs=1. The paper reported '
              '0.964 (all), 0.978 [0.957, 0.995] (algorithmic), 0.806 [0.727, 0.911] (sample-based), with unstored, '
              'platform-dependent folds; facts_A §4.3 shows 0.960-0.965 / 0.975-0.978 / 0.783-0.827 across tie orders.\n\n')
    tb = []
    for cid in ('X3', 'X1', 'X2', 'I1', 'I2', 'P2', 'P4', 'C1'):
        a = next(r for r in ru if r['id'] == cid)
        b = rc[rc.id == cid].iloc[0]
        tb.append([cid, a['comparison'], f"{a['n_real']} vs {a['n_syn']}", L.fci(a), f"{b['n_real']} vs {b['n_syn']}", L.fci(b)])
    md.append(L.md_table(['id', 'comparison', 'files (uncorrected)', 'AUC uncorrected', 'files (corrected, no data_v2)',
                          'AUC corrected, no data_v2'], tb))
    for cid in ('X1', 'X2', 'X3'):
        key[f'uncorrected:{cid}'] = next(r for r in ru if r['id'] == cid)

    # ---------------- fold-seed sensitivity (point AUCs only)
    md.append('\n### 5.1 Sensitivity to the fold assignment (XGBoost, all 47)\n\n')
    md.append('The fold map above uses seed 0. Point AUCs (no bootstrap) with the same rule and seeds 0-4; '
              'per-note library rows as in section 3 (mf layer).\n\n')
    sens_rows = []
    ids = ['I1a', 'I1b', 'I2', 'P2a', 'P4', 'X1', 'X2', 'C1']
    for seed in range(5):
        fm = L.assign_folds(df.loc[df.base, 'grp'], df.loc[~df.base, 'grp'], seed=seed)
        ds = df.copy()
        ds['fold'] = ds['grp'].map(fm).astype(int)
        for cond in ('T1', 'T2'):
            sc, ext = run_cv(ds, 'xgb', 'all47', cond, ext={'libnotes': libv})
            y0, g0 = ds['y'].values, ds['grp'].values
            rec = {'seed': seed, 'cond': cond}
            for regime, cid, label, rm, sm in _comparisons(ds):
                if cid in ids:
                    rec[cid] = L.compare(y0, sc, g0, rm, sm, boot=False)['auc']
            if cond == 'T1':
                lf = {g: int(ds.loc[ds.grp == g, 'fold'].iloc[0]) for g in libv['grp'].unique()}
                lsc = ext['libnotes'][libv['grp'].map(lf).values, np.arange(len(libv))]
            else:
                lsc = ext['libnotes'].mean(axis=0)
            cc = (ds.kind == 'real_isolated').values
            mfm = libv['mf_layer'].values
            rec['libnotes_mf'] = L.auc_fast(np.r_[np.zeros(cc.sum()), np.ones(mfm.sum())], np.r_[sc[cc], lsc[mfm]])
            sens_rows.append(rec)
    sens = pd.DataFrame(sens_rows)
    sens.to_csv(os.path.join(OUT, 'fold_seed_sensitivity.csv'), index=False, float_format='%.4f')
    tb = []
    for cid in ids + ['libnotes_mf']:
        row = [cid]
        for cond in ('T1', 'T2'):
            v = sens.loc[sens.cond == cond, cid].values
            row.append(f'{v[0]:.3f} ({v.min():.3f}-{v.max():.3f})')
        tb.append(row)
        key[f'seedsens:{cid}'] = {c: sens.loc[sens.cond == c, cid].tolist() for c in ('T1', 'T2')}
    md.append(L.md_table(['id', 'T1: seed 0 (range seeds 0-4)', 'T2: seed 0 (range seeds 0-4)'], tb))

    md.append(f'\n(protocol_v2 runtime {time.time() - t0:.0f} s; xgboost {xgbv})\n')
    return ''.join(md), key, info


# ============================================================================ digital pianos (data_v2)
DIG_IDS = ['P2', 'P2a', 'P2d', 'P2d-line', 'P2d-mic', 'P2d-unk', 'P2d-lt', 'P2d-nolt', 'P2d-rv', 'P2d-norv',
           'P3', 'P3d', 'P5', 'P6']


def digital_section(df, runs, info, log=print):
    """Real performances vs digital pianos: brand-grouped CIs (primary) and model-grouped CIs (sensitivity),
    capture / flag splits, and raw vs peak-normalised (-1 dBFS PCM-16) digital features."""
    key = {}
    comps = {c[1]: c for c in _comparisons(df)}
    ids = [i for i in DIG_IDS if i in comps and comps[i][4].any()]
    if not ids:
        return '', key
    y, grp = df['y'].values, df['grp'].values
    dig = df['kind'].isin(['sampled_digital_performance', 'modelled_digital', 'digital_other', 'sound_module']).values
    grp_model = np.where(dig, 'dig:model:' + df['model_key'].fillna('').map(L.norm_token).values, grp)
    md = ['\n## 2b. Digital pianos (old demos + data_v2)\n\n']
    nb = lambda m: int(len(set(grp[m])))
    nm = lambda m: int(len(set(grp_model[m])))
    md.append('CV groups and the primary bootstrap use the BRAND for every digital piano (engines are shared within a '
              'brand; the SC-55 module is its own group); "CI by model" re-runs the same bootstrap with one group per '
              'model (sensitivity; same scores). Excluded: ' +
              '; '.join(f'{k_}: {v}' for k_, v in (info.get('v2', {}).get('digital_v2_features.csv', {})
                                                    .get('excluded_protocol') or {}).items()) +
              '. T2 = digital pianos never in training (mean of the 5 fold models); T1 = in training with label '
              'synthetic (brand-grouped folds). Capture and flags come from `data_v2/digital_v2_manifest.csv` '
              '(old demos have no capture information).\n\n')
    rows, csv = [], []
    for cid in ids:
        _, _, label, rm, sm = comps[cid]
        cells = [cid, label, f'{int(sm.sum())} / {nb(sm)} / {nm(sm)}']
        for tag in ('xgb_all47_T2', 'xgb_med16_T2', 'xgb_all47_T1', 'xgb_med16_T1'):
            sc = runs[tag][0]
            r = L.compare(y, sc, grp, rm, sm)
            rmod = L.compare(y, sc, grp_model, rm, sm)
            cells += [L.fci(r), f"[{rmod['ci_lo']:.3f}, {rmod['ci_hi']:.3f}]", L.fpct(r['det_syn'])]
            key[f'dig:{tag}:{cid}'] = {**r, 'ci_model': (rmod['ci_lo'], rmod['ci_hi'])}
            csv.append({'run': tag, 'id': cid, 'comparison': label, 'n_brands': nb(sm), 'n_models': nm(sm), **r,
                        'ci_lo_by_model': rmod['ci_lo'], 'ci_hi_by_model': rmod['ci_hi']})
        rows.append(cells)
    hdr = ['id', 'comparison', 'files / brands / models']
    for tag in ('XGB all47 T2', 'XGB med16 T2', 'XGB all47 T1', 'XGB med16 T1'):
        hdr += [f'{tag} AUC [CI by brand]', 'CI by model', 'det.']
    md.append(L.md_table(hdr, rows))
    rpm = (df.kind == 'real_performance').values
    md.append(f'\nReal performances: {int(rpm.sum())} files; their false-alarm rate is in section 2.\n')
    md.append('\nBy brand (digital_v2 sample-based, XGB all47 T2): ' + ', '.join(
        f"{b.split(':', 1)[1]} {int(m.sum())} files AUC {L.compare(y, runs['xgb_all47_T2'][0], grp, rpm, m, boot=False)['auc']:.2f}"
        for b in sorted(set(grp[comps['P2d'][4]])) for m in [comps['P2d'][4] & (grp == b)]) + '.\n')

    # ---- raw vs peak-normalised digital features
    dp, ip = build_corpus(digital_features='peaknorm', log=log)
    assert (dp['file'].values == df['file'].values).all()
    pn_runs = {}
    for fs in ('all47', 'med16'):
        for cond in ('T1', 'T2'):
            pn_runs[f'xgb_{fs}_{cond}'] = run_cv(dp, 'xgb', fs, cond)[0]
    md.append('\n### 2b.1 Level convention: raw vs peak-normalised digital-piano audio\n\n')
    md.append('The real corpus was peak-normalised to -1 dBFS and stored as PCM-16; the digital-piano excerpts were '
              'analysed as float at their downloaded level. `data_v2/*_peaknorm.csv` are the same excerpts after '
              '-1 dBFS PCM-16 normalisation (all ' + str(int(dig.sum())) + ' digital rows change). Same folds and models; '
              'under T2 only the scored features change, under T1 the training rows change too.\n\n')
    rows = []
    for cid in ('P2', 'P2a', 'P2d', 'P2d-line', 'P2d-unk', 'P3', 'P6'):
        if cid not in ids:
            continue
        _, _, label, rm, sm = comps[cid]
        cells = [cid, f'{int(sm.sum())}']
        for tag in ('xgb_all47_T2', 'xgb_med16_T2', 'xgb_all47_T1', 'xgb_med16_T1'):
            a = L.compare(y, runs[tag][0], grp, rm, sm)
            b = L.compare(y, pn_runs[tag], grp, rm, sm)
            cells.append(f"{a['auc']:.3f} -> {L.fci(b)}; det {L.fpct(a['det_syn'])} -> {L.fpct(b['det_syn'])}")
            key[f'dig_pn:{tag}:{cid}'] = {'raw': a['auc'], 'peaknorm': b['auc'], 'ci_pn': (b['ci_lo'], b['ci_hi']),
                                         'det_raw': a['det_syn'], 'det_pn': b['det_syn']}
            csv.append({'run': tag + '_peaknorm', 'id': cid, 'comparison': label, **b})
        rows.append(cells)
    md.append(L.md_table(['id', 'files', 'XGB all47 T2: raw -> peaknorm [CI]', 'XGB med16 T2', 'XGB all47 T1', 'XGB med16 T1'], rows))
    # per-feature level effect on the digital rows
    q = df[FEATURE_COLS].quantile([0.25, 0.75])
    iqr = (q.loc[0.75] - q.loc[0.25]).replace(0, np.nan)
    dd = (dp.loc[dig, FEATURE_COLS].values - df.loc[dig, FEATURE_COLS].values) / iqr.values
    med = np.nanmedian(np.abs(dd), axis=0)
    order = np.argsort(-np.nan_to_num(med))[:10]
    md.append('\nLargest per-feature changes raw -> peaknorm on the digital rows (median |change| in units of the '
              "feature's inter-file IQR over the whole corpus): " +
              ', '.join(f'{FEATURE_COLS[j]} {med[j]:.2f}' for j in order) +
              f'. Median over all 47 features: {np.nanmedian(med):.3f} (controlled level test: section 8).\n')
    key['dig_pn_feature_change'] = {FEATURE_COLS[j]: float(med[j]) for j in range(len(FEATURE_COLS))}
    pd.DataFrame(csv).to_csv(os.path.join(OUT, 'digital_pianos.csv'), index=False)
    return ''.join(md), key


# ============================================================================ f0 audit + register sensitivity
def f0_section(df, runs, iso_rows, iso_paths, libv, pn_scores, lib_fold, key_df, clip_paths):
    md, key = [], {}
    noms_iso = [L.nominal_midi(p) for p in iso_paths]
    noms_lib = [L.nominal_midi(p) for p in libv['path']]
    noms_clip = [L.sci_to_midi(n) for n in key_df['note']]
    a_iso = L.f0_cached(list(iso_paths), noms_iso, 'f0_isolated')
    a_lib = L.f0_cached(list(libv['path']), noms_lib, 'f0_libnotes')
    a_clip = L.f0_cached(list(clip_paths), noms_clip, 'f0_clips')
    for a in (a_iso, a_lib, a_clip):
        a['f0_ok'] = a['primary_err'].abs() < 1.0
        a['band'] = pd.cut(a['nominal_midi'], [-1, 47, 76, 200], labels=['<48', '48-76', '>76']).astype(str)
    sub = iso_rows['subkind'].values
    out_audit = pd.concat([a_iso.assign(source=iso_rows['subkind'].values),
                           a_lib.assign(source='library:' + libv['library'].values),
                           a_clip.assign(source='listening clip')], ignore_index=True)
    out_audit.drop(columns=FEATURE_COLS).to_csv(os.path.join(OUT, 'f0_audit.csv'), index=False, float_format='%.4f')

    # ---- audit table
    md.append('\n### 3.6 f0 audit of the 47-feature pipeline on isolated notes, and register / f0 sensitivity\n\n')
    md.append('The figure analysis found that the phaseogram pipeline\'s f0 is off by >= 1 semitone for many isolated CCMusic notes '
              'outside MIDI 48-76. The 47-feature pipeline uses the same segmenter (`detune/segment.build_segments`): its '
              'per-segment f0 sets the partial grid for every per-note descriptor (partial energies, B, centroid/f0). '
              'Nominal pitch: CCMusic Helmholtz names / SoundFont, Otey and library note names (same parser as '
              '`figures/phaseo_lib.py`). "Primary" = the most energetic segment (the figure analysis\' criterion); '
              '"segments" = all per-note descriptor sets that enter a file\'s median/IQR (for a single strike the '
              'segmenter can return up to 3 pitch candidates of the same window). Failure = |estimated - nominal| >= 1 semitone.\n\n')
    srcs = [('CCMusic (real)', a_iso[sub == 'ccmusic']), ('SoundFont renders', a_iso[sub == 'soundfont_single']),
            ('Otey', a_iso[sub == 'otey_single'])] + \
           [(f'library {lb} (all layers)', a_lib[libv['library'].values == lb]) for lb in LIB_SPECS] + \
           [('listening clips', a_clip)]
    rows = []
    for name, a in srcs:
        cells = [name]
        for band in ('<48', '48-76', '>76', 'all'):
            g = a if band == 'all' else a[a['band'] == band]
            if not len(g):
                cells.append('-')
                continue
            cells.append(f"{int((~g['f0_ok']).sum())}/{len(g)} files; {int((g['n_seg'] - g['n_seg_ok']).sum())}/{int(g['n_seg'].sum())} seg.")
        rows.append(cells)
        key[f'f0audit:{name}'] = cells
    md.append(L.md_table(['source', 'nominal MIDI < 48', 'MIDI 48-76', 'MIDI > 76', 'all'], rows))
    cc = a_iso[sub == 'ccmusic']
    errs = cc.loc[~cc['f0_ok'], 'primary_err'].round().value_counts().head(6)
    md.append('\nCCMusic primary-f0 failures by size (semitones, rounded): ' +
              ', '.join(f'{int(k):+d}: {v}' for k, v in errs.items()) +
              f". CCMusic files with no f0-valid segment at all: {int((cc['n_notes_clean'] == 0).sum())} of {len(cc)}. "
              'Per-file details: `out/f0_audit.csv`.\n\n')

    # ---- f0-clean retrain
    dfc = df.copy()
    ix = iso_rows.index.values
    dfc.loc[ix, FEATURE_COLS] = a_iso[FEATURE_COLS].values
    validc = dfc[FEATURE_COLS].notna().sum(axis=1) >= 30
    dropped = {s: int(((~validc).values[ix] & (sub == s)).sum()) for s in ('ccmusic', 'soundfont_single', 'otey_single')}
    dfc = dfc[validc].copy()
    libc = libv[['library', 'mf_layer', 'grp']].copy()
    libc[FEATURE_COLS] = a_lib[FEATURE_COLS].values
    libc_ok = (libc[FEATURE_COLS].notna().sum(axis=1) >= 30).values
    clean_runs = {}
    for fs in ('all47', 'med16'):
        for cond in ('T1', 'T2'):
            sc, ext = run_cv(dfc.reset_index(drop=True), 'xgb', fs, cond, ext={'libnotes': libc[libc_ok]})
            clean_runs[f'xgb_{fs}_{cond}'] = (sc, ext)

    # ---- comparisons under restrictions
    ok_iso = a_iso['f0_ok'].values
    band_iso = (a_iso['band'] == '48-76').values
    ok_lib = a_lib['f0_ok'].values
    band_lib = (a_lib['band'] == '48-76').values
    restr = {'baseline': (np.ones(len(ix), bool), np.ones(len(libv), bool)),
             'f0-valid files': (ok_iso, ok_lib),
             'MIDI 48-76': (band_iso, band_lib),
             'f0-valid & 48-76': (ok_iso & band_iso, ok_lib & band_lib)}
    comps = [('SoundFont renders', sub == 'soundfont_single', None), ('Otey notes', sub == 'otey_single', None),
             ('algorithmic (both)', np.isin(sub, ['soundfont_single', 'otey_single']), None),
             ('library notes, mf layer', None, libv['mf_layer'].values),
             ('library notes, all layers', None, np.ones(len(libv), bool))]
    cc_m = sub == 'ccmusic'
    grp_iso = iso_rows['grp'].values
    res_rows = []
    for tag in ('xgb_all47_T2', 'xgb_all47_T1', 'xgb_med16_T2'):
        s_iso = runs[tag][0][ix]
        s_lib = pn_scores[tag].values
        for rname, (m_iso, m_lib) in restr.items():
            for cname, msyn_iso, msyn_lib in comps:
                rm = cc_m & m_iso
                if msyn_iso is not None:
                    sm = msyn_iso & m_iso
                    yy = np.r_[np.zeros(rm.sum()), np.ones(sm.sum())]
                    pp = np.r_[s_iso[rm], s_iso[sm]]
                    gg = np.r_[grp_iso[rm], grp_iso[sm]]
                else:
                    sm = msyn_lib & m_lib
                    yy = np.r_[np.zeros(rm.sum()), np.ones(sm.sum())]
                    pp = np.r_[s_iso[rm], s_lib[sm]]
                    gg = np.r_[grp_iso[rm], libv['grp'].values[sm]]
                r = L.compare(yy, pp, gg, yy == 0, yy == 1)
                res_rows.append({'run': tag, 'restriction': rname, 'comparison': cname, **r})
        # f0-clean features, retrained (+ restricted to 48-76)
        sc, ext = clean_runs[tag]
        dcr = dfc.reset_index(drop=True)
        isoc = dcr['kind'].isin(['real_isolated', 'algo_single']).values
        # map cleaned rows back to the audit (band) via file name
        band_map = dict(zip(iso_rows['file'].values, a_iso['band'].values))
        bandc = np.array([band_map.get(f, '') == '48-76' for f in dcr['file']])
        if tag.endswith('T1'):
            lsc = ext['libnotes'][libc.loc[libc_ok, 'grp'].map(lib_fold).values, np.arange(int(libc_ok.sum()))]
        else:
            lsc = ext['libnotes'].mean(axis=0)
        lib_mf = libv['mf_layer'].values[libc_ok]
        lib_band = band_lib[libc_ok]
        for rname, use_band in (('f0-clean features (retrained)', False), ('f0-clean features & 48-76', True)):
            for cname, _, _ in comps:
                rm = (dcr['subkind'].values == 'ccmusic') & (bandc if use_band else True)
                if cname.startswith('library'):
                    lm = (lib_mf if 'mf' in cname else np.ones(len(lsc), bool)) & (lib_band if use_band else True)
                    yy = np.r_[np.zeros(rm.sum()), np.ones(lm.sum())]
                    pp = np.r_[sc[rm], lsc[lm]]
                    gg = np.r_[dcr['grp'].values[rm], libc.loc[libc_ok, 'grp'].values[lm]]
                else:
                    want = {'SoundFont renders': ['soundfont_single'], 'Otey notes': ['otey_single'],
                            'algorithmic (both)': ['soundfont_single', 'otey_single']}[cname]
                    sm = np.isin(dcr['subkind'].values, want) & (bandc if use_band else True)
                    yy = np.r_[np.zeros(rm.sum()), np.ones(sm.sum())]
                    pp = np.r_[sc[rm], sc[sm]]
                    gg = np.r_[dcr['grp'].values[rm], dcr['grp'].values[sm]]
                r = L.compare(yy, pp, gg, yy == 0, yy == 1)
                res_rows.append({'run': tag, 'restriction': rname, 'comparison': cname, **r})
    res = pd.DataFrame(res_rows)
    res.to_csv(os.path.join(OUT, 'f0_sensitivity.csv'), index=False)
    rnames = list(restr) + ['f0-clean features (retrained)', 'f0-clean features & 48-76']
    md.append('Sensitivity of the isolated-regime comparisons (CCMusic notes vs ...). "f0-valid files" / "MIDI 48-76" '
              'restrict BOTH classes at evaluation (same out-of-fold / zero-shot scores as sections 2-3). '
              '"f0-clean features" re-aggregates every isolated-note file (CCMusic, SoundFont, Otey, library notes) from '
              f'its f0-valid segments only and retrains the CV with the same folds (files left without any f0-valid segment are '
              f'dropped: CCMusic {dropped["ccmusic"]}, SoundFont {dropped["soundfont_single"]}, Otey {dropped["otey_single"]}; '
              f'library notes {int((~libc_ok).sum())}). Library 14-note takes cannot be restricted this way and are not shown. '
              'Cells: AUC [95% CI] (CCMusic files vs synthetic files).\n\n')
    for tag, title in (('xgb_all47_T2', 'XGBoost all 47, T2'), ('xgb_all47_T1', 'XGBoost all 47, T1'),
                       ('xgb_med16_T2', 'XGBoost 16 Group-A medians, T2')):
        md.append(f'**{title}**\n\n')
        tb = []
        for rname in rnames:
            cells = [rname]
            for cname, _, _ in comps:
                r = res[(res.run == tag) & (res.restriction == rname) & (res.comparison == cname)].iloc[0]
                cells.append(f"{L.fci(r)} ({r['n_real']}/{r['n_syn']})")
                key[f'f0sens:{tag}:{rname}:{cname}'] = r.to_dict()
            tb.append(cells)
        md.append(L.md_table(['restriction'] + [c[0] for c in comps], tb))
        md.append('\n')
    return ''.join(md), key


# ============================================================================ stimuli
CLASS_OF = {'Iowa Steinway': 'Iowa', 'K18 upright': 'K18', 'Salamander (sampled)': 'Salamander',
            'soundfont': 'SoundFont (Casio CA-110)', 'Otey physical model': 'Otey'}
REC = ['Iowa', 'K18', 'Salamander']
ALG = ['SoundFont (Casio CA-110)', 'Otey']


def score_stimuli(df, key_df, stim_clip, stim_src):
    """Out-of-sample detector scores for the 30 listening stimuli (T2 training population,
    leave-own-group-out)."""
    import xgboost as xgb
    st = key_df[['clip', 'source', 'note', 'src']].copy()
    st['cls'] = st['source'].map(CLASS_OF)
    st['super'] = np.where(st['cls'].isin(REC), 'recorded', 'algorithmic')
    otey_g = 'syn:otey'
    casio_g = 'syn:sf2:' + L.norm_token('Casio_CA_110_Piano_FM')
    assert casio_g in set(df.grp), casio_g
    trainable = df[~df.kind.isin(L.SB_KINDS)]
    excl = {'T2_full': [], 'T2_noOtey': [otey_g], 'T2_noCasio': [casio_g], 'T2_noOteyCasio': [otey_g, casio_g]}
    own = np.where(st['cls'] == 'Otey', 'T2_noOtey', np.where(st['cls'] == ALG[0], 'T2_noCasio', 'T2_full'))
    out = st.copy()
    out['n_notes_clip'] = stim_clip['n_notes'].values
    out['n_notes_src'] = stim_src['n_notes'].values
    for mk in ('xgb', 'lr'):
        for fs in ('all47', 'med16'):
            cols = FEATSETS[fs]
            preds = {}
            for name, ex in excl.items():
                tr = trainable[~trainable.grp.isin(ex)]
                spw = float((tr.y == 0).sum()) / max(1, int((tr.y == 1).sum()))
                _, (pc, ps) = L.fit_predict(mk, tr[cols].values.astype(float), tr['y'].values,
                                            [stim_clip[cols].values.astype(float), stim_src[cols].values.astype(float)], spw=spw)
                preds[name] = (pc, ps)
            idx = np.arange(len(st))
            out[f'{mk}_{fs}_clip_LOGO'] = [preds[o][0][i] for i, o in zip(idx, own)]
            out[f'{mk}_{fs}_src_LOGO'] = [preds[o][1][i] for i, o in zip(idx, own)]
            out[f'{mk}_{fs}_clip_noOteyCasio'] = preds['T2_noOteyCasio'][0]
    # old in-sample model (pre_hard) for the record
    m = xgb.XGBClassifier()
    m.load_model(os.path.join(QCACHE, 'versions', 'realness_xgb_pre_hard.json'))
    out['prehard_clip'] = m.predict_proba(stim_clip[FEATURE_COLS].values.astype(float))[:, 1]
    out['prehard_src'] = m.predict_proba(stim_src[FEATURE_COLS].values.astype(float))[:, 1]
    stored = {r['clip']: r['p_synth'] for r in json.load(open(os.path.join(KIT, 'critic_on_stimuli.json'), encoding='utf-8'))['per_clip']}
    out['prehard_src_stored'] = out['clip'].map(stored)
    out.to_csv(os.path.join(OUT, 'stimuli_detector_scores.csv'), index=False, float_format='%.6f')

    # ---- summary
    md = ['\n## 4. Listening-test stimuli scored OUT OF SAMPLE\n\n']
    md.append('Features were extracted from each of the 30 **clips the listeners heard** (`listening_kit/web/clip_XX.wav`, 3 s, '
              'onset-trimmed, RMS-normalised) with the single-file pipeline used for isolated-note files. Each clip is scored by '
              'an XGBoost (or LR) model fitted on the T2 training population (real + algorithmic; sample-based kinds absent) '
              'with the clip\'s own group removed: Otey clips by a model without the Otey engine, Casio CA-110 clips by a model '
              'without that SoundFont, library clips (Iowa, K18, Salamander) by the full T2 model (no library in training). '
              '"LOGO" = this leave-own-group-out scheme; "noOteyCasio" = one model without both groups for all 30 clips. '
              'Detection = P(synthetic) > 0.5.\n\n')
    max_diff = float(np.nanmax(np.abs(out['prehard_src'] - out['prehard_src_stored'])))
    variants = [('xgb_all47_clip_LOGO', 'XGB all47, clip, LOGO (primary)'),
                ('xgb_all47_clip_noOteyCasio', 'XGB all47, clip, one model w/o Otey+CA-110'),
                ('xgb_med16_clip_LOGO', 'XGB med16, clip, LOGO'),
                ('lr_all47_clip_LOGO', 'LR all47, clip, LOGO'),
                ('lr_med16_clip_LOGO', 'LR med16, clip, LOGO'),
                ('xgb_all47_src_LOGO', 'XGB all47, unprocessed source file, LOGO'),
                ('prehard_src_stored', 'OLD in-sample model (realness_xgb_pre_hard), source file, as in the paper'),
                ('prehard_clip', 'OLD in-sample model, clip')]
    rows, key = [], {}
    rec = (out['super'] == 'recorded').values
    oldreal = out['cls'].isin(['Iowa', 'K18']).values
    alg = ~rec
    sal = (out['cls'] == 'Salamander').values
    cls_arr = out['cls'].values

    def clip_ci(p, neg, pos, n=L.N_BOOT):
        """bootstrap over clips, resampled within each source class (seed 0)."""
        rng = np.random.default_rng(L.SEED)
        idx_by = {c: np.where(cls_arr == c)[0] for c in REC + ALG}
        v = np.empty(n)
        for b in range(n):
            ii = np.concatenate([rng.choice(ix, len(ix), replace=True) for ix in idx_by.values()])
            m = neg[ii] | pos[ii]
            v[b] = L.auc_fast(pos[ii][m].astype(int), p[ii][m])
        return np.nanpercentile(v, 2.5), np.nanpercentile(v, 97.5)

    for col, label in variants:
        p = out[col].values
        cnt = {c: f"{int((p[out.cls == c] > 0.5).sum())}/{int((out.cls == c).sum())}" for c in REC + ALG}
        a1 = L.auc_fast(alg[rec | alg].astype(int), p[rec | alg])
        lo1, hi1 = clip_ci(p, rec, alg)
        a2 = L.auc_fast(alg[oldreal | alg].astype(int), p[oldreal | alg])
        a3 = L.auc_fast(sal[oldreal | sal].astype(int), p[oldreal | sal])
        rows.append([label] + [cnt[c] for c in REC + ALG] + [f'{a1:.3f} [{lo1:.3f}, {hi1:.3f}]', f'{a2:.3f}', f'{a3:.3f}'])
        key[f'stim:{col}'] = {'counts': cnt, 'auc_rec_vs_alg': a1, 'auc_rec_vs_alg_ci': (lo1, hi1),
                              'auc_iowak18_vs_alg': a2, 'auc_sal_vs_iowak18': a3,
                              'meanP': {c: float(p[out.cls == c].mean()) for c in REC + ALG}}
    md.append(L.md_table(['scoring', 'Iowa det.', 'K18 det.', 'Salamander det.', 'SoundFont det.', 'Otey det.',
                          'AUC recorded(20) vs algorithmic(10) [95% CI, clips resampled within class]',
                          'AUC Iowa+K18(14) vs algorithmic(10)', 'AUC Salamander(6) vs Iowa+K18(14)'], rows))
    md.append(f'\nThe stored paper values (`critic_on_stimuli.json`) are reproduced by `realness_xgb_pre_hard.json` on the '
              f'source files to max |dP| = {max_diff:.4f}. That model was fitted on all 4227 corpus rows: the 10 algorithmic '
              f'source files are training rows, and Iowa/K18/Salamander were in training as synthetic 14-note takes.\n\n')
    pc = ['clip', 'cls', 'note', 'xgb_all47_clip_LOGO', 'xgb_med16_clip_LOGO', 'lr_all47_clip_LOGO', 'xgb_all47_src_LOGO',
          'prehard_src_stored', 'n_notes_clip']
    md.append('Per-clip scores (P(synthetic)):\n\n')
    md.append(L.md_table(['clip', 'class', 'note', 'XGB all47 LOGO', 'XGB med16 LOGO', 'LR all47 LOGO', 'XGB all47 src LOGO',
                          'old in-sample (paper)', 'segments found in clip'],
                         [[r['clip'], r['cls'], r['note']] + [f'{r[c]:.3f}' for c in pc[3:8]] + [int(r['n_notes_clip'])]
                          for _, r in out.iterrows()]))
    return ''.join(md), key


if __name__ == '__main__':
    text, key, info = main()
    print(text)
