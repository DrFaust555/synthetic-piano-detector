"""Corpus assembly, source groups, library note files and comparison masks used by the paper.

Corpus = the per-file 47-feature tables of the real recordings (data/corpus/features_cal.csv), the synthetic
         renders (synth_features.csv), the library takes (samplelib_features.csv) and the performance
         sources (data/performances/*.csv); the old digital-piano demos are regrouped by brand.
The corpus retains the source groups and ordering used for the published folds.
CV     = 5-fold grouped by instrument with a fixed fold map.
T1     = every synthetic kind in training.
T2     = sample-based kinds never trained, scored by the fold models.

Expected schemas of data/performances/ (all feature CSVs must be produced with the same single-file pipeline,
detector.pipeline.note_raw_descriptors + aggregate_file + data/corpus/register_trends_cal.json;
training/perf_features.py does exactly that):
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

import protocol.common as L
from protocol.common import FEATURE_COLS, ROOT, QCACHE, DATA_V2

from detector.grouping import group_key

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


def build_corpus(log=print):
    """Assemble the corrected corpus used by the paper, with fixed source groups."""
    info = {}
    rdf = pd.read_csv(os.path.join(QCACHE, 'features_cal.csv'))
    sdf = pd.read_csv(os.path.join(QCACHE, 'synth_features.csv'))
    ldf = pd.read_csv(os.path.join(QCACHE, 'samplelib_features.csv'))
    man = pd.read_csv(os.path.join(L.DATA, 'protocol', 'manifest.csv'))
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
    ren['grp'] = 'syn:sf2:' + ren['instrument'].map(L.norm_token)
    ot = s[s.synth_type == 'otey'].copy()
    ot['kind'], ot['subkind'] = 'algo_single', 'otey_single'
    ot['grp'] = 'syn:otey'
    yt = s[s.synth_type == 'yt'].copy()
    yt['kind'], yt['subkind'] = 'sampled_digital_performance', 'old_demo'
    reg = _old_demo_table(list(yt['file']))
    info['old_demo_regroup'] = reg
    reg = reg.set_index('file')
    yt['grp'] = yt['file'].map(reg['group'])
    yt['device_type'] = yt['file'].map(reg['device_type'])
    yt['brand_model'] = yt['file'].map(reg['brand'] + ' ' + reg['model'])
    yt['model_key'] = yt['file'].map(reg['model_level_id'].where(reg['model_level_id'] != '', reg['brand'] + '_' + reg['model']))
    yt['kind'] = ['sound_module' if reg.loc[f, 'device_type'] == 'sound_module' else digital_kind(reg.loc[f, 'sound_method'])
                  for f in yt['file']]
    parts += [ren, ot, yt]
    m1 = rdf[is_m1].copy()
    m1['y'], m1['kind'], m1['subkind'], m1['grp'] = 1, 'm1', 'm1', 'syn:m1'
    m1['device_type'], m1['brand_model'] = '', 'Korg M1'
    lib = ldf.copy()
    lib['y'], lib['kind'] = 1, 'sampled_library'
    lib['subkind'] = 'library:' + lib['instrument']
    lib['grp'] = 'syn:lib:' + lib['instrument']
    lib['device_type'], lib['brand_model'] = '', ''
    parts += [m1, lib]

    keep = ['file', 'y', 'kind', 'subkind', 'grp', 'device_type', 'brand_model', 'model_key', 'capture', 'flags'] + FEATURE_COLS
    for p_ in parts:
        p_['capture'] = ''
        p_['flags'] = ''
        if 'model_key' not in p_:
            p_['model_key'] = ''
    df = pd.concat([p[keep] for p in parts], ignore_index=True)
    # Isolated-note rows (CCMusic, SoundFont / Otey): one strike-anchored window
    # at the nominal pitch for each file.
    iso = df.kind.isin(['real_isolated', 'algo_single']).values
    paths = [os.path.join(ROOT, 'audio', f) if f.startswith('ccmusic_pianos/') else os.path.join(ROOT, f)
             for f in df.loc[iso, 'file']]
    fx = L.extract_cached(paths, 'corpus_isolated_reextract', isolated=True)
    df.loc[iso, FEATURE_COLS] = fx[FEATURE_COLS].values
    info['n_iso_extract_errors'] = int((fx['error'].fillna('') != '').sum())
    df, nbad = _filter_valid(df)
    info['n_invalid_base'] = nbad
    df['base'] = True

    # ---------------- performance sources
    info['v2'] = {}
    v2parts = []
    p = os.path.join(DATA_V2, 'digital_v2_features.csv')
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


def _listing(d):
    """Files of a library folder; without the audio, the names in the shipped library feature table."""
    if os.path.isdir(os.path.join(ROOT, d)):
        return sorted(os.listdir(os.path.join(ROOT, d)))
    t = pd.read_csv(os.path.join(L.FEAT_CACHE, 'library_notes_iso1.csv'), usecols=['file'])['file']
    pre = d.replace('\\', '/').rstrip('/') + '/'
    return sorted(f[len(pre):] for f in t if f.startswith(pre) and '/' not in f[len(pre):])


def library_note_files():
    rows = []
    for lib, (d, rx, layer_of, mf) in LIB_SPECS.items():
        for f in _listing(d):
            m = re.match(rx, f)
            if not m:
                continue
            layer = layer_of(m)
            rows.append({'library': lib, 'path': os.path.join(ROOT, d, f), 'layer': layer, 'mf_layer': layer == mf})
    return pd.DataFrame(rows)
