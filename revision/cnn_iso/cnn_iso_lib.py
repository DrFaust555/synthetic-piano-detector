"""Format-matched (isolated-note) re-evaluation of the paper's phaseogram CNN and fusion -- shared helpers.

Read-only use of the repository: quality/realness_net.py (model + training code, unchanged),
quality/phasefeat.py and detune/segment.py (phaseogram construction, unchanged), and the protocol-v2
artefacts in Stattia/revision/analysis/ (fold map, library-note list, cached features of library
notes and listening clips, bootstrap code).  Everything written goes to Stattia/revision/cnn_iso/.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))          # project root
ANA = os.path.join(ROOT, 'Stattia', 'revision', 'analysis')
ANA_OUT = os.path.join(ANA, 'out')
KIT = os.path.join(ROOT, 'Stattia', 'listening_kit')
QCACHE = os.path.join(ROOT, 'quality', 'cache')
CACHE = os.path.join(HERE, 'cache')
JOBS = os.path.join(HERE, 'jobs')
LOGS = os.path.join(HERE, 'logs')

for p in (ROOT, ANA):
    if p not in sys.path:
        sys.path.insert(0, p)

import pv2_lib as L                                   # noqa: E402  (bootstrap, nominal pitch, MED16)
from pv2_lib import FEATURE_COLS, MED16               # noqa: E402

CLASS_OF = {'Iowa Steinway': 'Iowa', 'K18 upright': 'K18', 'Salamander (sampled)': 'Salamander',
            'soundfont': 'SoundFont (Casio CA-110)', 'Otey physical model': 'Otey'}
REC = ['Iowa', 'K18', 'Salamander']
ALG = ['SoundFont (Casio CA-110)', 'Otey']
LIBS = ['Salamander', 'Iowa', 'K18', 'FreePats']
OTEY_G = 'syn:otey'
CASIO_G = 'syn:sf2:' + L.norm_token('Casio_CA_110_Piano_FM')
MIDI_LO, MIDI_HI = 48, 76


def _sig(path):
    st = os.stat(path)
    return f'{st.st_size}:{int(st.st_mtime)}'


def universe() -> pd.DataFrame:
    """Every file of the format-matched study, one row each (one file = one struck note).

    source: ccmusic | soundfont | otey (training corpus, fixed folds from analysis/out/folds.csv),
            library (the 4 libraries' note files, all layers; fold = fold of the library group in
            folds.csv), clip (the 30 listening-test clips).
    Columns: file (id), path (absolute), source, sub, y, grp, fold, nominal_midi, layer, mf_layer.
    """
    from protocol_v2 import library_note_files                       # read-only import
    folds = pd.read_csv(os.path.join(ANA_OUT, 'folds.csv'))
    iso = folds[folds.subkind.isin(['ccmusic', 'soundfont_single', 'otey_single'])].copy()
    src = {'ccmusic': 'ccmusic', 'soundfont_single': 'soundfont', 'otey_single': 'otey'}
    iso['source'] = iso['subkind'].map(src)
    iso['path'] = [os.path.join(ROOT, 'audio', f) if f.startswith('ccmusic_pianos/') else os.path.join(ROOT, f)
                   for f in iso['file']]
    iso['sub'] = iso['grp']
    iso['layer'] = ''
    iso['mf_layer'] = False
    rows = [iso[['file', 'path', 'source', 'sub', 'y', 'grp', 'fold', 'layer', 'mf_layer']]]

    lib_fold = folds[folds.kind == 'sampled_library'].groupby('grp')['fold'].first().to_dict()
    libs = library_note_files()
    lb = pd.DataFrame({'file': [os.path.relpath(p, ROOT).replace('\\', '/') for p in libs['path']],
                       'path': libs['path'].values, 'source': 'library', 'sub': libs['library'].values, 'y': 1,
                       'grp': 'syn:lib:' + libs['library'].values, 'layer': libs['layer'].astype(str).values,
                       'mf_layer': libs['mf_layer'].values.astype(bool)})
    lb['fold'] = lb['grp'].map(lib_fold).astype(int)
    rows.append(lb)

    key = pd.read_csv(os.path.join(KIT, 'answer_key.csv'))
    cl = pd.DataFrame({'file': ['Stattia/listening_kit/web/' + c + '.wav' for c in key['clip']],
                       'path': [os.path.join(KIT, 'web', c + '.wav') for c in key['clip']],
                       'source': 'clip', 'sub': key['source'].map(CLASS_OF).values,
                       'y': np.where(key['source'].map(CLASS_OF).isin(ALG), 1, 0),
                       'grp': 'clip', 'fold': -1, 'layer': '', 'mf_layer': False})
    cl['note'] = key['note'].values
    rows.append(cl)
    u = pd.concat(rows, ignore_index=True)
    # nominal pitch: file names for corpus + libraries (pv2_lib.nominal_midi; CCMusic = Helmholtz token of the
    # dataset's `pitch` label), answer_key.csv 'note' for the clips
    nom = []
    for r in u.itertuples():
        nom.append(L.sci_to_midi(r.note) if r.source == 'clip' else L.nominal_midi(r.path))
    u['nominal_midi'] = nom
    if u['nominal_midi'].isna().any():
        bad = u[u['nominal_midi'].isna()]['file'].tolist()[:5]
        raise SystemExit(f'nominal pitch not parseable for {u["nominal_midi"].isna().sum()} files, e.g. {bad}')
    u['nominal_midi'] = u['nominal_midi'].astype(int)
    u['in_48_76'] = (u['nominal_midi'] >= MIDI_LO) & (u['nominal_midi'] <= MIDI_HI)
    u['sig'] = [_sig(p) for p in u['path']]
    return u.drop(columns=['note']).reset_index(drop=True)


def features(u: pd.DataFrame) -> np.ndarray:
    """47-feature vectors (FEATURE_COLS order) for every universe row, from the stored tables only:
    corpus rows = quality/cache/features_cal.csv (CCMusic) and synth_features.csv (SoundFont, Otey);
    library notes = analysis/out/cache/library_notes.csv; clips = analysis/out/cache/stimuli_clips.csv
    (both extracted by protocol v2 with the same single-file pipeline).  Read-only."""
    tabs = {}
    for name in ('features_cal.csv', 'synth_features.csv'):
        d = pd.read_csv(os.path.join(QCACHE, name))
        for f, v in zip(d['file'], d[FEATURE_COLS].apply(pd.to_numeric, errors='coerce').values):
            tabs[f] = v
    for name in ('library_notes.csv', 'stimuli_clips.csv'):
        d = pd.read_csv(os.path.join(ANA_OUT, 'cache', name), float_precision='round_trip')
        d['sig'] = d['sig'].astype(str)
        for f, s, v in zip(d['file'], d['sig'], d[FEATURE_COLS].apply(pd.to_numeric, errors='coerce').values):
            tabs[(f, s)] = v
    X = np.full((len(u), len(FEATURE_COLS)), np.nan)
    miss = []
    for i, r in enumerate(u.itertuples()):
        k = (r.file, r.sig) if r.source in ('library', 'clip') else r.file
        if k in tabs:
            X[i] = tabs[k]
        else:
            miss.append(r.file)
    if miss:
        raise SystemExit(f'47-feature row missing for {len(miss)} files, e.g. {miss[:3]}')
    return X.astype(np.float32)
