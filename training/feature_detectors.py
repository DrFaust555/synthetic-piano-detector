"""Out-of-fold feature-detector scores for the paper's isolated-note table.

XGBoost on 27 features, trained on CCMusic, SoundFont and Otey notes. Libraries
are never trained and are scored by the model of their assigned fold.
    python training/feature_detectors.py [--no-ci]
Writes results/feature_detectors/notes_scores.csv and summary.csv.
The performance feature detector is experiments/01_detection/performances_features.py.
"""
from __future__ import annotations

import os
import sys
import argparse


HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..')))   # repository root
from detector.paths import results_dir   # noqa: E402

import numpy as np            # noqa: E402
import pandas as pd           # noqa: E402
import protocol.common as L   # noqa: E402
import protocol.corpus as V   # noqa: E402
import protocol.corpus_base as P   # noqa: E402
from protocol.common import FEATURE_COLS, ROOT   # noqa: E402

OUT = results_dir('feature_detectors')


def library_notes(df):
    """Library notes (Salamander, Iowa, K18, FreePats) with valid features, grouped by library."""
    libs = P.library_note_files()
    lf = L.extract_cached(list(libs['path']), 'library_notes', isolated=True)
    libs = pd.concat([libs.reset_index(drop=True), lf], axis=1)
    libs['grp'] = 'syn:lib:' + libs['library']
    libs['valid'] = (libs[FEATURE_COLS].notna().sum(axis=1) >= 30) & (libs['error'].fillna('') == '')
    return libs[libs['valid']].reset_index(drop=True)


def notes_table(df, libv):
    """One row per isolated-note file: source, label, group, fold and its features."""
    rows = []
    for _, r in df[df.subkind == 'ccmusic'].iterrows():
        rows.append(dict(src='CCMusic', y=0, grp=r.grp, fold=r.fold, file=r.file,
                         inst=r.file.split('/')[-1].split('_')[0], **r[L.FEATURE_COLS].to_dict()))
    for _, r in df[(df.subkind == 'soundfont_single') & ~df.sf_dup].iterrows():
        rows.append(dict(src='SoundFont', y=1, grp=r.grp, fold=r.fold, file=r.file, inst=r.file.split('/')[-2],
                         **r[L.FEATURE_COLS].to_dict()))
    for _, r in df[df.subkind == 'otey_single'].iterrows():
        rows.append(dict(src='Otey', y=1, grp=r.grp, fold=r.fold, file=r.file, inst='Otey',
                         **r[L.FEATURE_COLS].to_dict()))
    lib_fold = {g: int(df.loc[df.grp == g, 'fold'].iloc[0]) for g in libv['grp'].unique()}
    for _, r in libv.iterrows():
        rows.append(dict(src='library', y=1, grp=r['grp'], fold=lib_fold[r['grp']],
                         file=os.path.relpath(r['path'], ROOT).replace('\\', '/'), inst=r['library'],
                         mf=bool(r['mf_layer']), **r[L.FEATURE_COLS].to_dict()))
    t = pd.DataFrame(rows)
    t['mf'] = t['mf'].fillna(False).astype(bool)
    for c in L.FEATURE_COLS:
        t[c] = t[c].astype(float)
    return t


def notes_detector(src):
    """Train on CCMusic + SoundFont + Otey (out of fold), score library notes with their fold's model."""
    d = src.copy()
    d['kind'] = np.where(d.src == 'library', 'sampled_library', np.where(d.y == 0, 'real_isolated', 'algo_single'))
    ok = d[FEATURE_COLS].notna().sum(axis=1) >= 30
    d = d[ok | (d.src == 'library')].reset_index(drop=True)
    train = d[d.src != 'library'].reset_index(drop=True)
    lib = d[d.src == 'library'].reset_index(drop=True)
    res = V.run_cv3(train, V.FS3_ISO['all47'], 'T1', ext={'lib': lib}, ext_fold={'lib': lib['fold'].values})
    allr = pd.concat([train, lib], ignore_index=True)
    allr['P'] = np.r_[res['own'], res['ext_own']['lib']]
    return allr


def main(ci=True):
    df, _ = V.build_corpus_v3(log=lambda *a, **k: None)
    libv = library_notes(df)
    rows = []

    notes = notes_detector(notes_table(df, libv))
    notes[['file', 'src', 'inst', 'grp', 'fold', 'P']].to_csv(
        os.path.join(OUT, 'notes_scores.csv'), index=False, float_format='%.6f')
    y, p, g = notes['y'].values, notes['P'].values, notes['grp'].values
    real = (notes.src == 'CCMusic').values
    lib = (notes.src == 'library').values
    for name, sm in (('SoundFont single notes', (notes.src == 'SoundFont').values),
                     ('Otey single notes', (notes.src == 'Otey').values),
                     ('SoundFont + Otey single notes', notes.src.isin(['SoundFont', 'Otey']).values),
                     ('library notes, mf layer', lib & notes['mf'].values),
                     ('library notes, all layers', lib)):
        yy = np.where(sm, 1, 0)
        r = L.compare(yy, p, g, real, sm, boot=ci)
        rows.append({'detector': 'isolated notes', 'comparison': f'CCMusic vs {name}', **r})

    s = pd.DataFrame(rows)
    s.to_csv(os.path.join(OUT, 'summary.csv'), index=False, float_format='%.6f')
    for r in s.itertuples():
        print(f'{r.comparison[:60]:60s} {r.n_syn:5d} files  AUC {r.auc:.3f} [{r.ci_lo:.3f}, {r.ci_hi:.3f}]  '
              f'det {r.det_syn:.1%}  mean P {r.meanP_syn:.3f}')
    print(f'wrote {OUT}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--no-ci', action='store_true', help='skip the grouped bootstrap intervals')
    main(ci=not ap.parse_args().no_ci)
