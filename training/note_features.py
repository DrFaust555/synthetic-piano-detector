"""Per-file feature tables of data/features/ re-extracted from the audio (needs SPD_AUDIO).

    python training/note_features.py [--tables corpus,library,clips,pianoteq] [--limit N] [--workers 8]

Tables (one row per audio file: file, sig = size:mtime, n_notes, duration_s, mode, error, 47 features):
    corpus    corpus_isolated_reextract_iso1.csv  CCMusic notes, SoundFont single-note renders, Otey notes
    library   library_notes_iso1.csv              the four sample libraries' note files
    clips     stimuli_clips_iso1.csv              the 30 listening-test clips (pitch from the answer key)
    pianoteq  table3_perf_flux.csv                Pianoteq demo segments (performance segmentation)
Isolated notes are analysed as one note: one window from the strike, the nominal pitch (file name, or the answer
key for the clips) refined on the spectrum (protocol.common.note_raw_iso).  Features: detector.pipeline
note_raw_descriptors + aggregate_file with the fixed register trends data/corpus/register_trends_cal.json.

The file list is that of the shipped table; the new table is written to results/note_features/<table>.csv and
compared with the shipped one (number of files whose 47 features differ by more than 1e-6).  --limit N takes the
first N files of each table (a quick check).  The protocol scripts read the shipped tables in data/features/: a
file listed there is always taken from the table; with SPD_REEXTRACT=1 files are keyed by their own size:mtime and
the ones not found are extracted and appended.
"""
from __future__ import annotations

import os
import sys
import argparse


HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..')))   # repository root
from detector.paths import DATA, results_dir   # noqa: E402

import numpy as np            # noqa: E402
import pandas as pd           # noqa: E402
import protocol.common as L   # noqa: E402

OUT = results_dir('note_features')
TABLES = {'corpus': ('corpus_isolated_reextract_iso1.csv', True),
          'library': ('library_notes_iso1.csv', True),
          'clips': ('stimuli_clips_iso1.csv', True),
          'pianoteq': ('table3_perf_flux.csv', False)}


def _job(args):
    path, isolated, midi = args
    return L.extract_one(path, isolated=isolated, midi=midi)


def main(tables, limit=None, workers=8):
    key = pd.read_csv(os.path.join(DATA, 'listening', 'answer_key.csv'))
    clip_midi = {f'Stattia/listening_kit/web/{c}.wav': L.sci_to_midi(n) for c, n in zip(key['clip'], key['note'])}
    for t in tables:
        name, isolated = TABLES[t]
        ref = pd.read_csv(os.path.join(DATA, 'features', name), float_precision='round_trip')
        ref['sig'] = ref['sig'].astype(str)
        if limit:
            ref = ref.head(limit)
        fcols = L.FEATURE_COLS if isolated else L.FEATURE_COLS
        jobs = [(os.path.join(L.ROOT, f), isolated, clip_midi.get(f)) for f in ref['file']]
        missing = [j[0] for j in jobs if not os.path.exists(j[0])]
        if missing:
            raise SystemExit(f'{len(missing)} audio file(s) not found (set SPD_AUDIO), e.g. {missing[0]}')
        if workers > 1 and len(jobs) > 4:
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=workers) as ex:
                res = list(ex.map(_job, jobs, chunksize=4))
        else:
            res = [_job(j) for j in jobs]
        new = pd.DataFrame([{'file': f, 'sig': L._sig(j[0]), **r} for f, j, r in zip(ref['file'], jobs, res)])
        for c in fcols:
            if c not in new:
                new[c] = np.nan
        new = new[['file', 'sig', 'n_notes', 'duration_s', 'mode', 'error'] + fcols]
        out = os.path.join(OUT, name)
        new.to_csv(out, index=False)
        new = pd.read_csv(out, float_precision='round_trip')         # compare the values as stored
        a, b = new[fcols].values.astype(float), ref[fcols].values.astype(float)
        same = (np.isnan(a) == np.isnan(b)) & (np.isnan(a) | (np.abs(a - np.nan_to_num(b)) <= 1e-6))
        nd = int((~same.all(axis=1)).sum())
        md = float(np.nanmax(np.abs(a - b))) if np.isfinite(a - b).any() else 0.0
        print(f'{t}: {len(new)} files -> {out}; {nd} file(s) differ from data/features/{name} '
              f'(max |difference| {md:.2e}); size:mtime keys equal for {int((new["sig"].astype(str) == ref["sig"]).sum())}',
              flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--tables', default='corpus,library,clips,pianoteq')
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    main(a.tables.split(','), a.limit, a.workers)
