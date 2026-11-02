"""47 features of the SoundFont performance renders (data/performances/sf2_perf_features.csv), with the code path
of the corpus tables (training/synth_features.py, training/corpus_features.py):

    meta, notes, tuning = detector.pipeline.note_raw_descriptors(path)
    feats = detector.pipeline.aggregate_file(notes, tuning, trends)
    trends = data/corpus/register_trends_cal.json   (fixed, fitted on the real corpus)

Output layout = synth_features.csv: file, synth_type, instrument, 47 FEATURE_COLS; floats rounded to 6 decimals,
NaN written as ''.  Files whose extraction fails are omitted and listed in perf_features_log.csv.  File list:
data/performances/sf2_perf_manifest.csv (written by corpus/render_soundfont_performances.py).  Needs the audio.

    python training/perf_features.py [--workers 8]      -> results/training/sf2_perf_features.csv

Each file runs in its own worker process with a 300 s timeout.
"""
from __future__ import annotations
import os
import sys
import csv
import json
import time
import argparse
import multiprocessing as mp

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))   # repository root
from detector.paths import DATA, RESULTS, AUDIO_ROOT, results_dir   # noqa: E402
OUT = results_dir('training')
ROOT = AUDIO_ROOT
DATA_V2 = os.path.join(DATA, 'performances')
TRENDS = os.path.join(DATA, 'corpus', 'register_trends_cal.json')
JOBS = [  # (manifest, output csv)
    (os.path.join(DATA_V2, 'sf2_perf_manifest.csv'), os.path.join(OUT, 'sf2_perf_features.csv')),
]
LOG = os.path.join(OUT, 'perf_features_log.csv')
META = ['file', 'synth_type', 'instrument']
TIMEOUT = 300.0


def _work(path):
    import warnings
    warnings.filterwarnings('ignore')
    from detector.pipeline import note_raw_descriptors, aggregate_file
    trends = json.load(open(TRENDS, encoding='utf-8'))
    t = time.time()
    meta, notes, tuning = note_raw_descriptors(path)
    feats = aggregate_file(notes, tuning, trends)
    return meta, feats, time.time() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    from detector.pipeline import FEATURE_COLS
    cols = META + FEATURE_COLS
    log = []
    for man, out_csv in JOBS:
        if not os.path.exists(man):
            print(f"[skip] no manifest {man}")
            continue
        items = list(csv.DictReader(open(man, encoding='utf-8')))
        print(f"{os.path.basename(man)}: {len(items)} files", flush=True)
        pool = mp.Pool(a.workers, maxtasksperchild=1)
        asyncs = [(r, pool.apply_async(_work, (os.path.join(ROOT, r['file'].replace('/', os.sep)),)))
                  for r in items]
        results = {}
        for r, ar in asyncs:
            try:
                meta, feats, dt = ar.get(timeout=TIMEOUT)
                results[r['file']] = feats
                log.append({'file': r['file'], 'status': 'ok', 'n_notes_analysed': meta['n_notes'],
                            'median_midi': round(meta['median_midi'], 2),
                            'seg_mode': meta['mode'], 'duration_s': round(meta['duration_s'], 3),
                            'n_nan_features': sum(1 for c in FEATURE_COLS
                                                  if feats.get(c) is None or feats.get(c) != feats.get(c)),
                            'seconds': round(dt, 1)})
                print(f"  ok  {r['file']:70s} notes={meta['n_notes']:3d} {dt:5.1f}s", flush=True)
            except mp.TimeoutError:
                log.append({'file': r['file'], 'status': 'timeout'})
                print(f"  TIMEOUT {r['file']}", flush=True)
            except Exception as e:
                log.append({'file': r['file'], 'status': f'error: {e}'[:200]})
                print(f"  ERR {r['file']}: {e}", flush=True)
        pool.terminate()
        pool.join()
        n = 0
        with open(out_csv, 'w', newline='', encoding='utf-8') as fh:
            w = csv.writer(fh)
            w.writerow(cols)
            for r in items:                      # manifest order
                if r['file'] not in results:
                    continue
                rec = {'file': r['file'], 'synth_type': r['synth_type'],
                       'instrument': r['instrument'], **results[r['file']]}
                row = []
                for c in cols:
                    v = rec.get(c, '')
                    if isinstance(v, float):
                        v = '' if v != v else round(v, 6)
                    row.append(v)
                w.writerow(row)
                n += 1
        print(f"Wrote {n} rows -> {out_csv}", flush=True)
    keys = ['file', 'status', 'n_notes_analysed', 'median_midi', 'seg_mode', 'duration_s',
            'n_nan_features', 'seconds']
    with open(LOG, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in log:
            w.writerow({k: row.get(k, '') for k in keys})


if __name__ == '__main__':
    main()
