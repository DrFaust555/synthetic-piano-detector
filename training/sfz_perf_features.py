"""47 features of the SFZ sample-library performance renders (corpus/render_sfz_performances.py), with the
per-file code path of training/perf_features.py (note_raw_descriptors + aggregate_file, fixed register trends).

    python training/sfz_perf_features.py [--audio-root DIR] [--workers 8] [--check SF2_WAV]

--audio-root: folder holding audio/sfz_performances_v2/<lib>/<lib>_slotNNN.wav (default: SPD_AUDIO).
--check: also extract one SoundFont performance file and compare it with data/performances/sf2_perf_features.csv.
Writes results/training/sfz_perf_features.csv (file, synth_type, instrument, 47 FEATURE_COLS; layout as
sf2_perf_features.csv) and results/training/sfz_perf_features_log.csv.
"""
from __future__ import annotations
import os
import sys
import csv
import glob
import argparse
import multiprocessing as mp

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))   # repository root
from detector.paths import DATA, RESULTS, AUDIO_ROOT, results_dir   # noqa: E402
from training.perf_features import ROOT, DATA_V2, META, TIMEOUT, _work

LIB_ORDER = ['Salamander', 'K18', 'FreePats', 'Iowa']
OUT = os.path.join(results_dir('training'), 'sfz_perf_features.csv')
LOG = os.path.join(results_dir('training'), 'sfz_perf_features_log.csv')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--audio-root', default=ROOT)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--check', default='')
    a = ap.parse_args()
    from detector.pipeline import FEATURE_COLS
    if a.check:
        rel = 'audio/sf2_performances_v2/' + '/'.join(a.check.replace('\\', '/').split('/')[-2:])
        ref = {r['file']: r for r in csv.DictReader(open(os.path.join(DATA_V2, 'sf2_perf_features.csv'), encoding='utf-8'))}[rel]
        _, feats, _ = _work(a.check)
        bad = [c for c in FEATURE_COLS if (ref[c] == '') != (feats[c] != feats[c])
               or (ref[c] != '' and abs(float(ref[c]) - round(feats[c], 6)) > 1.5e-6)]
        print(f'check {rel}: {len(FEATURE_COLS) - len(bad)}/{len(FEATURE_COLS)} features identical', bad[:5], flush=True)
    files = []
    for lib in LIB_ORDER:
        for p in sorted(glob.glob(os.path.join(a.audio_root, 'audio', 'sfz_performances_v2', lib, f'{lib}_slot*.wav'))):
            files.append((lib, p, f'audio/sfz_performances_v2/{lib}/{os.path.basename(p)}'))
    print(len(files), 'files', flush=True)
    pool = mp.Pool(a.workers, maxtasksperchild=1)
    asyncs = [(lib, p, rel, pool.apply_async(_work, (p,))) for lib, p, rel in files]
    rows, log = [], []
    for lib, p, rel, ar in asyncs:
        try:
            meta, feats, dt = ar.get(timeout=TIMEOUT)
            row = [rel, 'sampled_perf', lib]
            for c in FEATURE_COLS:
                v = feats.get(c, '')
                row.append(('' if v != v else round(v, 6)) if isinstance(v, float) else v)
            rows.append(row)
            log.append({'file': rel, 'status': 'ok', 'n_notes_analysed': meta['n_notes'],
                        'median_midi': round(meta['median_midi'], 2), 'seg_mode': meta['mode'],
                        'duration_s': round(meta['duration_s'], 3), 'seconds': round(dt, 1)})
            print(f'  ok  {rel:60s} notes={meta["n_notes"]:3d} {dt:5.1f}s', flush=True)
        except mp.TimeoutError:
            log.append({'file': rel, 'status': 'timeout'})
            print(f'  TIMEOUT {rel}', flush=True)
        except Exception as e:
            log.append({'file': rel, 'status': f'error: {e}'[:200]})
            print(f'  ERR {rel}: {e}', flush=True)
    pool.terminate()
    pool.join()
    with open(OUT, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(META + FEATURE_COLS)
        w.writerows(rows)
    keys = ['file', 'status', 'n_notes_analysed', 'median_midi', 'seg_mode', 'duration_s', 'seconds']
    with open(LOG, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows({k: r.get(k, '') for k in keys} for r in log)
    print(f'Wrote {len(rows)} rows -> {OUT}', flush=True)


if __name__ == '__main__':
    main()
