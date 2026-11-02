"""Per-file features of the real corpus with quality labels: data/corpus/features_cal.csv and the register
trends data/corpus/register_trends_cal.json (needs the audio).

    python training/corpus_features.py

Pass 1: raw per-note descriptors of every file (detector.pipeline.note_raw_descriptors), checkpointed per file in
results/training/notes_cal/ (a crash or a kill loses nothing; a file that hangs longer than PER_FILE_TIMEOUT is
skipped by a supervisor).  Pass 2: register (MIDI) trends fitted on all notes, then every file register-normalised
and aggregated into the 47 features (aggregate_file).  The file list and the labels (rating_cal, is_anchor,
brand, source) are those of data/corpus/features_cal.csv.  Outputs go to results/training/.
CAL_SHARD=i/n runs rows i::n in this process; CAL_PASS=1 or 2 runs one pass only; CAL_KEEP_TRENDS=1 aggregates
with the shipped data/corpus/register_trends_cal.json instead of refitting it.
"""

from __future__ import annotations
import os
import sys
import csv
import json
import time
import multiprocessing as mp

from detector.pipeline import (
    note_raw_descriptors, fit_trends, aggregate_file, FEATURE_COLS,
)
from detector.phaseogram_cnn import safe_name

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..')))   # repository root
from detector.paths import DATA, RESULTS, AUDIO_ROOT, results_dir   # noqa: E402
OUT = results_dir('training')
ROOT = AUDIO_ROOT
AUDIO_DIR = os.path.join(ROOT, 'audio')
RATINGS = os.path.join(DATA, 'corpus', 'features_cal.csv')    # file list and labels
CACHE = OUT
CKPT_DIR = os.path.join(CACHE, 'notes_cal')
SKIP_PATH = os.path.join(CKPT_DIR, '_skip.txt')
STATUS = os.path.join(CKPT_DIR, '_status.txt')
OUT_CSV = os.path.join(CACHE, 'features_cal.csv')
TRENDS = os.path.join(CACHE, 'register_trends_cal.json')
META = ['file', 'rating_cal', 'is_anchor', 'brand', 'source']
PER_FILE_TIMEOUT = 180.0      # a single file taking longer than this = hung → skip
# CAL_SHARD=i/n runs this process on rows i::n with its own heartbeat and skip-list; CAL_PASS=1 or 2 runs
# one pass only; CAL_KEEP_TRENDS=1 makes pass 2 aggregate with the shipped register trends.
SHARD = os.environ.get('CAL_SHARD', '')
if SHARD:
    STATUS = os.path.join(CKPT_DIR, f"_status_{SHARD.replace('/', 'of')}.txt")
    SKIP_PATH = os.path.join(CKPT_DIR, f"_skip_{SHARD.replace('/', 'of')}.txt")
PASS = os.environ.get('CAL_PASS', '')
KEEP_TRENDS = os.environ.get('CAL_KEEP_TRENDS') == '1'


def _rows():
    rows = []
    with open(RATINGS, encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if (r.get('rating_cal') or '').strip() not in ('', 'nan'):
                rows.append(r)
    return rows


def _load_skip():
    if not os.path.exists(SKIP_PATH):
        return set()
    return set(l.strip() for l in open(SKIP_PATH, encoding='utf-8') if l.strip())


def _worker(rows):
    skip = _load_skip()
    for r in rows:
        fn = r['file']
        cp = os.path.join(CKPT_DIR, safe_name(fn) + '.json')
        if os.path.exists(cp) or fn in skip:
            continue
        with open(STATUS, 'w', encoding='utf-8') as s:   # heartbeat = current file
            s.write(fn)
        path = os.path.join(AUDIO_DIR, fn)
        rec = {k: r.get(k, '') for k in META}
        if not os.path.exists(path):
            json.dump({'rec': rec, 'skip': 'missing'}, open(cp, 'w'))
            continue
        try:
            meta, notes, tuning = note_raw_descriptors(path)
            json.dump({'rec': rec, 'meta': meta, 'notes': notes, 'tuning': tuning},
                      open(cp, 'w'))
        except Exception as e:
            json.dump({'rec': rec, 'error': str(e)}, open(cp, 'w'))


def _pass1(rows):
    """Run the worker under a stall-supervisor until every file is checkpointed."""
    os.makedirs(CKPT_DIR, exist_ok=True)
    total = len(rows)
    cps = [os.path.join(CKPT_DIR, safe_name(r['file']) + '.json') for r in rows]
    n_done = lambda: sum(os.path.exists(c) for c in cps)
    while True:
        done = n_done()
        if done >= total - len(_load_skip()):
            break
        p = mp.Process(target=_worker, args=(rows,))
        p.start()
        ticks = 0
        while p.is_alive():
            time.sleep(15)
            ticks += 1
            done = n_done()
            if ticks % 4 == 0:
                print(f"  ... {done}/{total} checkpointed", flush=True)
            age = time.time() - os.path.getmtime(STATUS) if os.path.exists(STATUS) else 0
            if age > PER_FILE_TIMEOUT:
                hung = open(STATUS, encoding='utf-8').read().strip()
                with open(SKIP_PATH, 'a', encoding='utf-8') as sf:
                    sf.write(hung + '\n')
                print(f"  [SKIP hung] {hung}  (done={done}/{total})", flush=True)
                p.terminate(); p.join(); break
        else:
            print(f"  worker finished (done={done}/{total})", flush=True)
            if p.exitcode == 0:
                break
        if not p.is_alive() and p.exitcode == 0:
            break


def _pass2():
    per_file, all_notes = [], []
    n_err = n_skip = 0
    for jf in os.listdir(CKPT_DIR):
        if not jf.endswith('.json'):
            continue
        d = json.load(open(os.path.join(CKPT_DIR, jf), encoding='utf-8'))
        if 'notes' not in d:
            n_err += ('error' in d); n_skip += ('skip' in d)
            continue
        per_file.append((d['rec'], d['notes'], d['tuning']))
        all_notes.extend(d['notes'])
    print(f"  pass2: {len(per_file)} files, {len(all_notes)} notes "
          f"({n_err} errors, {n_skip} missing/skipped)", flush=True)
    if not per_file:
        return
    if KEEP_TRENDS:
        kept = os.path.join(DATA, 'corpus', 'register_trends_cal.json')
        trends = json.load(open(kept, encoding='utf-8'))
        print(f"  pass2: register trends kept from {kept}", flush=True)
    else:
        trends = fit_trends(all_notes)
        json.dump(trends, open(TRENDS, 'w', encoding='utf-8'), ensure_ascii=False)
    cols = META + FEATURE_COLS
    with open(OUT_CSV, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(cols)
        for rec, notes, tuning in per_file:
            feats = aggregate_file(notes, tuning, trends)
            row = {**{k: rec.get(k, '') for k in META}, **feats}
            out = []
            for c in cols:
                v = row.get(c, '')
                if isinstance(v, float):
                    v = '' if v != v else round(v, 6)
                out.append(v)
            w.writerow(out)
    print(f"\nWrote {len(per_file)} rows -> {OUT_CSV}  ({len(FEATURE_COLS)} features)",
          flush=True)


def main():
    rows = _rows()
    if SHARD:
        i, n = map(int, SHARD.split('/'))
        rows = rows[i::n]
    print(f"Files to extract: {len(rows)}  (skipped: {len(_load_skip())})", flush=True)
    if PASS != '2':
        _pass1(rows)
    if PASS != '1':
        _pass2()


if __name__ == '__main__':
    main()
