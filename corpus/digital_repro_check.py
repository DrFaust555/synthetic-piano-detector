"""Reproducibility check: recompute the 47 features of existing 'yt' demo files with the
current code (detector.pipeline, the calls of corpus/digital_finalize.py features)
and compare with the rows of data/corpus/synth_features.csv.

    python corpus/digital_repro_check.py [n_files]
"""
from __future__ import annotations
import os, sys, csv, json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))   # repository root
import corpus.digital_common as common
common.setup()
import numpy as np
from detector.pipeline import note_raw_descriptors, aggregate_file, FEATURE_COLS

REL = os.path.join(common.DATA, 'corpus', 'synth_features.csv')
TRENDS = common.TRENDS


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 14
    trends = json.load(open(TRENDS, encoding='utf-8'))
    rows = [r for r in csv.DictReader(open(REL, encoding='utf-8')) if r['synth_type'] == 'yt'][:n]
    worst = 0.0
    for r in rows:
        p = os.path.join(common.ROOT, r['file'])
        meta, notes, tun = note_raw_descriptors(p)
        f = aggregate_file(notes, tun, trends)
        diffs = []
        for c in FEATURE_COLS:
            a = r[c]
            b = f.get(c, float('nan'))
            if a == '' and (b != b):
                continue
            if a == '' or (b != b):
                diffs.append((c, a, b, 'NaN-mismatch')); continue
            a = float(a); b = round(float(b), 6)
            if abs(a - b) > 1e-5 * max(1.0, abs(a)):
                diffs.append((c, a, b, abs(a - b)))
        worst = max(worst, len(diffs))
        print(f"{os.path.basename(r['file'])[:50]:50s} n_notes={meta['n_notes']:4d} mismatches={len(diffs)}")
        for d in diffs[:6]:
            print('    ', d)
    print('MAX mismatching columns per file:', worst)


if __name__ == '__main__':
    main()
