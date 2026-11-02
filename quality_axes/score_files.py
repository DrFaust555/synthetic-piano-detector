"""The 47-feature vector of an audio file with the register trends of the rated corpus (_feat), and a quick
score of WAV files with the analytic 47-feature quality regressor fitted on all of data/corpus/features_cal.csv.
The regressor's scale is compressed (synthetic ~13, top real ~45); read it against the reference files printed last.

    SPD_AUDIO=<audio root> python -m quality_axes.score_files <glob_or_dir> [more...]
"""

from __future__ import annotations
import os
import csv
import json
import sys
import glob
import numpy as np
import xgboost as xgb

from detector.pipeline import note_raw_descriptors, aggregate_file, FEATURE_COLS
from quality_axes.regressor import XGB, ANCHOR_W
from quality_axes.phaser import phaser_index
from quality_axes.objectives import phaser_penalty
from quality_axes.aversiveness import aversiveness_features as _aver_features
from detector.segment import load_mono

from detector.paths import AUDIO_ROOT as ROOT   # noqa: E402  (relative audio paths)
from quality_axes.models import FEATURES_CAL, TRENDS   # noqa: E402

REFS = [  # context anchors (path, label)
    ('otey/baseline/C4_v127.wav', 'Otey фізмодель'),
    ('audio/m1_patches/m1_M09-03_Toy-Piano.wav', 'm1 синт'),
    ('audio/129_stanford_fizpiano_37.wav', 'Stanford physmod'),
    ('audio/42_bosendorfer_280vc_97.wav', 'Bösendorfer (реал 97)'),
    ('audio/02_ukraina_tuned.wav', 'Ukraina tuned (реал)'),
]


def _train():
    rows = list(csv.DictReader(open(FEATURES_CAL, encoding='utf-8')))
    M = np.array([[float(r[c]) if r.get(c, '') not in ('', 'nan') else np.nan
                   for c in FEATURE_COLS] for r in rows])
    y = np.array([float(r['rating_cal']) for r in rows])
    w = np.array([ANCHOR_W if int(r.get('is_anchor', 0) or 0) else 1.0 for r in rows])
    return xgb.XGBRegressor(**XGB).fit(M, y, sample_weight=w)


def _feat(path, trends):
    meta, notes, tuning = note_raw_descriptors(path)
    row = aggregate_file(notes, tuning, trends)
    return np.array([row.get(c, np.nan) for c in FEATURE_COLS]), len(notes)


def main():
    args = sys.argv[1:]
    files = []
    for a in args:
        p = a if os.path.isabs(a) else os.path.join(ROOT, a)
        if os.path.isdir(p):
            files += sorted(glob.glob(os.path.join(p, '*.wav')))
        else:
            files += sorted(glob.glob(p))
    if not files:
        print("Вкажи теку або glob WAV."); return

    trends = json.load(open(TRENDS, encoding='utf-8'))
    model = _train()

    print(f"{'gated':>7} {'xgb':>6} {'phaser':>7} {'aver':>5}  {'notes':>5}  file")
    print('-' * 72)
    scores = []
    for f in files:
        try:
            x, n = _feat(f, trends)
            p = float(model.predict(x[None, :])[0])
            wav, sr = load_mono(f)
            ph = phaser_index(wav, sr)
            av = _aver_features(wav, sr)['aversiveness']  # 0-100 психоакустична противність
            gated = p - phaser_penalty(f, idx=ph)        # phaser red-flag penalty
            scores.append(gated)
            flag = '  ⚑PHASER' if ph > 3.5 else ''
            flag += '  ⚑НЕПРИЄМНО' if av >= 50 else ''
            print(f"{gated:7.1f} {p:6.1f} {ph:7.1f} {av:5.0f}  {n:5d}  {os.path.basename(f)}{flag}")
        except Exception as e:
            print(f"{'ERR':>7}                       {os.path.basename(f)}: {e}")
    if scores:
        print(f"\n  target files: median={np.median(scores):.1f}  "
              f"mean={np.mean(scores):.1f}  min={min(scores):.1f}  max={max(scores):.1f}")

    print("\n  --- референси (та сама модель, стиснута шкала) ---")
    print(f"  {'xgb':>7} {'aver':>5}  {'notes':>5}  label")
    for rel, lab in REFS:
        p = os.path.join(ROOT, rel)
        if not os.path.exists(p):
            continue
        try:
            x, n = _feat(p, trends)
            wav, sr = load_mono(p)
            av = _aver_features(wav, sr)['aversiveness']
            print(f"  {model.predict(x[None,:])[0]:7.1f} {av:5.0f}  {n:5d}  {lab}")
        except Exception as e:
            print(f"  {'ERR':>7}         {lab}: {e}")


if __name__ == '__main__':
    main()
