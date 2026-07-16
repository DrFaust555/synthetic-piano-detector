"""Score arbitrary WAVs with the trained analytic quality critic (held-out).

    python -m quality.score_files <glob_or_dir> [more...]

Trains XGBoost on ALL of features_cal.csv (anchor-weighted), extracts the 45 analytic
features for each given file using the SAME register trends as training, and predicts
rating_cal.  Reminder: the critic's scale is compressed (synth≈13, top real≈45) — judge
relative to the reference anchors printed at the end.
"""

from __future__ import annotations
import os
import csv
import json
import sys
import glob
import numpy as np
import xgboost as xgb

from quality.pipeline import note_raw_descriptors, aggregate_file, FEATURE_COLS
from quality.xgb_quality import XGB, ANCHOR_W
from quality.effects import phaser_index
from quality.loss import phaser_penalty
from quality.aversiveness import aversiveness_features as _aver_features
from detune.segment import load_mono

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FEATURES_CAL = os.path.join(HERE, 'cache', 'features_cal.csv')
TRENDS = os.path.join(HERE, 'cache', 'register_trends_cal.json')

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
