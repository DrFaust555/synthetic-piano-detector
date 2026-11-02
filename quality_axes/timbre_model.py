"""Timbre regressor of the current quality tool (Sec. 3.8): gradient-boosted regressor on the 40 features of
Groups A (16 medians + 16 IQRs) and B (8 register-localized features); the 7 tuning features of Group C are left out,
so an instrument that is well built but out of tune keeps its timbre score.  Fitted on all rows of
data/corpus/features_cal.csv (rating_cal, CCMusic anchors weighted x3), same XGBoost settings as regressor.py.

_calibration(): isotonic regression of the rating on the in-sample predictions, then an affine stretch that maps the
2nd / 98th percentiles of the calibrated corpus to 3 / 97.
"""
from __future__ import annotations
import os
import sys
import csv
import json
import numpy as np


import xgboost as xgb
from sklearn.isotonic import IsotonicRegression

from detector.pipeline import DESCRIPTORS, BASS_FEATURE_COLS

from quality_axes.models import FEATURES_CAL, TRENDS   # noqa: E402

# The timbre model sees Group A (16 med + 16 iqr) + Group B (8 register), not Group C (tuning).
TIMBRE_COLS = ([f'{d}_med' for d in DESCRIPTORS]
               + [f'{d}_iqr' for d in DESCRIPTORS]
               + list(BASS_FEATURE_COLS))

XGB = dict(n_estimators=500, max_depth=4, learning_rate=0.03, subsample=0.8,
           colsample_bytree=0.8, reg_lambda=2.0, objective='reg:pseudohubererror',
           random_state=0, n_jobs=0)
ANCHOR_W = 3.0


# --------------------------------------------------------------------------- fit
def _train():
    rows = list(csv.DictReader(open(FEATURES_CAL, encoding='utf-8')))
    def col(c):
        return np.array([float(r[c]) if r.get(c, '') not in ('', 'nan') else np.nan
                         for r in rows])
    M = np.column_stack([col(c) for c in TIMBRE_COLS])
    y = np.array([float(r['rating_cal']) for r in rows])
    w = np.array([ANCHOR_W if int(r.get('is_anchor', 0) or 0) else 1.0 for r in rows])
    model = xgb.XGBRegressor(**XGB).fit(M, y, sample_weight=w)
    return model, M, y


def _calibration(model, M, y):
    """Isotonic GBM->human, then an affine stretch anchoring the calibrated
    corpus span to a fuller 0..99 (§5.2).  Isotonic is fit on the deployed
    predictions so it is a monotone recalibration; the affine maps the 2nd/98th
    percentile of the calibrated corpus to fixed anchors (3 and 97) so reference
    sources land near their Table-3 values without distorting the interior."""
    raw = model.predict(M)
    iso = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=99.0).fit(raw, y)
    calib = iso.predict(raw)
    lo, hi = float(np.percentile(calib, 2)), float(np.percentile(calib, 98))
    a = (97.0 - 3.0) / (hi - lo) if hi > lo else 1.0
    b = 3.0 - a * lo
    # isotonic as a compact monotone step function (dedup x)
    xs = np.asarray(iso.f_.x, dtype=float)
    ys = np.asarray(iso.f_.y, dtype=float)
    return dict(iso_x=xs.tolist(), iso_y=ys.tolist(), affine_a=float(a), affine_b=float(b))

