"""Shared two-stage score calibration (QUALITY_TOOL_PLAN.md §8).

The XGBoost quality predictions are compressed (regression to mean) and the
rating_cal label itself tops ~74.  We (1) isotonic-decompress the prediction back
toward the human rating, then (2) affine-stretch the observed range to a human
0-99 so a top instrument reads ~95 and cheap synth reads low.  Used by both the
overall combiner (D) and the timbre+liveness axis (B+L).
"""
from __future__ import annotations
import json

import numpy as np
from sklearn.isotonic import IsotonicRegression

# percentiles of the calibrated-pred distribution → output span.  OUT_LO is a
# soft floor (>0) so the very worst synth reads "very low" rather than a hard 0.
_STRETCH_LO_PCT, _STRETCH_HI_PCT = 3.0, 97.0
_STRETCH_OUT_LO, _STRETCH_OUT_HI = 10.0, 95.0


def fit_calibration(oof_pred: np.ndarray, y: np.ndarray) -> dict:
    """Fit isotonic(oof→y) + the percentile stretch anchors.  Returns a plain
    dict (JSON-serialisable) with isotonic knots and lo/hi stretch anchors."""
    iso = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=99.0)
    iso.fit(oof_pred, y)
    cal = iso.predict(oof_pred)
    return {'x': iso.X_thresholds_.tolist(), 'y': iso.y_thresholds_.tolist(),
            'lo': float(np.percentile(cal, _STRETCH_LO_PCT)),
            'hi': float(np.percentile(cal, _STRETCH_HI_PCT))}


def apply_calibration(raw, c: dict):
    """Map raw prediction(s) → human 0-99 via the fitted calibration dict."""
    xk, yk = np.asarray(c['x']), np.asarray(c['y'])
    cal = np.interp(raw, xk, yk, left=yk[0], right=yk[-1])
    span = max(c['hi'] - c['lo'], 1e-6)
    out = (cal - c['lo']) / span * (_STRETCH_OUT_HI - _STRETCH_OUT_LO) + _STRETCH_OUT_LO
    return np.clip(out, 0.0, 99.0)


def save_calibration(c: dict, path: str):
    json.dump(c, open(path, 'w'))


def load_calibration(path: str) -> dict:
    return json.load(open(path))
