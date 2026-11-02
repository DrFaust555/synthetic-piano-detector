"""Overall quality D of the quality tool (Sec. 3.8): the version behind the printed quality column of Table 6.

One human-scaled 0-99 number for a recording:
  gradient-boosted regressor on 52 descriptors (the 47 features + the 5 liveness descriptors) -> rating_cal
  -> isotonic calibration + affine stretch to fixed anchors (calibration.py)
  -> minus the guardrail G (phaser index, psychoacoustic aversiveness), capped at 25 points.
Provenance (P(synthetic)) never lowers D.

Scoring reads the fitted models of data/quality/models (combiner_overall.json + .cols, combiner_iso.json) unless
SPD_QUALITY_MODELS names another folder; --save retrains them into results/quality_models.

    python -m quality_axes.combiner            # grouped-CV Spearman of the regressor + reference files
    python -m quality_axes.combiner --save     # fit + save model and calibration
    SPD_AUDIO=<audio root> python -m quality_axes.combiner <wav>...   # overall score of file(s)
"""
from __future__ import annotations
import os
import json
import sys

import numpy as np
import xgboost as xgb
from scipy.stats import spearmanr

from detector.pipeline import FEATURE_COLS
from quality_axes.regressor import XGB, _oof, _report
from quality_axes.timbre import _load_with_liveness, LIVENESS_COLS

from detector.paths import AUDIO_ROOT   # noqa: E402
from quality_axes.models import MODELS, TRENDS, save_path   # noqa: E402
MODEL = os.path.join(MODELS, 'combiner_overall.json')
ISO = os.path.join(MODELS, 'combiner_iso.json')

FULL_L_COLS = FEATURE_COLS + LIVENESS_COLS   # 52: everything


# ---- guardrail G ------------------------------------------------------------
G_MAX = 25.0        # cap: a guardrail flag docks, never annihilates
PHASER_MIN_NOTES = 8  # phaser LFO detector is unreliable on isolated/short material


def guardrail_penalty(path: str, n_notes: int = 999) -> float:
    """Explicit red-flag penalty the rating model can't learn (no phaser or artifact
    positives in training): phaser + high aversiveness dock the overall score.
    Phaser is gated to material with enough notes (it false-fires on isolated
    notes); the total is capped at G_MAX."""
    from quality_axes.objectives import phaser_penalty, aversiveness
    ph = phaser_penalty(path) if n_notes >= PHASER_MIN_NOTES else 0.0
    av = aversiveness(path)
    return float(min(G_MAX, ph + max(0.0, av - 50.0) * 0.5))


# ---- calibration: isotonic (decompress) → affine stretch to human 0-99 ------
from quality_axes.calibration import (fit_calibration as _fit_calibration,  # noqa: E402
                               apply_calibration as _apply_calibration,
                               save_calibration, load_calibration)


def _save_cal(c: dict):
    save_calibration(c, ISO)


def _load_cal():
    return load_calibration(ISO)


def train_and_save():
    rows, y, anchor, w, groups, X = _load_with_liveness()
    oof = _oof(FULL_L_COLS, X, y, w, groups)
    c = _fit_calibration(oof, y)
    M = np.column_stack([X[c_] for c_ in FULL_L_COLS])
    model = xgb.XGBRegressor(**XGB).fit(M, y, sample_weight=w)
    out = save_path('combiner_overall.json')
    model.save_model(out)
    json.dump(FULL_L_COLS, open(out + '.cols', 'w'))
    save_calibration(c, save_path('combiner_iso.json'))
    print(f"saved overall model ({len(FULL_L_COLS)} feats) → {out}")
    print(f"raw ρ={spearmanr(oof, y).statistic:.3f}  "
          f"(stretch anchors lo={c['lo']:.1f} hi={c['hi']:.1f})")
    return model, c


# ---- inference --------------------------------------------------------------
def _raw_pred(path: str, model, cols, trends):
    from quality_axes.score_files import _feat
    from quality_axes.modulation import modulation_features
    from detector.segment import load_mono
    xt, n = _feat(path, trends)
    fidx = {c: FEATURE_COLS.index(c) for c in cols if c in FEATURE_COLS}
    wav, sr = load_mono(path)
    m = modulation_features(wav, sr)
    vec = [xt[fidx[c]] if c in fidx else m.get(c, np.nan) for c in cols]
    return float(model.predict(np.array(vec)[None, :])[0]), n


def overall_score(path: str, with_guardrail: bool = True) -> dict:
    """Overall quality 0-99 for one file (+ diagnostics)."""
    model = xgb.XGBRegressor()
    model.load_model(MODEL)
    cols = json.load(open(MODEL + '.cols'))
    cal_params = _load_cal()
    trends = json.load(open(TRENDS, encoding='utf-8'))
    raw, n = _raw_pred(path, model, cols, trends)
    cal = float(_apply_calibration(raw, cal_params))
    g = guardrail_penalty(path, n) if with_guardrail else 0.0
    overall = float(np.clip(cal - g, 0.0, 99.0))
    return {'overall': overall, 'quality_cal': cal, 'guardrail': g,
            'raw': raw, 'n_notes': n, 'path': path}


# ---- reporting --------------------------------------------------------------
def _report_refs(model, cols, cal_params):
    from quality_axes.score_files import _feat, TRENDS
    from quality_axes.modulation import modulation_features
    from detector.segment import load_mono
    trends = json.load(open(TRENDS, encoding='utf-8'))
    fidx = {c: FEATURE_COLS.index(c) for c in cols if c in FEATURE_COLS}
    refs = [('otey/out_full/C4_v50.wav', 'Otey фізмодель'),
            ('audio/m1_patches/m1_M09-03_Toy-Piano.wav', 'm1 синт'),
            ('audio/42_bosendorfer_280vc_97.wav', 'Bösendorfer реал97'),
            ('audio/02_ukraina_tuned.wav', 'Ukraina реал'),
            ('audio/pianoteq/pianoteq_modartt9_01.wav', 'Pianoteq'),
            ('modelSamples/Salamander_Yam/C3_v25_p0.wav', 'Salamander семпл')]
    print("\n  --- референси: загальна якість (D) ---")
    print(f"  {'cal':>5} {'G':>5} {'D':>5}  label")
    for rel, lab in refs:
        rel = os.path.join(AUDIO_ROOT, rel)
        if not os.path.exists(rel):
            print(f"  {'--':>5}       {lab} (нема)"); continue
        try:
            xt, n = _feat(rel, trends)
            wav, sr = load_mono(rel)
            m = modulation_features(wav, sr)
            vec = [xt[fidx[c]] if c in fidx else m.get(c, np.nan) for c in cols]
            raw = float(model.predict(np.array(vec)[None, :])[0])
            cal = float(_apply_calibration(raw, cal_params))
            g = guardrail_penalty(rel, n)
            print(f"  {cal:5.1f} {g:5.1f} {np.clip(cal-g,0,99):5.1f}  {lab}  (n={n})")
        except Exception as e:  # noqa: BLE001
            print(f"  ERR  {lab}: {type(e).__name__} {e}")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    if '--save' in sys.argv:
        train_and_save()
        return
    if args:  # score files
        for f in args:
            r = overall_score(f)
            print(f"  D={r['overall']:5.1f}  (якість {r['quality_cal']:.1f} − G {r['guardrail']:.1f})"
                  f"  n={r['n_notes']}  {os.path.basename(f)}")
        return
    # report mode
    rows, y, anchor, w, groups, X = _load_with_liveness()
    oof = _oof(FULL_L_COLS, X, y, w, groups)
    cal_params = _fit_calibration(oof, y)
    cal = _apply_calibration(oof, cal_params)
    _report(y, oof, anchor, f"OVERALL raw {len(FULL_L_COLS)} фіч")
    print(f"  ρ(ALL)={spearmanr(oof, y).statistic:.3f}  "
          f"(stretch lo={cal_params['lo']:.1f} hi={cal_params['hi']:.1f})")
    M = np.column_stack([X[c] for c in FULL_L_COLS])
    model = xgb.XGBRegressor(**XGB).fit(M, y, sample_weight=w)
    _report_refs(model, FULL_L_COLS, cal_params)


if __name__ == '__main__':
    main()
