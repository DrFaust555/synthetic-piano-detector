"""Timbre axis (+ liveness) of the quality tool: the quality regressor retrained without the 7 tuning features
(Group C), so a well-built but out-of-tune instrument keeps its timbre score; axis B+L adds the 5 file-level
modulation descriptors (liveness_features.LIVENESS_COLS).

    python -m quality_axes.timbre          # grouped-CV Spearman: full 47 / timbre 40 / timbre + liveness 45
    python -m quality_axes.timbre --save   # fit axis B+L on all rows -> results/quality_models/timbre_quality.json

The report mode gives the rho of Sec. 4.5: timbre + liveness (45 descriptors) under instrument-grouped CV, rho ~ 0.54.
"""
from __future__ import annotations
import os
import csv
import numpy as np

from detector.pipeline import FEATURE_COLS
from quality_axes.regressor import _load, _oof, _report
from quality_axes.liveness_features import LIVENESS_COLS, OUT as LIVENESS_CSV

# The 7 tuning descriptors → axis A, dropped from the timbre-quality model.
TUNING_COLS = ['detune_final', 'unison_index', 'global_index', 'med_spread_cents',
               'frac_split', 'global_scatter', 'ref_offset_abs']
TIMBRE_COLS = [c for c in FEATURE_COLS if c not in TUNING_COLS]
# Axis B+L = timbre + liveness (modulation) descriptors.
TIMBRE_L_COLS = TIMBRE_COLS + LIVENESS_COLS


def _liveness_map() -> dict[str, dict]:
    if not os.path.exists(LIVENESS_CSV):
        return {}
    out = {}
    for r in csv.DictReader(open(LIVENESS_CSV, encoding='utf-8')):
        out[r['file']] = r
    return out


def _load_with_liveness():
    """_load() + liveness columns merged in (by file), NaN where missing."""
    rows, y, anchor, w, groups, X = _load()
    lv = _liveness_map()
    for c in LIVENESS_COLS:
        vals = []
        for r in rows:
            rec = lv.get(r['file'])
            v = rec.get(c, '') if rec else ''
            vals.append(float(v) if v not in ('', 'nan') else np.nan)
        X[c] = np.array(vals)
    return rows, y, anchor, w, groups, X


def train_and_save(cols=None, path: str | None = None):
    """Fit axis B+L (timbre+liveness) on ALL labelled data and persist it (model
    + feature order + 0-99 calibration).  Returns (model, cols)."""
    import json
    import xgboost as xgb
    from quality_axes.regressor import XGB, _oof
    from quality_axes.calibration import fit_calibration, save_calibration
    cols = cols or TIMBRE_L_COLS
    rows, y, anchor, w, groups, X = _load_with_liveness()
    # honest OOF for the calibration (no leakage), then fit on all data
    oof = _oof(cols, X, y, w, groups)
    cal = fit_calibration(oof, y)
    M = np.column_stack([X[c] for c in cols])
    model = xgb.XGBRegressor(**XGB).fit(M, y, sample_weight=w)
    if path is None:
        path = save_path('timbre_quality.json')
    model.save_model(path)
    json.dump(cols, open(path + '.cols', 'w'))
    save_calibration(cal, save_path('timbre_quality_iso.json'))
    print(f"saved axis-B+L model ({len(cols)} feats) + calibration → {path}")
    return model, cols


def _validate_refs(model, cols):
    """Score reference sources with the fitted model (timbre from score_files._feat
    + liveness from modulation_features), to see the real/synth spread."""
    import json
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
    print("\n  --- референси (вісь B+L) ---")
    for rel, lab in refs:
        rel = os.path.join(AUDIO_ROOT, rel)
        if not os.path.exists(rel):
            print(f"  {'--':>6}  {lab} (нема)"); continue
        try:
            xt, n = _feat(rel, trends)
            wav, sr = load_mono(rel)
            m = modulation_features(wav, sr)
            vec = [xt[fidx[c]] if c in fidx else m.get(c, np.nan) for c in cols]
            p = float(model.predict(np.array(vec)[None, :])[0])
            print(f"  {p:6.1f}  {lab}  (n={n})")
        except Exception as e:  # noqa: BLE001
            print(f"  ERR  {lab}: {type(e).__name__} {e}")


def main():
    import sys
    if '--save' in sys.argv:
        model, cols = train_and_save()
        _validate_refs(model, cols)
        return
    rows, y, anchor, w, groups, X = _load_with_liveness()
    print(f"Файлів: {len(y)}  анкерів: {int(anchor.sum())}  "
          f"інструмент-груп: {len(set(groups))}")
    nlive = int(np.isfinite(X['beat_flatness']).sum())
    print(f"timbre-only: {len(TIMBRE_COLS)} фіч; +liveness: {len(TIMBRE_L_COLS)} "
          f"(живість покрито {nlive}/{len(y)})\n")

    p_full = _oof(FEATURE_COLS, X, y, w, groups)
    _report(y, p_full, anchor, f"FULL {len(FEATURE_COLS)} (стрій+тембр)")
    p_timb = _oof(TIMBRE_COLS, X, y, w, groups)
    _report(y, p_timb, anchor, f"TIMBRE {len(TIMBRE_COLS)} (вісь B)")
    p_tl = _oof(TIMBRE_L_COLS, X, y, w, groups)
    _report(y, p_tl, anchor, f"TIMBRE+L {len(TIMBRE_L_COLS)} (вісь B+L)")

    import xgboost as xgb
    from quality_axes.regressor import XGB
    M = np.column_stack([X[c] for c in TIMBRE_L_COLS])
    imp = xgb.XGBRegressor(**XGB).fit(M, y, sample_weight=w).feature_importances_
    print("\n  топ-12 B+L фіч:")
    for i in np.argsort(-imp)[:12]:
        tag = ' *LIVE' if TIMBRE_L_COLS[i] in LIVENESS_COLS else ''
        print(f"    {TIMBRE_L_COLS[i]:<24} {imp[i]:.3f}{tag}")


from quality_axes.models import save_path   # noqa: E402
from detector.paths import AUDIO_ROOT   # noqa: E402

if __name__ == '__main__':
    main()
