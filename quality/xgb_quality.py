"""Analytic XGBoost quality regressor on rating_cal — the working path.

    python -m quality.xgb_quality

Reads quality/cache/features_cal.csv (45 analytic features over all labelled files +
anchors), groups by INSTRUMENT identity (quality/grouping.group_key — not brand),
weights CCMusic anchors higher, and cross-validates an XGBoost regressor on rating_cal.
Reports the old-35 vs new-45 feature sets so we can see whether the added channel-robust
descriptors help, plus per-band error and feature importance.
"""

from __future__ import annotations
import os
import csv
import numpy as np

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr, pearsonr
import xgboost as xgb

from quality.pipeline import FEATURE_COLS
from quality.grouping import group_key

HERE = os.path.dirname(os.path.abspath(__file__))
FEATURES_CAL = os.path.join(HERE, 'cache', 'features_cal.csv')
ANCHOR_W = 3.0
SEED = 0

_NEW = ['tristim2', 'spec_irregularity', 'hf_decay_excess', 'centroid_var', 'partial_persist']
NEW_COLS = [f'{d}_{s}' for d in _NEW for s in ('med', 'iqr')]
OLD_COLS = [c for c in FEATURE_COLS if c not in NEW_COLS]

XGB = dict(n_estimators=500, max_depth=4, learning_rate=0.03, subsample=0.8,
           colsample_bytree=0.8, reg_lambda=2.0, objective='reg:pseudohubererror',
           random_state=SEED, n_jobs=0)


def _load():
    rows = list(csv.DictReader(open(FEATURES_CAL, encoding='utf-8')))
    y = np.array([float(r['rating_cal']) for r in rows])
    anchor = np.array([int(r.get('is_anchor', 0) or 0) for r in rows])
    w = np.where(anchor == 1, ANCHOR_W, 1.0)
    groups = np.array([group_key(r['file'], r.get('brand', '')) for r in rows])
    X = {c: np.array([float(r[c]) if r.get(c, '') not in ('', 'nan') else np.nan
                      for r in rows]) for c in FEATURE_COLS}
    return rows, y, anchor, w, groups, X


def _oof(cols, X, y, w, groups):
    M = np.column_stack([X[c] for c in cols])
    bins = np.clip((y // 10).astype(int), 0, 9)
    pred = np.zeros(len(y))
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    for tr, va in skf.split(M, bins, groups):
        m = xgb.XGBRegressor(**XGB)
        m.fit(M[tr], y[tr], sample_weight=w[tr])
        pred[va] = m.predict(M[va])
    return pred


def _report(y, p, anchor, name):
    def line(mask, tag):
        if mask.sum() < 5:
            return
        print(f"    {tag:<20} n={int(mask.sum()):4d}  MAE={mean_absolute_error(y[mask],p[mask]):5.1f}"
              f"  ρ={spearmanr(p[mask],y[mask]).statistic:.3f}  r={pearsonr(p[mask],y[mask])[0]:.3f}")
    print(f"  {name}:")
    line(np.ones(len(y), bool), 'ALL')
    line(anchor == 1, 'anchors (human GT)')
    line(anchor == 0, 'ours')


def main():
    if not os.path.exists(FEATURES_CAL):
        print(f"features_cal.csv ще нема — зачекай екстракцію.")
        return
    rows, y, anchor, w, groups, X = _load()
    print(f"Файлів: {len(y)}  анкерів: {int(anchor.sum())}  інструмент-груп: {len(set(groups))}\n")

    p_old = _oof(OLD_COLS, X, y, w, groups)
    _report(y, p_old, anchor, f"XGBoost OLD {len(OLD_COLS)} фіч (model-grouped)")
    p_new = _oof(FEATURE_COLS, X, y, w, groups)
    _report(y, p_new, anchor, f"XGBoost NEW {len(FEATURE_COLS)} фіч (model-grouped)")

    # per-band error (new model)
    print("\n  по тирах (NEW, ALL):")
    band = np.clip((y // 10) * 10, 0, 90).astype(int)
    for b in sorted(set(band)):
        msk = band == b
        if msk.sum() >= 5:
            print(f"    {b:>2}s  n={int(msk.sum()):4d}  MAE={mean_absolute_error(y[msk],p_new[msk]):5.1f}")

    # feature importance (full fit)
    M = np.column_stack([X[c] for c in FEATURE_COLS])
    imp = xgb.XGBRegressor(**XGB).fit(M, y, sample_weight=w).feature_importances_
    print("\n  топ-12 фіч за важливістю:")
    for i in np.argsort(-imp)[:12]:
        tag = ' *NEW' if FEATURE_COLS[i] in NEW_COLS else ''
        print(f"    {FEATURE_COLS[i]:<24} {imp[i]:.3f}{tag}")


if __name__ == '__main__':
    main()
