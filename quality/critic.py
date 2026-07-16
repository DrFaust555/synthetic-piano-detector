"""Real-vs-synthetic critic on the intrinsic feature set.

    python -m quality.critic

Target: synthetic (source ∈ {digital, model}) vs real (acoustic).  This is the
critic the FAUST generator needs (real piano = positive class; see the audio
dataset memory).  Reports CV ROC-AUC of a regularised logistic model + GBM, the
feature importances, and — specifically — whether the bass fundamental-centric
descriptors help separate real from synthetic.
"""

from __future__ import annotations
import os
import csv
import numpy as np
from scipy.stats import mannwhitneyu

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

from quality.pipeline import FEATURE_COLS

HERE = os.path.dirname(os.path.abspath(__file__))
FEATURES_CSV = os.path.join(HERE, 'cache', 'features.csv')
SUMMARY = os.path.join(HERE, 'cache', 'critic_summary.txt')
SEED = 0
BASS_KEYS = [c for c in FEATURE_COLS if c.startswith(('bass_', 'upper_', 'deepbass_'))]


def _load():
    rows = []
    with open(FEATURES_CSV, encoding='utf-8') as f:
        for r in csv.DictReader(f):
            rows.append(r)
    X = np.array([[float(r[c]) if r[c] != '' else np.nan for c in FEATURE_COLS]
                  for r in rows])
    y = np.array([1 if r.get('source', '') in ('digital', 'model') else 0 for r in rows])
    return rows, X, y


def _single_auc(x, y):
    ok = np.isfinite(x)
    a, b = x[ok & (y == 1)], x[ok & (y == 0)]
    if len(a) < 3 or len(b) < 3:
        return np.nan
    U, _ = mannwhitneyu(a, b, alternative='two-sided')
    return U / (len(a) * len(b))


def _pipe_lr():
    return Pipeline([
        ('impute', SimpleImputer(strategy='median', keep_empty_features=True)),
        ('scale', StandardScaler()),
        ('clf', LogisticRegression(penalty='elasticnet', l1_ratio=0.5, C=0.5,
                                   solver='saga', max_iter=20000,
                                   class_weight='balanced', random_state=SEED)),
    ])


def _pipe_gbm():
    return Pipeline([
        ('impute', SimpleImputer(strategy='median', keep_empty_features=True)),
        ('clf', GradientBoostingClassifier(n_estimators=200, max_depth=2,
                                           learning_rate=0.05, subsample=0.8,
                                           random_state=SEED)),
    ])


def main():
    rows, X, y = _load()
    lines = [f"Зразків: {len(y)}  (синтетичні: {int(y.sum())} [digital+model], "
             f"реальні acoustic: {int((1 - y).sum())})"]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for name, pipe in [('logreg', _pipe_lr()), ('gbm', _pipe_gbm())]:
        auc = cross_val_score(pipe, X, y, cv=cv, scoring='roc_auc')
        lines.append(f"{name:<8} CV ROC-AUC = {auc.mean():.3f} ± {auc.std():.3f}")

    # how well does the model do WITHOUT the bass/upper/deep features?
    keep = [i for i, c in enumerate(FEATURE_COLS) if c not in BASS_KEYS]
    auc_nob = cross_val_score(_pipe_lr(), X[:, keep], y, cv=cv, scoring='roc_auc')
    lines.append(f"logreg БЕЗ регістрових ознак ({len(keep)} ознак) "
                 f"CV ROC-AUC = {auc_nob.mean():.3f} ± {auc_nob.std():.3f}")

    # single-feature AUC of the fundamental-centric bass descriptors
    lines.append("\nОдноознакова AUC (синтетичне vs реальне) для регістрових ознак:")
    sa = []
    for c in BASS_KEYS:
        x = X[:, FEATURE_COLS.index(c)]
        sa.append((c, _single_auc(x, y)))
    for c, a in sorted(sa, key=lambda t: (-abs((t[1] or 0.5) - 0.5))):
        if a == a:
            lines.append(f"  {c:<26} AUC={a:.2f}")

    # full-model standardized coefficients
    pipe = _pipe_lr().fit(X, y)
    coef = pipe.named_steps['clf'].coef_.ravel()
    lines.append("\nЛог-рег коефіцієнти (+ => тягне до СИНТЕТИЧНОГО):")
    for i in np.argsort(-np.abs(coef))[:12]:
        if abs(coef[i]) > 1e-6:
            lines.append(f"  {FEATURE_COLS[i]:<26} {coef[i]:+.3f}")

    lines.append("\nЗастереження: синтетиків лише ~24 — CV-числа орієнтовні.")
    text = "\n".join(lines)
    with open(SUMMARY, 'w', encoding='utf-8') as f:
        f.write(text + "\n")
    print(text)
    print(f"\nПідсумок: {SUMMARY}")


if __name__ == '__main__':
    main()
