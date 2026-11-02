"""Sec. 4.5: how well the quality regressor predicts the human rating under instrument-grouped cross-validation.

Rated corpus data/corpus/features_cal.csv (rating_cal; 3299 files, of them 668 CCMusic anchor notes with listener MOS)
+ the 5 liveness descriptors of data/quality/liveness_cal.csv.  Out-of-fold predictions of the gradient-boosted
regressor (quality_axes/regressor.py: StratifiedGroupKFold, 5 folds, groups = instrument identity, anchors weighted
x3), Spearman rho with rating_cal over all files, for three descriptor sets:
    full 47            the 47 features (with the 7 tuning features of Group C)
    timbre 40          Groups A and B only (the timbre axis)
    timbre + liveness  the 40 + the 5 liveness descriptors (45)
Paper number (Sec. 4.5): "the regressor predicts human quality at rho ~ 0.54" = timbre + liveness (0.546 here; 0.543
when the models were fitted in July on another machine).  XGBoost runs with n_jobs=0, so the third decimal depends on
the number of CPU threads.

    python experiments/05_provenance_and_quality/quality_regressor_cv.py
        -> results/05_provenance_and_quality/quality_regressor_cv.md
"""
import os
import sys
import warnings
import numpy as np
from scipy.stats import spearmanr
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))   # repository root
from detector.paths import results_dir                                   # noqa: E402
from detector.pipeline import FEATURE_COLS                               # noqa: E402
from quality_axes.regressor import _oof                                  # noqa: E402
from quality_axes.timbre import _load_with_liveness, TIMBRE_COLS, TIMBRE_L_COLS   # noqa: E402

OUT = results_dir('05_provenance_and_quality')


def main():
    warnings.filterwarnings('ignore', message='The least populated class')
    rows, y, anchor, w, groups, X = _load_with_liveness()
    md = ['# Quality regressor, instrument-grouped cross-validation\n\n',
          f'{len(y)} rated files ({int(anchor.sum())} CCMusic anchor notes), {len(set(groups))} instrument groups.\n\n',
          '| descriptors | n | Spearman rho, all files | rho, anchors | rho, other files |\n|---|---|---|---|---|\n']
    for name, cols in (('full 47', FEATURE_COLS), ('timbre 40 (Groups A + B)', TIMBRE_COLS),
                       ('timbre + liveness 45', TIMBRE_L_COLS)):
        p = _oof(cols, X, y, w, groups)
        r = [spearmanr(p[m], y[m]).statistic for m in (np.ones(len(y), bool), anchor == 1, anchor == 0)]
        md.append(f'| {name} | {len(cols)} | {r[0]:.3f} | {r[1]:.3f} | {r[2]:.3f} |\n')
        print(md[-1], end='', flush=True)
    open(os.path.join(OUT, 'quality_regressor_cv.md'), 'w', encoding='utf-8').write(''.join(md))


if __name__ == '__main__':
    main()
