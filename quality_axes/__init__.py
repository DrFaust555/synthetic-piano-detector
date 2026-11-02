"""Quality tool of Sec. 3.8: tuning, timbre and liveness axes, the overall score, and its artifact guardrail.

Version behind the printed quality column of Table 6 (overall score D):
    combiner.py      overall quality D = calibrated regressor on 52 descriptors - artifact guardrail
    judge.py         fusion of D with the timbre + liveness regressor, minus a dock for dead modulation
    timbre.py        timbre (+ liveness) regressor on the corpus quality ratings; grouped cross-validation report
    regressor.py     corpus table, instrument groups, XGBoost settings, grouped out-of-fold predictions
    calibration.py   isotonic calibration + affine stretch to the 0-99 scale
    liveness_features.py, modulation.py   file-level modulation descriptors (beat_mod, beat_flatness, ...)
    liveness.py      the I/Q liveness critic (P(synthetic) from the modulation of each partial)
    objectives.py    phaser penalty, aversiveness and liveness critic per file
    aversiveness.py, psychoacoustics.py, phaser.py   artifact measures of the guardrail
    score_files.py   the 47-feature vector of a file with the corpus register trends
Current version (Sec. 3.8 as described in the paper):
    timbre_model.py  timbre regressor on the 40 features of Groups A and B with its calibration
    report.py        tuning / timbre / liveness / overall axes of one recording
    tuning/          tuning axis: stiff-string fit, entropy-minimizing target, score 100 exp(-d / 12 cents)
models.py names where fitted models are read (data/quality/models) and where retrained ones are written.
"""
