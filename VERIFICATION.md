# Verification

Use Python 3.10 with the dependencies in `requirements.txt`. Run the following
commands from the repository root:

```powershell
python verification/run_scripts.py --audio-root <audio-root>
python verification/smoke_training.py
```

Detection, listening and quality-regressor cross-validation analyses use the
shipped tables. Both Table 6 quality scripts require audio; include them with
`python verification/run_scripts.py --audio-root <audio-root> --audio`.
Complete reproduction of sampled play requires all 480 SFZ renders.

`smoke_training.py` checks training and prediction on small fixtures with real
metadata and features, synthetic images and one training epoch. These checks
test the software interfaces rather than detector accuracy. Full detector
training uses the scripts in `training/`.

Logs and JSON reports are written under `results/validation/`. Smoke-test
outputs use a separate directory through `SPD_RESULTS`.
