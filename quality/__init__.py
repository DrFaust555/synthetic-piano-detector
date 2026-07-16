"""Quality-segment analysis: which intrinsic timbre/tuning traits mark the 90+ tier.

Reuses the project's vetted extractors (``calib.features.extract`` for per-note
timbre, ``detune.analyze.analyze`` for tuning) and adds:
  * a register-normalised per-note descriptor reducer,
  * a dataset-wide statistical contrast (90+ vs 70-89 and 90+ vs rest),
  * a regularised "90+-likeness" predictor.

Run from the project root as modules:
    python -m quality.extract_features
    python -m quality.analyze_top
    python -m quality.train
    python -m quality.predict <wav>
"""

import sys as _sys

# Console on Windows defaults to cp1252; force UTF-8 so Ukrainian output works.
for _s in (_sys.stdout, _sys.stderr):
    try:
        _s.reconfigure(encoding='utf-8')
    except Exception:
        pass
