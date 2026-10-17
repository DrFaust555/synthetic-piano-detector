"""Make a data_v2 feature CSV with EXACTLY the corpus pipeline (so run_all.py can use it).

    python Stattia/revision/analysis/extract_features_v2.py <metadata.csv> <out.csv> [--workers 8]

<metadata.csv> has one row per audio file: a column `path` (absolute, or relative to the repo root)
plus any metadata columns, which are copied through unchanged -- e.g. for
  digital_v2_features.csv : brand, model, sound_method (sampling / modelling / hybrid ...)[, group]
  sf2_perf_features.csv   : instrument (= SoundFont name as in audio/SynthPianoSamples/rendered/<instrument>)
  otey_perf_features.csv  : (nothing required)
Features: quality.pipeline.note_raw_descriptors + aggregate_file with quality/cache/register_trends_cal.json
(the same call that produced features_cal / synth_features; verified to reproduce stored rows to < 5e-7).
Write <out.csv> into Stattia/revision/data_v2/ with the file name expected by protocol_v2.py.
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import pandas as pd             # noqa: E402
import pv2_lib as L             # noqa: E402


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    meta = pd.read_csv(sys.argv[1])
    out = sys.argv[2]
    workers = int(sys.argv[sys.argv.index('--workers') + 1]) if '--workers' in sys.argv else 8
    if 'path' not in meta:
        raise SystemExit('metadata CSV needs a `path` column')
    paths = [p if os.path.isabs(p) else os.path.join(L.ROOT, p) for p in meta['path'].astype(str)]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise SystemExit(f'{len(missing)} file(s) not found, e.g. {missing[0]}')
    tag = 'v2_' + os.path.splitext(os.path.basename(out))[0]
    f = L.extract_cached(paths, tag, workers=workers)
    res = pd.concat([meta.drop(columns=['path']).reset_index(drop=True),
                     f.drop(columns=['sig']).reset_index(drop=True)], axis=1)
    cols = ['file'] + [c for c in meta.columns if c != 'path'] + ['n_notes', 'duration_s', 'mode', 'error'] + L.FEATURE_COLS
    res[cols].to_csv(out, index=False, float_format='%.6f')
    bad = (res['error'].fillna('') != '').sum()
    print(f'wrote {out}: {len(res)} rows ({bad} extraction errors)')


if __name__ == '__main__':
    main()
