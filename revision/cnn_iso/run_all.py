"""Format-matched phaseogram-CNN study -- ONE command for every number in RESULTS_CNN_ISO.md.

    cd <project root>
    python Stattia/revision/cnn_iso/run_all.py [--workers 6] [--threads 2]

1. build_phaseograms.py : phaseogram cache (cnn_iso/cache/; skipped when up to date).
2. train_cnn_iso.py     : 69 model fits (3 models x T1/T2 x 5 folds, constrained f0; 9 LOGO fits for the clips;
                          30 fits of the MIDI 48-76 / unconstrained-f0 variant).  Resumable: fits already in
                          cnn_iso/jobs/ are not repeated.  CPU only, about 5.5-7 h with 6 workers x 2 threads.
3. evaluate_cnn_iso.py  : all tables -> RESULTS_CNN_ISO.md, oof_scores_cnn_iso.csv, comparisons_cnn_iso.csv,
                          stimuli_cnn_iso_scores.csv, key_numbers_cnn_iso.json.
Reads the repository and Stattia/revision/analysis/out/ only; writes only into Stattia/revision/cnn_iso/.
"""
from __future__ import annotations

import os
import sys
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--phaseo-workers', type=int, default=12)
    ap.add_argument('--skip-train', action='store_true', help='evaluate whatever fits exist')
    a = ap.parse_args()
    os.environ['CNN_ISO_THREADS'] = str(a.threads)
    import build_phaseograms
    build_phaseograms.build(a.phaseo_workers)
    if not a.skip_train:
        import train_cnn_iso
        train_cnn_iso.main(a.workers, a.threads)
    import evaluate_cnn_iso
    evaluate_cnn_iso.main()


if __name__ == '__main__':
    main()
