"""47 features of the room copies made by training/notes_phaseograms.py (the same signals: its _channel is
deterministic per note and copy), so that the signal-domain augmentation reaches BOTH branches of the fusion and
every detector can be tested in unseen rooms.  Features as for the isolated corpus: protocol.common.note_raw_iso
(one window from the strike, nominal pitch) + aggregate_file (data/corpus/register_trends_cal.json).

    python training/notes_room_features.py [--workers 16]

Writes results/cnn/cache_iso/feat_room_train.npy (N, K_TRAIN, 47) and feat_room_test.npy (N, 47), rows =
meta_iso.csv.
"""
from __future__ import annotations
import os
import sys
import time
import argparse
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..')))   # repository root
from detector.paths import DATA, RESULTS, AUDIO_ROOT, results_dir   # noqa: E402
OUT = results_dir('training')
import protocol.cnn as CL  # noqa: E402
import training.notes_phaseograms as B  # noqa: E402


def _feats(y, sr, path, midi):
    """features of the in-memory signal y (load_mono wrapped, as protocol.corpus.extract_variant)."""
    import protocol.common as L
    import detector.segment as S
    import detector.pipeline as QP
    orig = S.__dict__.get('_room_orig_load') or S.load_mono
    S._room_orig_load = orig
    S.load_mono = QP.load_mono = (lambda p: (y, sr))
    try:
        meta, notes, tuning = L.note_raw_iso(QP, path, midi=midi)
        f = QP.aggregate_file(notes, tuning, L._trends())
    finally:
        S.load_mono = QP.load_mono = orig
    return np.array([f.get(c, np.nan) for c in CL.FEATURE_COLS], dtype=float)


def one(args):
    path, file, midi = args
    from detector.segment import load_mono
    tr = np.full((B.K_TRAIN, len(CL.FEATURE_COLS)), np.nan)
    te = np.full(len(CL.FEATURE_COLS), np.nan)
    err = ''
    try:
        x, sr = load_mono(path)
        for k in range(B.K_TRAIN):
            tr[k] = _feats(B._channel(x, sr, f'phtrain{k}:{file}', False), sr, path, midi)
        te = _feats(B._channel(x, sr, f'phtest:{file}', True), sr, path, midi)
    except Exception as e:
        err = repr(e)[:200]
    return tr, te, err


def main(workers=16):
    from multiprocessing import Pool
    t0 = time.time()
    meta = pd.read_csv(os.path.join(B.CACHE_ISO, 'meta_iso.csv'))
    jobs = list(zip(meta['path'], meta['file'], meta['nominal_midi']))
    N = len(meta)
    tr = np.full((N, B.K_TRAIN, len(CL.FEATURE_COLS)), np.nan, np.float32)
    te = np.full((N, len(CL.FEATURE_COLS)), np.nan, np.float32)
    errs = 0
    with Pool(workers) as pool:
        for i, (a, b, e) in enumerate(pool.imap(one, jobs, chunksize=4)):
            tr[i], te[i] = a, b
            errs += bool(e)
    np.save(os.path.join(B.CACHE_ISO, 'feat_room_train.npy'), tr)
    np.save(os.path.join(B.CACHE_ISO, 'feat_room_test.npy'), te)
    print(f'{N} files, {errs} errors, {time.time() - t0:.0f} s; non-missing share train {np.isfinite(tr).mean():.2f}, '
          f'test {np.isfinite(te).mean():.2f}', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=16)
    main(ap.parse_args().workers)
