"""Phaseograms of the isolated notes for the phaseogram CNN (needs the audio and the MIT IR Survey in
<SPD_AUDIO>/audio/_ir/mit_survey/Audio/): one window from the strike, the nominal pitch
(detector.segment.build_segments(isolated=True, f0_hint=...)), detector.phaseogram.compute_phaseogram.

Universe (protocol.cnn.universe): CCMusic notes, SoundFont single-note renders, Otey notes, the four libraries'
note files and the 30 listening-test clips.  For the signal-domain augmentation every note also gets channel copies
made on the SIGNAL: a measured room impulse response of the MIT IR Survey + white noise at SNR ~ U(30, 50) dB re
the note, each copy segmented again as the detector would:
  train copies (K_TRAIN): impulse responses with an EVEN index of the survey,
  test copy (1):          impulse responses with an ODD index (rooms never used in training).
Deterministic per (note, copy).

    python training/notes_phaseograms.py [--workers 16] [--limit N]

Writes results/cnn/cache_iso/phaseo_iso.npy (N, C, P), phaseo_iso_room_train.npy (N, K_TRAIN, C, P),
phaseo_iso_room_test.npy (N, C, P), meta_iso.csv.
"""
from __future__ import annotations
import os
import sys
import zlib
import time
import argparse
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..')))   # repository root
from detector.paths import DATA, RESULTS, AUDIO_ROOT, results_dir   # noqa: E402
OUT = results_dir('training')
import protocol.cnn as CL  # noqa: E402

CACHE_ISO = os.path.join(RESULTS, 'cnn', 'cache_iso')
C, P = 48, 256
K_TRAIN = 3
SNR = (30.0, 50.0)


def _channel(x, sr, key, test):
    import protocol.corpus as V
    from scipy.signal import fftconvolve
    irs = V._ir_list()
    pool = [i for i in range(len(irs)) if (i % 2 == 1) == test]
    rng = np.random.default_rng(zlib.crc32(key.encode('utf-8')))
    i = pool[int(rng.integers(len(pool)))]
    snr = float(rng.uniform(*SNR))
    y = fftconvolve(x, V._ir(i, sr))[:len(x)]
    y *= (np.max(np.abs(x)) + 1e-30) / (np.max(np.abs(y)) + 1e-30)
    o = V.onset_index(y, sr)
    seg = y[o:o + int(1.0 * sr)]
    rms = float(np.sqrt(np.mean(seg ** 2))) if len(seg) else float(np.sqrt(np.mean(y ** 2)))
    return y + rng.standard_normal(len(y)) * rms * 10 ** (-snr / 20)


def _img(x, sr, f0_nom):
    from detector.segment import build_segments
    from detector.phaseogram import compute_phaseogram
    segs, _ = build_segments(x, sr, isolated=True, f0_hint=f0_nom)
    if not segs or not (24.0 < segs[0].f0 < 4500.0):
        return None, np.nan
    s = segs[0]
    return compute_phaseogram(s.x, sr, s.f0), float(s.f0)


def one(args):
    path, file, midi = args
    from detector.segment import load_mono
    out = {'ok': False, 'f0': np.nan, 'error': ''}
    imgs = {'clean': None, 'train': [None] * K_TRAIN, 'test': None}
    try:
        x, sr = load_mono(path)
        f0_nom = 440.0 * 2.0 ** ((float(midi) - 69) / 12.0)
        imgs['clean'], out['f0'] = _img(x, sr, f0_nom)
        for k in range(K_TRAIN):
            imgs['train'][k], _ = _img(_channel(x, sr, f'phtrain{k}:{file}', False), sr, f0_nom)
        imgs['test'], _ = _img(_channel(x, sr, f'phtest:{file}', True), sr, f0_nom)
        out['ok'] = imgs['clean'] is not None and imgs['test'] is not None and all(i is not None for i in imgs['train'])
    except Exception as e:
        out['error'] = repr(e)[:200]
    return out, imgs


def main(workers=16, limit=None):
    from multiprocessing import Pool
    t0 = time.time()
    os.makedirs(CACHE_ISO, exist_ok=True)
    u = CL.universe()
    if limit:
        u = pd.concat([g.head(max(1, limit // 5)) for _, g in u.groupby('source')]).sort_index()
    jobs = list(zip(u['path'], u['file'], u['nominal_midi']))
    N = len(u)
    clean = np.zeros((N, C, P), np.float32)
    train = np.zeros((N, K_TRAIN, C, P), np.float32)
    test = np.zeros((N, C, P), np.float32)
    meta = []
    with Pool(workers) as pool:
        for i, (m, im) in enumerate(pool.imap(one, jobs, chunksize=4)):
            meta.append(m)
            if m['ok']:
                clean[i], test[i] = im['clean'], im['test']
                for k in range(K_TRAIN):
                    train[i, k] = im['train'][k]
    meta = pd.concat([u.reset_index(drop=True), pd.DataFrame(meta)], axis=1)
    np.save(os.path.join(CACHE_ISO, 'phaseo_iso.npy'), clean)
    np.save(os.path.join(CACHE_ISO, 'phaseo_iso_room_train.npy'), train)
    np.save(os.path.join(CACHE_ISO, 'phaseo_iso_room_test.npy'), test)
    meta.to_csv(os.path.join(CACHE_ISO, 'meta_iso.csv'), index=False)
    print(f'{N} files, {int(meta.ok.sum())} usable, {int((meta.error != "").sum())} errors, '
          f'{time.time() - t0:.0f} s -> {CACHE_ISO}', flush=True)
    print(meta.groupby('source').ok.agg(['sum', 'size']).to_string(), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--limit', type=int, default=None)
    a = ap.parse_args()
    main(a.workers, a.limit)
