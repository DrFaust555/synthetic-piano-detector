"""Detect phaser / flanger / chorus artefacts in a recording.

A real acoustic piano has a stable harmonic spectrum that only DECAYS; phaser/flanger
add spectral notches that SWEEP periodically under an LFO (~0.1-3 Hz) — the "swoosh".
We detect that as **periodic modulation of the spectral SHAPE**:

  per-frame shape (level removed → ignores amplitude tremolo) → detrend over time
  (removes note decay) → modulation spectrum per band → energy concentrated at a single
  LFO frequency, COHERENT across many bands = phaser/flanger.

`phaser_index(x, sr)` → 0 (clean) … large (strong sweep).  File-level (the LFO spans
seconds, so it uses the whole signal, like the tuning features).
"""

from __future__ import annotations
import numpy as np
from numpy.fft import rfft, rfftfreq


def phaser_index(x: np.ndarray, sr: int, fmin: float = 150.0, fmax: float = 8000.0,
                 n_bands: int = 24, lfo_lo: float = 0.08, lfo_hi: float = 3.0) -> float:
    x = np.asarray(x, dtype=float)
    if x.ndim > 1:
        x = x.mean(axis=1)
    win, hop = 2048, 512
    n = len(x)
    nfr = 1 + (n - win) // hop if n >= win else 0
    if nfr < 32:                         # need several LFO periods to be meaningful
        return 0.0
    w = np.hanning(win)
    freqs = rfftfreq(win, 1.0 / sr)
    edges = np.geomspace(fmin, min(fmax, sr * 0.475), n_bands + 1)
    masks = [(freqs >= edges[b]) & (freqs < edges[b + 1]) for b in range(n_bands)]

    B = np.empty((nfr, n_bands), dtype=float)
    for i in range(nfr):
        seg = x[i * hop:i * hop + win] * w
        p = np.abs(rfft(seg)) ** 2
        for b, m in enumerate(masks):
            B[i, b] = 10.0 * np.log10(p[m].sum() + 1e-9) if m.any() else -90.0

    fps = sr / hop
    # only keep frames with audible energy (skip silence head/tail)
    lvl = B.mean(axis=1)
    keep = lvl > (lvl.max() - 40.0)
    if keep.sum() < 32:
        return 0.0
    B = B[keep]
    nfr = B.shape[0]

    # 1) per-frame shape only (remove overall level → ignore amplitude tremolo)
    B = B - B.mean(axis=1, keepdims=True)
    # 2) remove each band's time-mean (DC) — NO moving-average detrend (that would
    #    suppress the very LFO we want).  The note-decay drift lives near DC in the
    #    modulation spectrum and is excluded below by lfo_lo.
    B = B - B.mean(axis=0, keepdims=True)

    # 3) modulation spectrum per band
    F = np.abs(rfft(B * np.hanning(nfr)[:, None], axis=0))      # (modfreq, band)
    mf = rfftfreq(nfr, 1.0 / fps)
    lfo = (mf >= lfo_lo) & (mf <= lfo_hi)
    nz = mf >= lfo_lo                       # exclude decay ramp (near-DC) from the norm
    if lfo.sum() < 2 or not nz.any():
        return 0.0

    Flfo = F[lfo]
    band_sum = Flfo.sum(axis=1)             # total shape-modulation per LFO frequency
    peak = int(band_sum.argmax())
    # coherence: fraction of bands whose modulation peaks AT the dominant LFO freq
    band_peak = Flfo.argmax(axis=0)
    coherent = float(np.mean(np.abs(band_peak - peak) <= 1))    # 0..1 across bands
    # concentration: dominant-LFO power vs all shape-modulation power (LFO band)
    concentration = float(band_sum[peak] / (Flfo.sum() + 1e-9))
    return 100.0 * concentration * coherent     # ~0 clean … tens for strong sweep


def apply_flanger(x: np.ndarray, sr: int, depth_ms: float = 3.0, base_ms: float = 2.0,
                  lfo_hz: float = 0.5, mix: float = 0.7) -> np.ndarray:
    """Synthetic flanger (sweeping comb) — for testing the detector."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    t = np.arange(n)
    d = (base_ms + depth_ms * (0.5 + 0.5 * np.sin(2 * np.pi * lfo_hz * t / sr))) * sr / 1000.0
    idx = t - d
    i0 = np.floor(idx).astype(int)
    fr = idx - i0
    i0 = np.clip(i0, 0, n - 2)
    delayed = x[i0] * (1 - fr) + x[i0 + 1] * fr
    delayed[idx < 0] = 0.0
    y = (1 - mix) * x + mix * delayed
    return y / (np.max(np.abs(y)) + 1e-9)
