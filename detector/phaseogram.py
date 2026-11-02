"""Pitch-synchronous 'phaseogram' — the time-domain representation for realness_net.

Each note is resampled so one f0 period = P samples, then the first C cycles are stacked
into a (C, P) image.  Row c = the time-domain waveform shape of the c-th cycle; reading
DOWN the rows shows how the single-cycle shape EVOLVES over the note.  Real strings drift
(beating, inharmonic partial decay → the cycle shape keeps changing); sample-based digital
pianos loop/repeat → near-identical rows.  This fine phase/shape structure is exactly what
the 47 summary features and the log-mel both discard, so it is the marginal signal the
realness critic was blind to.
"""

from __future__ import annotations
import numpy as np

P = 256          # samples per cycle (within-cycle resolution; Nyquist = 128 harmonics)
C = 48           # number of cycles stacked (time evolution)


def compute_phaseogram(x: np.ndarray, sr: float, f0: float,
                       n_cycles: int = C, p: int = P) -> np.ndarray:
    """Return a (n_cycles, p) float32 phaseogram, peak-normalised, per-cycle DC-removed.

    Resample x so one f0 period spans exactly p samples (via linear interpolation — we
    only need the cycle SHAPE, not hi-fi), take the first n_cycles*p samples (zero-padded
    if the note is shorter), reshape to (n_cycles, p).
    """
    if f0 <= 0 or not np.isfinite(f0):
        return np.zeros((n_cycles, p), dtype=np.float32)
    period = sr / f0                                   # original samples per cycle
    if period < 2:
        return np.zeros((n_cycles, p), dtype=np.float32)
    need = n_cycles * p
    pos = np.arange(need) * (period / p)               # source index for each output sample
    x = x.astype(np.float32)
    y = np.interp(pos, np.arange(len(x)), x, left=0.0, right=0.0)
    y[pos > len(x) - 1] = 0.0                           # zero-pad past note end
    img = y.reshape(n_cycles, p)
    img = img - img.mean(axis=1, keepdims=True)        # per-cycle DC removal
    pk = np.max(np.abs(img))
    if pk > 1e-9:
        img = img / pk
    return img.astype(np.float32)
