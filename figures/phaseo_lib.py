"""Shared helpers for the revised phaseogram figure and the drift-regularity test.

Phaseograms are built EXACTLY as the detector builds them (REL/DEV
quality/build_wave_cache.py::_phaseograms and quality/extract_samplelib_features.py):

    x, sr = detune.segment.load_mono(path)            # stereo -> mono mean, native sr
    segs, _ = detune.segment.build_segments(x, sr)    # spectral-flux onset, -45 dB trim,
                                                      # 30 ms attack skip, stiff-string f0
    segs = [s for s in segs if 24 < s.f0 < 4500]
    s = max(segs, key=lambda s: s.energy)             # primary pitch of an isolated note
    img = quality.phasefeat.compute_phaseogram(s.x, sr, s.f0)   # P=256, C=48

The detector's own modules are imported read-only; nothing in the repository is modified.
For an isolated-note file build_segments works in 'single/chord' mode and may return up
to three pitch candidates for the same window; the primary pitch is the one with the
largest `energy` (extras are stored with 0.7x energy), which is also the rule used by
extract_samplelib_features.py for the sample libraries.
"""
from __future__ import annotations

import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))      # project root
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from detune.segment import load_mono, build_segments, freq_to_midi   # noqa: E402
from quality.phasefeat import compute_phaseogram, C, P                # noqa: E402

PC = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
MAX_LAG = 32          # primary lag search window, samples (= +/-45 deg of the period)


# --------------------------------------------------------------------------- phaseogram
def detector_phaseogram(path: str) -> dict:
    """Detector-exact phaseogram of the primary note in an isolated-note file."""
    x, sr = load_mono(path)
    return phaseogram_from_audio(x, sr)


def phaseogram_from_audio(x: np.ndarray, sr: int) -> dict:
    """Same as detector_phaseogram, for an already-loaded mono signal."""
    segs, info = build_segments(x, sr)
    segs_ok = [s for s in segs if 24.0 < s.f0 < 4500.0]
    if not segs_ok:
        return {'ok': False, 'n_segs': len(segs), 'mode': info.get('mode')}
    s = max(segs_ok, key=lambda s: s.energy)
    img = compute_phaseogram(s.x, sr, s.f0)
    n_win = int(round(C * sr / s.f0))                       # source samples under the image
    win = s.x[:n_win]
    # Locate row 0 in the file: s.x is an exact copy of x[start:...] (onset -> -45 dB trim
    # -> +30 ms), so find the first exact match after the spectral-flux onset.
    start = None
    o = int(s.onset)
    m = min(64, len(s.x))
    hits = np.flatnonzero(x[o:o + int(1.0 * sr)] == s.x[0]) + o
    for h in hits:
        if h + m <= len(x) and np.array_equal(x[h:h + m], s.x[:m]):
            start = int(h)
            break
    return {'ok': True, 'img': img, 'f0': float(s.f0), 'B': float(s.B),
            'midi_est': float(freq_to_midi(s.f0)), 'sr': int(sr),
            'n_segs': len(segs), 'mode': info.get('mode'),
            'onset_flux_s': o / sr,
            'row0_s': (start / sr) if start is not None else np.nan,
            'win_rms': float(np.sqrt(np.mean(win ** 2))) if len(win) else np.nan,
            'len_s': float(len(s.x) / sr)}


def noise_floor_rms(x: np.ndarray, sr: int, frame_s: float = 0.02, q: float = 5.0) -> float:
    """Background level: q-th percentile of 20-ms frame RMS over the whole file."""
    w = max(1, int(frame_s * sr))
    n = len(x) // w
    if n < 5:
        return np.nan
    r = np.sqrt(np.mean(x[:n * w].reshape(n, w) ** 2, axis=1))
    r = r[r > 0]
    return float(np.percentile(r, q)) if len(r) else np.nan


# --------------------------------------------------------------------------- drift metrics
def _parabolic(y_m, y_0, y_p):
    """Vertex offset of the parabola through 3 points; only for a true local maximum
    (y_0 >= both neighbours, concave).  At the edge of a restricted search window the
    integer argmax need not be a local maximum; then no sub-sample correction (0)."""
    d = y_m - 2.0 * y_0 + y_p
    if y_0 < y_m or y_0 < y_p or d >= -1e-15:
        return 0.0
    return float(np.clip(0.5 * (y_m - y_p) / d, -0.5, 0.5))


def row_pair_lags(img: np.ndarray, max_lag: int | None = MAX_LAG, rel_floor: float = 1e-3):
    """Per row pair (c, c+1): circular cross-correlation lag (sub-sample, parabolic),
    zero-lag Pearson correlation and peak normalised cross-correlation.

    Lag convention: positive lag = row c+1 is row c shifted to LATER phase
    (b[n] ~ a[n - lag]); a steady positive lag means the waveform period is longer than
    the resampling period (true frequency below the f0 grid).
    max_lag=None searches the whole circle; otherwise only |lag| <= max_lag.
    Pairs where either row is (near) silent are returned as NaN.
    """
    Cn, Pn = img.shape
    img = img.astype(np.float64)
    rms = np.sqrt(np.mean(img ** 2, axis=1))
    floor = rel_floor * (rms.max() + 1e-30)
    F = np.fft.rfft(img, axis=1)
    k = np.arange(Pn)
    kk = np.where(k > Pn // 2, k - Pn, k)                 # signed lag per bin
    lags = np.full(Cn - 1, np.nan)
    r0 = np.full(Cn - 1, np.nan)
    rpk = np.full(Cn - 1, np.nan)
    for c in range(Cn - 1):
        if rms[c] < floor or rms[c + 1] < floor:
            continue
        a, b = img[c], img[c + 1]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        xc = np.fft.irfft(np.conj(F[c]) * F[c + 1], n=Pn) / (na * nb)   # xc[k]=sum a[n] b[n+k]
        if max_lag is None:
            cand = np.arange(Pn)
        else:
            cand = np.where(np.abs(kk) <= max_lag)[0]
        i0 = int(cand[np.argmax(xc[cand])])
        off = _parabolic(xc[(i0 - 1) % Pn], xc[i0], xc[(i0 + 1) % Pn])
        lag = kk[i0] + off
        if lag > Pn / 2:
            lag -= Pn
        elif lag <= -Pn / 2:
            lag += Pn
        lags[c] = lag
        r0[c] = float(np.dot(a, b) / (na * nb))           # rows are already DC-free
        rpk[c] = float(xc[i0])
    return lags, r0, rpk


def drift_summary(img: np.ndarray, max_lag: int | None = MAX_LAG, min_pairs: int = 20) -> dict:
    """(i) mean drift rate, (ii) irregularity, (iii) mean row-to-row correlation."""
    lags, r0, rpk = row_pair_lags(img, max_lag=max_lag)
    ok = np.isfinite(lags)
    out = {'n_pairs': int(ok.sum())}
    if ok.sum() < min_pairs:
        out.update(drift=np.nan, irreg=np.nan, irreg_mad=np.nan, wander=np.nan,
                   r_row=np.nan, r_aligned=np.nan, jump_frac=np.nan, drift_slope=np.nan,
                   jitter=np.nan, slow=np.nan, irreg_early=np.nan, irreg_late=np.nan)
        return out
    c = np.arange(len(lags))[ok]
    L = lags[ok]
    coef = np.polyfit(c, L, 1)
    res = L - np.polyval(coef, c)
    # cumulative displacement wander (auxiliary): RMS of detrended cumulative shift
    X = np.concatenate([[0.0], np.cumsum(L)])
    t = np.arange(len(X))
    Xr = X - np.polyval(np.polyfit(t, X, 1), t)
    # decomposition of the detrended lag residual: row-to-row jitter vs slow wander
    jitter = float(np.std(np.diff(res), ddof=1) / np.sqrt(2.0))
    slow = (float(np.std(np.convolve(res, np.ones(7) / 7.0, mode='valid'), ddof=1))
            if len(res) >= 14 else np.nan)

    def _half(sel):
        cc, LL = c[sel], L[sel]
        if len(cc) < 10:
            return np.nan
        return float(np.std(LL - np.polyval(np.polyfit(cc, LL, 1), cc), ddof=1))
    out.update(
        jitter=jitter, slow=slow,
        irreg_early=_half(c < 23), irreg_late=_half(c >= 24),
        drift=float(np.mean(L)),                       # (i) samples per cycle (of P=256)
        drift_slope=float(coef[0]),                    # change of lag per cycle (trend)
        irreg=float(np.std(res, ddof=1)),              # (ii) std of detrended lag increments
        irreg_mad=float(1.4826 * np.median(np.abs(res - np.median(res)))),
        wander=float(np.sqrt(np.mean(Xr ** 2))),
        r_row=float(np.nanmean(r0)),                   # (iii) zero-lag row-to-row correlation
        r_aligned=float(np.nanmean(rpk)),              # after best circular alignment
        jump_frac=float(np.mean(np.abs(L) > 8.0)),     # pairs with |lag| > 8 samples
    )
    return out


def drift_to_cents(d_samples_per_cycle: float, p: int = P) -> float:
    """Frequency offset equivalent to a steady lag of d samples per cycle (cents)."""
    return float(-1200.0 * np.log2(1.0 + d_samples_per_cycle / p))


# --------------------------------------------------------------------------- note names
def sci_to_midi(name: str) -> int | None:
    """'C4', 'D#4', 'Ds4', 'Ab1', 'Bb0', 'A#3' -> MIDI."""
    m = re.fullmatch(r'([A-G])(#|s|b)?(-?\d)', name)
    if not m:
        return None
    pc = PC[m.group(1)] + {'#': 1, 's': 1, 'b': -1, None: 0}[m.group(2)]
    return 12 * (int(m.group(3)) + 1) + pc


def helmholtz_to_midi(tok: str) -> int | None:
    """CCMusic pitch token (Helmholtz) -> MIDI.  'c1' = C4 (middle C), 'C' = C2,
    'C1' = C1, 'A2' = A0, 'c1s-d1b' = C#4.  Sharps take the first spelling."""
    t = tok.split('-')[0]
    m = re.fullmatch(r'([A-Ga-g])(\d)?(s)?', t)
    if not m:
        return None
    letter, num, sharp = m.group(1), m.group(2), m.group(3)
    if letter.isupper():
        octv = 2 - int(num) if num else 2
    else:
        octv = 3 + int(num) if num else 3
    return 12 * (octv + 1) + PC[letter.upper()] + (1 if sharp else 0)


def harmonic_drift(img: np.ndarray, kmax: int = 8) -> dict:
    """Per within-cycle harmonic k (DFT bin k of each row): mean shift per cycle in
    samples (same sign convention as row_pair_lags) and the bin's share of row power.
    Tuning offset shifts every k by the same amount; stiffness shifts bin k by about
    -128*B*k^2 samples per cycle (partial k at k*f0*sqrt(1+B*k^2))."""
    Pn = img.shape[1]
    F = np.fft.rfft(img.astype(np.float64), axis=1)
    pw = np.sum(np.abs(F[:, 1:kmax + 1]) ** 2, axis=0)
    out = {}
    for k in range(1, kmax + 1):
        z = np.sum(np.conj(F[:-1, k]) * F[1:, k])
        out[k] = {'shift_per_cycle': float(-np.angle(z) * Pn / (2 * np.pi * k)),
                  'power_share': float(pw[k - 1] / pw.sum())}
    return out
