"""Shared spectral descriptors used by both the main pipeline and the bass probe.

Kept in their own module to avoid duplication and an import cycle
(pipeline ↔ bass_modulation ↔ analyze_top).
"""

from __future__ import annotations
import numpy as np
from numpy.fft import rfft, rfftfreq


def nonharm_ratio(x: np.ndarray, sr: int, f0: float, B: float,
                  band=(500.0, 3000.0)) -> float:
    """Fraction of mid-band energy NOT on the inharmonicity-corrected harmonic
    series — a crude proxy for non-harmonic content (phantom modes + noise +
    other notes). Higher = messier spectrum."""
    n = len(x)
    if n < int(0.1 * sr):
        return np.nan
    nfft = 1 << int(np.ceil(np.log2(min(n, sr))))
    nfft = min(max(nfft, 1 << 14), 1 << 16)
    c = min(n - 1, int(0.2 * sr))
    half = nfft // 2
    a, b = max(0, c - half), min(n, c + half)
    seg = np.zeros(nfft)
    seg[:b - a] = x[a:b]
    seg *= np.hanning(nfft)
    mag = np.abs(rfft(seg))
    freqs = rfftfreq(nfft, 1.0 / sr)
    df = freqs[1] - freqs[0]
    p = mag * mag
    bandmask = (freqs >= band[0]) & (freqs <= band[1])
    total = float(p[bandmask].sum())
    if total <= 0 or not bandmask.any():
        return np.nan
    harm = np.zeros(len(freqs), dtype=bool)
    k = 1
    while k <= 300:
        fk = k * f0 * np.sqrt(1.0 + B * k * k)
        if fk > band[1]:
            break
        if fk >= band[0] * 0.8:
            tol = max(3 * df, 0.006 * fk)
            lo, hi = int((fk - tol) / df), int((fk + tol) / df) + 1
            harm[max(0, lo):min(len(harm), hi)] = True
        k += 1
    harm_e = float(p[bandmask & harm].sum())
    return (total - harm_e) / total


def phantom_index(x: np.ndarray, sr: int, f0: float, B: float,
                  K: int = 14, band=(400.0, 3500.0)) -> float:
    """Longitudinal / phantom-partial prominence.

    A phantom partial sits at a combination frequency f_i + f_j of two transverse
    partials. For B=0, f_1+f_{m-1} = m*f0 = f_m, hiding under the harmonic.
    Inharmonicity (B>0) lifts the true transverse f_m = m*f0*sqrt(1+B m²) ABOVE
    f_1+f_{m-1}, opening a gap where a genuine phantom shows as a separate peak.
    Returns its median prominence (dB over the inter-partial trough), 0 if none.
    Exploratory proxy — noisy on polyphonic audio."""
    if not (B > 1e-5):
        return np.nan
    n = len(x)
    if n < int(0.12 * sr):
        return np.nan
    nfft = 1 << int(np.ceil(np.log2(min(n, sr))))
    nfft = min(max(nfft, 1 << 15), 1 << 16)
    c = min(n - 1, int(0.25 * sr))
    half = nfft // 2
    a, b = max(0, c - half), min(n, c + half)
    seg = np.zeros(nfft)
    seg[:b - a] = x[a:b]
    seg *= np.hanning(nfft)
    mag = np.abs(rfft(seg))
    df = sr / nfft
    if mag.max() <= 0:
        return np.nan

    def fk(k):
        return k * f0 * np.sqrt(1.0 + B * k * k)

    proms = []
    for m in range(3, K + 1):
        fm = fk(m)
        if fm < band[0] or fm > band[1]:
            continue
        pm = fk(1) + fk(m - 1)
        gap = fm - pm
        if gap < 3 * df:
            continue
        tol = max(2 * df, 0.3 * gap)
        lo, hi = int((pm - tol) / df), int((pm + tol) / df) + 1
        if hi - lo < 2 or hi >= len(mag):
            continue
        loc = lo + int(np.argmax(mag[lo:hi]))
        peak = float(mag[loc])
        bg_lo, bg_hi = int((pm + tol) / df), int((fm - tol) / df)
        if bg_hi - bg_lo >= 2 and bg_hi < len(mag):
            bg = float(np.median(mag[bg_lo:bg_hi]))
        else:
            bg = float(np.median(mag[max(0, loc - 20):loc + 20]))
        if bg > 0 and peak > 2.0 * bg:
            proms.append(20.0 * np.log10(peak / bg))
    return float(np.median(proms)) if proms else 0.0
