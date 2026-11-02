"""Hinrichsen entropy-minimisation tuning solver.

Faithful Python port of pianotuner/dsp/EntropySolver.cpp.

A well-tuned piano MINIMISES the Shannon entropy of the superposed power
spectrum of all its notes: coincident partials of different notes merge into one
tall peak, concentrating spectral energy (lower entropy).  Minimising entropy
over the 88 pitches reproduces the Railsback stretch as an EMERGENT result — no
beat-rate targets, no template.  A4 is held fixed to anchor global pitch.

Deterministic zero-temperature coordinate descent with an annealed (halving)
step, fixed proposal order → reproducible / testable (Hinrichsen arXiv:1203.5101;
Applied Acoustics 2020).
"""
from __future__ import annotations
import math
from dataclasses import dataclass

import numpy as np

from .noteref import (
    midi_to_freq, partial_freq, K_FIRST_PIANO_MIDI, K_NUM_PIANO_KEYS,
)


@dataclass
class EntropyParams:
    partials: int = 12          # partials summed per note
    amp_rolloff: float = 0.82   # partial amplitude a_n = rolloff^(n-1)
    bin_cents: float = 2.0      # log-frequency bin width (cents)
    sigma_cents: float = 9.0    # partial peak broadening (cents)
    min_hz: float = 25.0        # spectrum low edge (below A0)
    max_hz: float = 10000.0     # ignore partials above this
    max_shift_cents: float = 60.0    # clamp |offset| per note
    start_step_cents: float = 8.0    # initial descent step
    end_step_cents: float = 0.5      # final descent step (annealed by halving)
    sweeps_per_step: int = 25        # max sweeps at each step level


def _a_weight_linear(f: float) -> float:
    """A-weighting (IEC 61672) as a LINEAR amplitude gain.  The +2 dB norm const
    is dropped (cancels in the entropy's probability normalisation)."""
    f2 = f * f
    num = 12194.0 * 12194.0 * f2 * f2
    den = ((f2 + 20.6 * 20.6)
           * math.sqrt((f2 + 107.7 * 107.7) * (f2 + 737.9 * 737.9))
           * (f2 + 12194.0 * 12194.0))
    return num / den if den > 0.0 else 0.0


def _xlogx_arr(x: np.ndarray) -> np.ndarray:
    return np.where(x > 1.0e-300, x * np.log(np.where(x > 1.0e-300, x, 1.0)), 0.0)


class _Pre:
    """Precomputed partial bins + A-weighted powers for all 88 notes."""
    __slots__ = ("base_bin", "power", "nb", "S", "sig_bins", "hw", "bin_cents")

    def __init__(self, b_curve, a4_hz: float, p: EntropyParams):
        bins_per_oct = 1200.0 / p.bin_cents

        def bin_of(f: float) -> float:
            return bins_per_oct * math.log2(f / p.min_hz)

        self.bin_cents = p.bin_cents
        self.nb = int(math.ceil(bin_of(p.max_hz))) + 1
        self.sig_bins = p.sigma_cents / p.bin_cents
        self.hw = max(1, int(math.ceil(3.0 * self.sig_bins)))
        self.base_bin: list[np.ndarray] = []
        self.power: list[np.ndarray] = []
        self.S = 0.0

        for k in range(K_NUM_PIANO_KEYS):
            f0 = midi_to_freq(K_FIRST_PIANO_MIDI + k, a4_hz)
            bb, pw = [], []
            for n in range(1, p.partials + 1):
                f = partial_freq(f0, n, b_curve[k])
                if f < p.min_hz or f > p.max_hz:
                    continue
                amp = (p.amp_rolloff ** (n - 1)) * _a_weight_linear(f)
                power = amp * amp
                if power <= 0.0:
                    continue
                bb.append(bin_of(f))
                pw.append(power)
                self.S += power
            self.base_bin.append(np.asarray(bb, dtype=np.float64))
            self.power.append(np.asarray(pw, dtype=np.float64))


def _gauss_over(bins: np.ndarray, center: float, sig_bins: float) -> np.ndarray:
    d = (bins - center) / sig_bins
    return np.exp(-0.5 * d * d)


def _gnorm(center: float, sig_bins: float, hw: int) -> float:
    """Sum of the un-normalised gaussian over a peak's own ±hw window."""
    c0 = int(math.floor(center))
    bins = np.arange(c0 - hw, c0 + hw + 1, dtype=np.float64)
    s = float(_gauss_over(bins, center, sig_bins).sum())
    return s if s > 0.0 else 1.0


class EntropySolver:
    def __init__(self, params: EntropyParams | None = None):
        self.params = params or EntropyParams()

    # ---- rasterise the whole spectrum from a set of offsets into P ----------
    def _build_into(self, pre: _Pre, c: np.ndarray, P: np.ndarray) -> None:
        P.fill(0.0)
        hw = pre.hw
        for k in range(K_NUM_PIANO_KEYS):
            sh = c[k] / pre.bin_cents
            bb = pre.base_bin[k]
            pw = pre.power[k]
            for i in range(bb.shape[0]):
                center = bb[i] + sh
                c0 = int(math.floor(center))
                lo, hi = c0 - hw, c0 + hw
                bins = np.arange(lo, hi + 1, dtype=np.float64)
                g = _gauss_over(bins, center, pre.sig_bins)
                norm = float(g.sum())
                if norm <= 0.0:
                    norm = 1.0
                lo_c, hi_c = max(lo, 0), min(hi, pre.nb - 1)
                if lo_c > hi_c:
                    continue
                a, b = lo_c - lo, hi_c - lo
                P[lo_c:hi_c + 1] += pw[i] * g[a:b + 1] / norm

    # ---- dQ of retuning note k to newc (Q = sum_bins xlogx(P)) --------------
    def _move_dq(self, pre: _Pre, c: np.ndarray, P: np.ndarray,
                 k: int, newc: float, commit: bool) -> float:
        p = self.params
        hw = pre.hw
        dQ = 0.0
        sh_old = c[k] / pre.bin_cents
        sh_new = newc / pre.bin_cents
        bb = pre.base_bin[k]
        pw = pre.power[k]
        for i in range(bb.shape[0]):
            c_old = bb[i] + sh_old
            c_new = bb[i] + sh_new
            n_old = _gnorm(c_old, pre.sig_bins, hw)
            n_new = _gnorm(c_new, pre.sig_bins, hw)
            lo = int(math.floor(min(c_old, c_new))) - hw
            hi = int(math.ceil(max(c_old, c_new))) + hw
            lo = max(lo, 0)
            hi = min(hi, pre.nb - 1)
            if lo > hi:
                continue
            bins = np.arange(lo, hi + 1, dtype=np.float64)
            dp = pw[i] * (_gauss_over(bins, c_new, pre.sig_bins) / n_new
                          - _gauss_over(bins, c_old, pre.sig_bins) / n_old)
            seg = P[lo:hi + 1]
            new_seg = seg + dp
            dQ += float((_xlogx_arr(new_seg) - _xlogx_arr(seg)).sum())
            if commit:
                P[lo:hi + 1] = new_seg
        if commit:
            c[k] = newc
        return dQ

    def solve(self, b_curve, a4_hz: float = 440.0,
              warm_start=None) -> np.ndarray:
        """Stretch curve (cents from 12-TET, A4=0) minimising spectral entropy
        for the given per-key inharmonicity B curve.  Index 0 = A0 .. 87 = C8."""
        p = self.params
        pre = _Pre(b_curve, a4_hz, p)
        c = np.zeros(K_NUM_PIANO_KEYS, dtype=np.float64)  # start at equal temperament
        if pre.S <= 0.0:
            return c

        anchor = 69 - K_FIRST_PIANO_MIDI  # A4, held fixed (key 48)

        if warm_start is not None:
            for k in range(K_NUM_PIANO_KEYS):
                c[k] = min(max(float(warm_start[k]), -p.max_shift_cents),
                           p.max_shift_cents)
            c[anchor] = 0.0
        start_step = min(p.start_step_cents, 2.0) if warm_start is not None \
            else p.start_step_cents

        P = np.zeros(pre.nb, dtype=np.float64)

        step = start_step
        while step >= p.end_step_cents - 1.0e-9:
            for _sweep in range(p.sweeps_per_step):
                # Resync P each sweep: probe window is slightly wider than the
                # deposit window, so committed P accumulates tail-order phantom
                # weight; the rebuild bounds that drift to within one sweep.
                self._build_into(pre, c, P)
                improved = False
                for k in range(K_NUM_PIANO_KEYS):
                    if k == anchor:
                        continue
                    best_dq, best_c = 1.0e-9, c[k]
                    for direction in (-1, 1):
                        nc = min(max(c[k] + direction * step, -p.max_shift_cents),
                                 p.max_shift_cents)
                        if nc == c[k]:
                            continue
                        dq = self._move_dq(pre, c, P, k, nc, False)  # probe
                        if dq > best_dq:
                            best_dq, best_c = dq, nc
                    if best_c != c[k]:
                        self._move_dq(pre, c, P, k, best_c, True)
                        improved = True
                if not improved:
                    break
            step *= 0.5

        c[anchor] = 0.0
        return c

    def entropy_at(self, offsets, b_curve, a4_hz: float = 440.0) -> float:
        """Shannon entropy (nats) of the superposed spectrum at a given set of
        per-key offsets — lower = more concentrated / better tuned."""
        p = self.params
        pre = _Pre(b_curve, a4_hz, p)
        if pre.S <= 0.0:
            return 0.0
        P = np.zeros(pre.nb, dtype=np.float64)
        self._build_into(pre, np.asarray(offsets, dtype=np.float64), P)
        Q = float(_xlogx_arr(P).sum())
        return math.log(pre.S) - Q / pre.S  # Shannon entropy of P/S, in nats
