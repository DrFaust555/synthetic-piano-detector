"""
SoundboardEQ — FIR convolution + a parallel cascade of tunable bandpass
biquads (an additive parametric EQ on top of the body IR).

Rationale: the FIR derived by harmonic-notching average captures
broadband body+room colouration, but is biased by the residual bridge
force spectrum we couldn't fully cancel.  A small parallel set of
biquads at fixed piano-body-typical centres lets the calibrator
*correct* the FIR (positive gain adds resonance, negative gain
notches it out) without throwing away the data-derived baseline.

Centre frequencies follow piano-acoustics literature (cf. Chaigne,
Bank–Chabassier): rib/bridge body modes cluster from 50–200 Hz with
sparser peaks up to ~3 kHz.  We use 12 log-spaced bands.

The biquad bank reads its per-band gains (and shared Q) from globals
`eq_gain_0` .. `eq_gain_11` and `eq_q`.
"""

from __future__ import annotations
import os
import numpy as np
from scipy.signal import fftconvolve
from dsp import Biquad

# 32 log-spaced band centres from 40 Hz to 12 kHz (~1/3-octave spacing).
import numpy as _np
EQ_CENTRES_HZ = [float(round(f, 1)) for f in
                  _np.geomspace(40.0, 12000.0, 32)]

_IR_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'calib', 'soundboard_ir.npy')


class SoundboardEQ:
    """FIR (buffered) + parallel IIR resonator bank (per-sample)."""

    def __init__(self, midi: int, sr: int, max_len: int = 8 * 48000):
        from calib.store import get_calibration
        cal = get_calibration()
        self.sr = sr
        self.midi = midi
        # FIR baseline
        if os.path.exists(_IR_PATH):
            self.ir = np.load(_IR_PATH).astype(np.float64)
        else:
            self.ir = np.zeros(1)
        self._buf = np.zeros(max_len, dtype=np.float64)
        self._n = 0

        # Parallel IIR resonators.  Low Q (broad bands) — narrow resonators ring
        # for tens of ms and become audibly metallic during sustain.
        q_shared = float(np.clip(cal.get_global('eq_q'), 1.0, 4.0))
        self.filters = []
        self.gains = []
        for k, f in enumerate(EQ_CENTRES_HZ):
            if not (20 < f < sr * 0.45):
                continue
            self.filters.append(Biquad.bandpass(f, q_shared, sr))
            mult = cal.get_global(f'eq_gain_{k}')
            # Clamp the effective EQ contribution to ±8 dB so a resonance
            # can't dominate the sustain.  Optimiser's log-multiplier still
            # has room above the clamp, but the audible effect is bounded.
            g_db = float(np.clip(20.0 * np.log10(max(mult, 1e-3)), -8.0, 8.0))
            self.gains.append(10.0 ** (g_db / 20.0) - 1.0)
            # subtract 1 so multiplier=1.0 -> gain=0 (EQ disabled)

    def tick(self, x: float) -> float:
        # Buffer for FIR
        if self._n < len(self._buf):
            self._buf[self._n] = x
            self._n += 1
        # Parallel IIR (immediate, returns sample contribution)
        out = 0.0
        for filt, g in zip(self.filters, self.gains):
            if g == 0.0:
                # Still run the filter so its state stays consistent
                filt.tick(x)
            else:
                out += filt.tick(x) * g
        return out

    def flush(self) -> np.ndarray:
        """Same as SoundboardFIR.flush(); returns FIR contribution
        sample-aligned with the input."""
        if self._n == 0 or len(self.ir) <= 1:
            return np.zeros(self._n)
        return fftconvolve(self._buf[:self._n], self.ir, mode='full')[:self._n]

    def n_collected(self) -> int:
        return self._n
