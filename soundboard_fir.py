"""
FIR-convolution soundboard — uses the 4096-tap impulse response
derived from averaging 176 OGG samples (calib/soundboard_ir.npy).

This is the third path:

  - SoundboardModes (legacy biquad bank, hand-tuned 25 modes)
  - SoundboardData  (40 data-derived modes with per-note coupling)
  - SoundboardFIR   (direct convolution with measured average IR)

The FIR captures both discrete body modes AND the broadband diffuse
decay of the recording chain, whereas the modal version only fits
peaks.

Because direct sample-by-sample 4096-tap convolution is slow in Python,
this class buffers the input and flushes a batch fft-convolution at
the end of each render.  It supports the same `tick(x)->float`
interface as the other soundboards (returning zero during the tick),
plus a `flush()` that does the convolution; the engine should call
`flush()` once at the end of a render to retrieve the soundboard
contribution.
"""

from __future__ import annotations
import os
import numpy as np
from scipy.signal import fftconvolve

_IR_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'calib', 'soundboard_ir.npy')


class SoundboardFIR:
    """
    Streaming-API soundboard that records bridge input during tick()
    and replays the convolution result at flush() time.
    """

    def __init__(self, midi: int, sr: int, max_len: int = 8 * 48000):
        self.sr = sr
        self.midi = midi
        if os.path.exists(_IR_PATH):
            self.ir = np.load(_IR_PATH).astype(np.float64)
        else:
            self.ir = np.zeros(1)
        # Buffer of bridge inputs collected during tick()
        self._buf = np.zeros(max_len, dtype=np.float64)
        self._n = 0
        # Convolved output (lazily computed at flush)
        self._out = None
        self._out_idx = 0

    def tick(self, x: float) -> float:
        # We don't have the convolved output yet — return 0 and let the engine
        # add the convolved stream after the fact (via flush()).
        if self._n < len(self._buf):
            self._buf[self._n] = x
            self._n += 1
        return 0.0

    def flush(self) -> np.ndarray:
        """Convolve the buffered bridge stream with the IR and return the
        soundboard contribution (same length as the buffer)."""
        if self._n == 0 or len(self.ir) <= 1:
            return np.zeros(self._n)
        result = fftconvolve(self._buf[:self._n], self.ir, mode='full')[:self._n]
        return result.astype(np.float64)

    def n_collected(self) -> int:
        return self._n
