"""Core DSP primitives for the piano waveguide model."""

import numpy as np


class DelayLine:
    """
    Circular-buffer delay line with linear-interpolation fractional read.

    Convention (critical for tuning):
      write(x)          — stores x at write_ptr, then advances ptr
      read_from_out(d)  — returns sample written d samples ago
                          (d=1 → most recent write, d=2 → one before that …)

    So if you write() then immediately read_from_out(1), you get back
    exactly what you just wrote.
    """

    def __init__(self, max_len: int):
        self.buf = np.zeros(max_len + 2, dtype=np.float64)
        self.mask = len(self.buf) - 1  # only works if len is NOT power-of-2 in general
        self.N = len(self.buf)
        self.wp = 0

    def write(self, x: float):
        self.buf[self.wp] = x
        self.wp = (self.wp + 1) % self.N

    def read_from_out(self, delay: float) -> float:
        """Read with fractional delay via linear interpolation."""
        d = delay
        di = int(d)
        frac = d - di
        i0 = (self.wp - di) % self.N
        i1 = (self.wp - di - 1) % self.N
        return self.buf[i0] * (1.0 - frac) + self.buf[i1] * frac

    def read_from_out_int(self, delay: int) -> float:
        return self.buf[(self.wp - delay) % self.N]

    def accum(self, delay: int, value: float):
        """Add value at position 'delay' samples behind write head."""
        self.buf[(self.wp - delay) % self.N] += value

    def scale(self, delay: int, factor: float):
        self.buf[(self.wp - delay) % self.N] *= factor


class OnePole:
    """y[n] = (1-g)*x[n] + g*y[n-1].  g∈[0,1): higher → more LPF."""

    def __init__(self, g: float):
        self.g = g
        self.y1 = 0.0

    def tick(self, x: float) -> float:
        self.y1 = (1.0 - self.g) * x + self.g * self.y1
        return self.y1

    def group_delay(self, freq: float, sr: float) -> float:
        w = 2.0 * np.pi * freq / sr
        return self.g / (1.0 + self.g ** 2 - 2.0 * self.g * np.cos(w))


class DCBlock:
    """y[n] = x[n] - x[n-1] + R*y[n-1]."""

    def __init__(self, R: float = 0.9995):
        self.R = R
        self.x1 = 0.0
        self.y1 = 0.0

    def tick(self, x: float) -> float:
        y = x - self.x1 + self.R * self.y1
        self.x1 = x
        self.y1 = y
        return y


class Biquad:
    """Transposed Direct Form II."""

    def __init__(self, b0=1., b1=0., b2=0., a1=0., a2=0.):
        self.b0, self.b1, self.b2 = b0, b1, b2
        self.a1, self.a2 = a1, a2
        self.s1 = self.s2 = 0.0

    def tick(self, x: float) -> float:
        y = self.b0 * x + self.s1
        self.s1 = self.b1 * x - self.a1 * y + self.s2
        self.s2 = self.b2 * x - self.a2 * y
        return y

    def reset(self):
        self.s1 = self.s2 = 0.0

    @staticmethod
    def bandpass(f: float, Q: float, sr: float):
        w = 2 * np.pi * f / sr
        a = np.sin(w) / (2 * Q)
        c = np.cos(w)
        a0 = 1 + a
        return Biquad(a / a0, 0., -a / a0, -2 * c / a0, (1 - a) / a0)


class Allpass1:
    """First-order allpass y[n] = c*x[n] + x[n-1] - c*y[n-1]."""

    def __init__(self, c: float):
        self.c = c
        self.x1 = 0.0
        self.y1 = 0.0

    def tick(self, x: float) -> float:
        y = self.c * x + self.x1 - self.c * self.y1
        self.x1 = x
        self.y1 = y
        return y
