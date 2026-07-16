"""Soundboard resonator — bandpass filter bank for body resonances."""

from dsp import Biquad


class Soundboard:
    def __init__(self, modes, sr):
        self.filters = []
        self.gains = []
        for f, Q, g in modes:
            if 20 < f < sr * 0.45:
                self.filters.append(Biquad.bandpass(f, Q, sr))
                self.gains.append(g)

    def tick(self, x: float) -> float:
        out = 0.0
        for filt, g in zip(self.filters, self.gains):
            out += filt.tick(x) * g
        return out
