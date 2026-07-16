"""Waveform-shape complexity: inflection density per period, normalised by f0.

Magnitude-spectral features (centroid, partial levels) can MATCH between a real piano
and the physical model, yet the model's WAVEFORM is too smooth (aligned phases, little
fine structure) → it sounds synthetic.  Count the inflection points (2nd-derivative sign
changes) in a sustained window and divide by the number of periods (window_s · f0) →
"inflections per period" = how wiggly one cycle is = how many partials genuinely shape it.

Real > model in bass/mid (model ratio ~0.54-0.66, smoother); ~equal in treble where even
the real note is nearly a sine (few partials below Nyquist).
"""

from __future__ import annotations
import numpy as np
from scipy.signal import butter, sosfiltfilt


def inflection_density(wave: np.ndarray, sr: int, f0: float,
                       skip_s: float = 0.05, dur_s: float = 0.4,
                       lp_hz: float = 7000.0) -> float:
    w = np.asarray(wave, dtype=float)
    if w.ndim > 1:
        w = w.mean(axis=1)
    if f0 <= 0 or len(w) < int(0.12 * sr):
        return float('nan')
    a = int(skip_s * sr)
    b = min(len(w), a + int(dur_s * sr))
    seg = w[a:b]
    if len(seg) < int(0.06 * sr):
        seg = w[:min(len(w), int(0.2 * sr))]
    seg = seg / (np.max(np.abs(seg)) + 1e-12)
    # low-pass to drop HF mic/room noise (which would add spurious inflections); keep the
    # partials that actually shape the waveform
    fc = min(lp_hz, sr * 0.45)
    sos = butter(4, fc / (sr * 0.5), btype='low', output='sos')
    seg = sosfiltfilt(sos, seg)
    d2 = np.diff(seg, 2)
    s = np.sign(d2)
    s[s == 0] = 1
    n_infl = int(np.sum(s[1:] != s[:-1]))
    periods = (len(seg) / sr) * f0
    return float(n_infl / periods) if periods > 0 else float('nan')


if __name__ == '__main__':
    import os, glob
    from detune.segment import load_mono
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    NOTE_MIDI = {'B0': 23, 'C1': 24, 'C2': 36, 'E2': 40, 'G1': 31, 'C3': 48,
                 'C4': 60, 'E4': 64, 'G5': 79, 'E6': 88, 'C5': 72, 'G3': 55}
    def f0(n): return 440.0 * 2 ** ((NOTE_MIDI[n] - 69) / 12.0)
    print(f"{'note':5s}{'f0':>7s}{'real':>8s}{'otey':>8s}{'ratio':>8s}")
    ratios = []
    for n in ['B0', 'G1', 'C2', 'E6']:
        rs = glob.glob(os.path.join(ROOT, f'oggSamples/{n}_v127_PedalOFF.ogg'))
        ot = glob.glob(os.path.join(ROOT, f'otey/out_full/{n}_v127.wav'))
        if not rs or not ot:
            continue
        xr, sr = load_mono(rs[0]); dr = inflection_density(xr, sr, f0(n))
        xo, so = load_mono(ot[0]); do = inflection_density(xo, so, f0(n))
        r = do / dr if dr else float('nan')
        ratios.append(r)
        print(f"{n:5s}{f0(n):7.0f}{dr:8.2f}{do:8.2f}{r:8.2f}")
    if ratios:
        print(f"\nmean model/real ratio = {np.nanmean(ratios):.2f}")
