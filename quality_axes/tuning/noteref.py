"""Note/cents math + stiff-string physics + Railsback stretch reference.

Faithful Python port of pianotuner's NoteUtil.h, StiffString.h and
StretchReference.cpp.  All doubles → Python float (both IEEE-754 double), so the
port is bit-comparable to the C++ within FP rounding.
"""
from __future__ import annotations
import math

# Piano key range: A0 (MIDI 21) .. C8 (MIDI 108) == 88 keys.
K_FIRST_PIANO_MIDI = 21   # A0
K_LAST_PIANO_MIDI = 108   # C8
K_NUM_PIANO_KEYS = K_LAST_PIANO_MIDI - K_FIRST_PIANO_MIDI + 1  # 88


# ---- NoteUtil.h -------------------------------------------------------------
def midi_to_freq(midi: float, a4_hz: float = 440.0) -> float:
    """12-TET frequency of a (possibly fractional) MIDI note, A4 = a4_hz."""
    return a4_hz * (2.0 ** ((midi - 69.0) / 12.0))


def freq_to_midi(freq_hz: float, a4_hz: float = 440.0) -> float:
    """Inverse: fractional MIDI number of a frequency."""
    return 69.0 + 12.0 * math.log2(freq_hz / a4_hz)


def cents_between(f1: float, f2: float) -> float:
    """Signed interval in cents (positive = f2 sharp of f1)."""
    return 1200.0 * math.log2(f2 / f1)


def nearest_note(freq_hz: float, a4_hz: float = 440.0) -> tuple[int, float]:
    """(nearest integer MIDI, signed cents deviation in (-50, +50])."""
    m = freq_to_midi(freq_hz, a4_hz)
    nearest = int(round(m))
    return nearest, 100.0 * (m - nearest)


# ---- StiffString.h ----------------------------------------------------------
def partial_freq(f0: float, n: int, B: float) -> float:
    """n-th partial (n=1 is fundamental) of a note with ideal f0 and inharm B:
    f_n = n * f0 * sqrt(1 + B * n^2)   (Fletcher 1964)."""
    return n * f0 * math.sqrt(1.0 + B * float(n) * n)


def fundamental_from_partial(fn: float, n: int, B: float) -> float:
    """Recover ideal f0 from a measured partial f_n at index n given B."""
    return fn / (n * math.sqrt(1.0 + B * float(n) * n))


# Average-piano inharmonicity reference B(f), calibrated to published
# measurements (Schuck & Young 1943 / Young 1952 / Fletcher-Blackham-Stratton
# 1962).  Anchor table, interpolated log-log in frequency; ends hold their value.
_REF_B_ANCHORS = (
    (27.5, 3.0e-4),    # A0 — wound bass sits above the valley
    (55.0, 1.8e-4),    # A1 — valley (long wound strings)
    (110.0, 2.2e-4),   # A2
    (220.0, 3.5e-4),   # A3
    (440.0, 7.5e-4),   # A4
    (880.0, 1.8e-3),   # A5
    (1760.0, 5.0e-3),  # A6
    (4186.0, 1.5e-2),  # C8
)


def reference_B(f: float) -> float:
    """Calibrated average-piano B(f), log-log interpolated between anchors."""
    a = _REF_B_ANCHORS
    n = len(a)
    if f <= a[0][0]:
        return a[0][1]
    if f >= a[n - 1][0]:
        return a[n - 1][1]
    hi = 1
    while hi < n - 1 and a[hi][0] < f:
        hi += 1
    t = math.log(f / a[hi - 1][0]) / math.log(a[hi][0] / a[hi - 1][0])
    return a[hi - 1][1] * ((a[hi][1] / a[hi - 1][1]) ** t)


# ---- StretchReference.cpp ---------------------------------------------------
def build_reference_stretch_cents() -> list[float]:
    """Railsback stretch curve (cents from 12-TET, A4=0) for MIDI 0..127,
    derived from the physical reference B(f) via per-semitone octave-closure.
    Bass flat (negative), treble sharp (positive).  Port of
    buildReferenceStretchCents()."""
    cents = [0.0] * 128

    def B_at(midi: int) -> float:
        m = (K_FIRST_PIANO_MIDI if midi < K_FIRST_PIANO_MIDI
             else (K_LAST_PIANO_MIDI if midi > K_LAST_PIANO_MIDI else midi))
        return reference_B(midi_to_freq(float(m)))

    # 2:1 octave closure (upper fundamental vs lower note's 2nd partial).
    def inc21(m: int) -> float:
        bl, bu = B_at(m), B_at(m + 12)
        return 1200.0 * math.log2(math.sqrt(1.0 + 4.0 * bl)
                                  / math.sqrt(1.0 + bu)) / 12.0

    # 4:2 octave (bass): quadruples B's effect → deeper flattening.
    def inc42(m: int) -> float:
        bl, bu = B_at(m), B_at(m + 12)
        return 1200.0 * math.log2(math.sqrt(1.0 + 16.0 * bl)
                                  / math.sqrt(1.0 + 4.0 * bu)) / 12.0

    cents[69] = 0.0  # A4 anchor
    for m in range(70, 128):
        cents[m] = cents[m - 1] + inc21(m - 12)
    for m in range(68, -1, -1):
        cents[m] = cents[m + 1] - inc42(m)
    return cents
