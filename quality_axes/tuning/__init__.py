"""Tuning axis of the quality tool (Sec. 3.8): Python port of the tuning DSP core.
  noteref.py  note / cents math, stiff-string physics, reference stretch curve
  entropy.py  entropy-minimizing tuning solver (Hinrichsen)
  measure.py  recording -> tuning score 0-100
"""
from .noteref import (
    midi_to_freq, freq_to_midi, cents_between, nearest_note,
    partial_freq, fundamental_from_partial, reference_B,
    build_reference_stretch_cents, K_FIRST_PIANO_MIDI, K_LAST_PIANO_MIDI,
    K_NUM_PIANO_KEYS,
)
from .entropy import EntropySolver, EntropyParams

__all__ = [
    "midi_to_freq", "freq_to_midi", "cents_between", "nearest_note",
    "partial_freq", "fundamental_from_partial", "reference_B",
    "build_reference_stretch_cents", "EntropySolver", "EntropyParams",
    "K_FIRST_PIANO_MIDI", "K_LAST_PIANO_MIDI", "K_NUM_PIANO_KEYS",
]
