"""
Data-driven soundboard — replacement for the hand-tuned biquad bank in
soundboard.py.

Modes (f, Q) come from `calib/soundboard_ir.json` (extracted by
`calib/soundboard_ir.py` from averaging 176 OGG samples).  Per-note
*coupling* between bridge position and each mode is also stored there
and is interpolated linearly to MIDI notes outside the observed set.

The interface (`tick(x) -> float`) is identical to the original
`Soundboard`, so it drops into note_engine.py with one swap.
"""

from __future__ import annotations
import json
import os
import numpy as np
from dsp import Biquad

_JSON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'calib', 'soundboard_ir.json')


def _load_ir_data() -> dict:
    if not os.path.exists(_JSON_PATH):
        return {}
    with open(_JSON_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


def _coupling_for_midi(coupling_map: dict, midi: int) -> np.ndarray:
    """Linear interpolate per-mode coupling for any midi from the observed map."""
    if not coupling_map:
        return None
    keys = sorted(int(k) for k in coupling_map.keys())
    n_modes = len(coupling_map[str(keys[0])])
    if midi <= keys[0]:
        return np.array(coupling_map[str(keys[0])])
    if midi >= keys[-1]:
        return np.array(coupling_map[str(keys[-1])])
    for i in range(len(keys) - 1):
        if keys[i] <= midi <= keys[i + 1]:
            m0, m1 = keys[i], keys[i + 1]
            v0 = np.array(coupling_map[str(m0)])
            v1 = np.array(coupling_map[str(m1)])
            t = (midi - m0) / (m1 - m0)
            return v0 * (1 - t) + v1 * t
    return np.array(coupling_map[str(keys[0])])


class SoundboardData:
    """Same API as Soundboard, but modes from data + per-note coupling."""

    def __init__(self, midi: int, sr: int, *,
                 coupling_floor_db: float = -15.0,
                 coupling_ceiling_db: float = 12.0,
                 base_gain_db: float = -15.0,
                 q_min: float = 4.0,
                 q_max: float = 15.0):
        """
        midi              MIDI note of this voice (selects per-note coupling)
        coupling_floor_db Clamp coupling so it doesn't drive a mode to -inf
        coupling_ceiling_db  Clamp the upper end too (LSQ can shoot to +50 dB)
        base_gain_db      How loud the average mode is at "coupling=0 dB"
        q_min, q_max      Clamp the data-derived Q values — too high Q rings forever,
                          too low Q has no body character
        """
        ir = _load_ir_data()
        self.filters = []
        self.gains = []
        if not ir.get('modes'):
            return

        coupling = _coupling_for_midi(ir.get('coupling_per_midi', {}), midi)

        for k, mode in enumerate(ir['modes']):
            f, Q = mode['f'], mode['Q']
            if not (20 < f < sr * 0.45):
                continue
            Q = float(np.clip(Q, q_min, q_max))
            self.filters.append(Biquad.bandpass(f, Q, sr))
            # gain = base × coupling-derived multiplier × normalized mode_gain
            mode_g_db = mode.get('gain_db', 0.0)
            mode_g_db = mode_g_db - max(m['gain_db'] for m in ir['modes'])  # peak=0
            if coupling is not None and k < len(coupling):
                c_db = float(np.clip(coupling[k], coupling_floor_db, coupling_ceiling_db))
            else:
                c_db = 0.0
            total_db = base_gain_db + mode_g_db + c_db
            g = 10.0 ** (total_db / 20.0)
            self.gains.append(g)

    def tick(self, x: float) -> float:
        out = 0.0
        for filt, g in zip(self.filters, self.gains):
            out += filt.tick(x) * g
        return out
