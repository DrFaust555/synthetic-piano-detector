"""
Calibration storage — JSON-backed per-anchor parameter multipliers.

The model has a small set of physically meaningful parameters per note
(t60, lp coefficients, inharmonicity, hammer K/p, bridge admittance, ...).
We store *multipliers* on top of the formulas in `params.py`, anchored at
a sparse set of MIDI notes; intermediate notes interpolate piecewise-linearly
in log space.

Why multipliers rather than absolute values?  The hand-tuned formulas in
`params.py` capture the broad shape (bass heavier, treble brighter, etc.)
and only need correction.  Optimising a multiplier per anchor keeps the
search space small and the parameter surface monotone in the right
directions, which matters for gradient descent.

The "global" block holds parameters that don't make sense per-note
(overall output level, soundboard gain trim).
"""

from __future__ import annotations
import json
import os
from dataclasses import dataclass, field, asdict
import numpy as np

# Names of per-note tunable multipliers (default 1.0).
PER_NOTE_KEYS = [
    't60_scale',         # T60 (decay) — pushes loss_g toward 1
    'lp1_scale',         # bridge-loss LPF amount (high-frequency loss)
    'B_scale',           # inharmonicity coefficient
    'hammer_K_scale',    # hammer stiffness (attack hardness)
    'hammer_p_scale',    # hammer power-law exponent (attack shape)
    'hammer_mass_scale', # hammer mass (contact duration)
    'bridge_adm_scale',  # bridge admittance (string→soundboard coupling)
    'strike_pos_scale',  # strike ratio (which partials excited)
    'vel_gain',          # overall amplitude per note
    'long_gain_scale',   # nonlinear longitudinal coupling strength
    'long_freq_scale',   # longitudinal resonance frequency multiplier
    'long_q_scale',      # longitudinal resonance Q
]

GLOBAL_KEYS = [
    'output_gain',
    'soundboard_gain',
    'direct_mix',     # weight of raw bridge velocity in output (1.0 = original)
    'board_mix',      # weight of soundboard radiation in output (1.0 = original)
    'use_data_board', # legacy flag (kept for backward compat in JSON files)
    'board_kind',     # 0=legacy biquad bank, 1=modal-data, 2=FIR-only, 3=FIR+EQ
    'eq_q',           # shared Q for the parametric-EQ bandpass bank (board_kind=3)
    'long_lp_cutoff', # LPF cutoff Hz applied to longitudinal phantom signal
    'long_mode_decay',# per-mode gain decay exponent: g_k = 1/k^decay
    'out_hf_cutoff',  # final-output one-pole LPF cutoff Hz (HF tilt down)
    'out_lf_boost',   # final-output low-shelf boost (linear, ~1.0 = flat)
    'out_hf_keep',    # fraction of HF (above cutoff) to retain; 0.1 = -20 dB cut
    'lf_res_freq',    # artificial low-frequency resonator centre Hz
    'lf_res_Q',       # LF resonator Q
    'lf_res_gain',    # LF resonator output gain (added to direct+FIR)
    'lf_res2_freq',   # second LF resonator (sub-bass "barrel") centre Hz
    'lf_res2_Q',
    'lf_res2_gain',
    'hammer_thump_gain', # sample-based hammer thump injected at note-on
    'body_thump_gain',   # 600ms LP-filtered bass body response injected at note-on
    'voice_main_0_scale',   # multiplier on string 0 detune (base -0.6)
    'voice_main_1_scale',   # base +0.4
    'voice_main_2_scale',   # base +1.5
    'voice_main_3_scale',   # base -1.0 (4th string)
    'voice_amp_0',          # per-string output amplitude (bridge mix)
    'voice_amp_1',
    'voice_amp_2',
    'voice_amp_3',
] + [f'sb_mode_{i}' for i in range(26)]  \
  + [f'eq_gain_{i}' for i in range(32)]  # per-band tunable EQ on top of FIR

DEFAULT_ANCHORS = [24, 36, 48, 60, 72, 84, 96]  # C1, C2, C3, C4, C5, C6, C7


@dataclass
class Calibration:
    """In-memory representation; round-trips through JSON."""
    anchors: list[dict] = field(default_factory=list)
    global_: dict = field(default_factory=dict)
    version: int = 1

    @staticmethod
    def default() -> 'Calibration':
        c = Calibration()
        for m in DEFAULT_ANCHORS:
            c.anchors.append({'midi': m, **{k: 1.0 for k in PER_NOTE_KEYS}})
        c.global_ = {k: 1.0 for k in GLOBAL_KEYS}
        return c

    # ---- per-note interpolation ----
    def get(self, midi: int, key: str) -> float:
        ms = np.array([a['midi'] for a in self.anchors])
        vs = np.array([a.get(key, 1.0) for a in self.anchors])
        if midi <= ms[0]:
            return float(vs[0])
        if midi >= ms[-1]:
            return float(vs[-1])
        # piecewise-linear in log(value) so 0.1 ↔ 10 interpolate symmetrically
        idx = int(np.searchsorted(ms, midi))
        m0, m1 = ms[idx - 1], ms[idx]
        v0, v1 = vs[idx - 1], vs[idx]
        t = (midi - m0) / (m1 - m0)
        return float(np.exp((1 - t) * np.log(max(v0, 1e-6))
                            + t * np.log(max(v1, 1e-6))))

    def get_global(self, key: str) -> float:
        return float(self.global_.get(key, 1.0))

    def anchor_for(self, midi: int) -> dict:
        for a in self.anchors:
            if a['midi'] == midi:
                return a
        raise KeyError(f'No anchor at MIDI {midi}')

    # ---- serialisation ----
    def to_dict(self) -> dict:
        return {'version': self.version, 'anchors': self.anchors, 'global': self.global_}

    @staticmethod
    def from_dict(d: dict) -> 'Calibration':
        c = Calibration()
        c.version = d.get('version', 1)
        c.anchors = d.get('anchors', [])
        c.global_ = d.get('global', {})
        # Fill missing keys with 1.0 for forward-compat.
        for a in c.anchors:
            for k in PER_NOTE_KEYS:
                a.setdefault(k, 1.0)
        for k in GLOBAL_KEYS:
            c.global_.setdefault(k, 1.0)
        return c

    def save(self, path: str):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2)

    @staticmethod
    def load(path: str) -> 'Calibration':
        if not os.path.exists(path):
            return Calibration.default()
        with open(path, 'r', encoding='utf-8') as f:
            return Calibration.from_dict(json.load(f))


# Module-level cache so params.py doesn't re-read the file every call.
_CACHED: Calibration | None = None
_CACHED_PATH: str | None = None


def get_calibration(path: str | None = None) -> Calibration:
    global _CACHED, _CACHED_PATH
    if path is None:
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(here, 'calibration.json')
    if _CACHED is None or _CACHED_PATH != path:
        _CACHED = Calibration.load(path)
        _CACHED_PATH = path
    return _CACHED


def reload_calibration():
    """Force re-read on next get_calibration() (used after writes)."""
    global _CACHED, _CACHED_PATH
    _CACHED = None
    _CACHED_PATH = None
