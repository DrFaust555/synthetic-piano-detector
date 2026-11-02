"""Artifact penalties and the file-level I/Q liveness critic used by both quality versions."""
from __future__ import annotations
import numpy as np
from detector.segment import load_mono
from quality_axes.phaser import phaser_index as _phaser_index
from quality_axes.aversiveness import aversiveness_features as _aver_features

PHASER_THRESH = 3.5
PHASER_K = 6.0


def iq_synth(path: str) -> float:
    """File-level P(synth) from the TRAINED I/Q modulation critic (quality/iq_critic):
    median over the file's notes of p_synth_iq.  0 = real-like modulation patterns,
    ~1 = dead synthetic modulation (Otey ≈0.99, real/Pianoteq/samplers ≈0.0).

    Its native domain is ISOLATED notes (exactly what the synthesis optimizer renders),
    and it is constrained to the I/Q modulation domain only — timbre/decay/level/channel
    cannot enter. NaN if no scorable notes."""
    from detector.segment import build_segments
    from quality_axes.liveness import p_synth_iq
    x, sr = load_mono(path)
    segs, _ = build_segments(x, sr, max_notes=24)
    ps = []
    for s in segs:
        if not (24.0 < s.f0 < 4500.0):
            continue
        p = p_synth_iq(s.x, s.sr, s.f0, s.B)
        if np.isfinite(p):
            ps.append(p)
    return float(np.median(ps)) if ps else float('nan')


def phaser(path: str) -> float:
    """LFO notch-sweep index (phaser/flanger). ~0 real, ≥~4 phasered."""
    x, sr = load_mono(path)
    return _phaser_index(x, sr)


def phaser_penalty(path: str, idx: float | None = None) -> float:
    """Deterministic quality penalty (rating_cal points) for a phaser/flanger artefact."""
    if idx is None:
        idx = phaser(path)
    return PHASER_K * max(0.0, idx - PHASER_THRESH)


def aversiveness(path: str) -> float:
    """Absolute psychoacoustic aversiveness 0-100 (jackhammer/growl/squeak qualities:
    roughness 30-150 Hz, sharpness, non-harmonic noisiness). ~0 clean piano, high =
    unpleasant. See quality/aversiveness.py.  Lower = better (BO must keep this low)."""
    x, sr = load_mono(path)
    return float(_aver_features(x, sr)['aversiveness'])
