"""Tuning axis: recording -> deviation from the instrument's own target tuning -> 0-100 (Sec. 3.8).

  audio -(detector.segment.build_segments)-> per-note {midi, f0, B, energy}
        -(build_b_curve)-> 88-key inharmonicity curve (energy-weighted median B, physical reference elsewhere)
        -(EntropySolver)-> target tuning: the 88 per-key offsets minimizing the entropy of the summed partial spectrum
        -(deviation from the target, one common offset removed)-> score = 100 exp(-d / 12 cents)
d is the energy-weighted median absolute deviation in cents; the common offset (fitted A4) is reported separately,
so a piano tuned to A4 = 442 Hz is not penalized.  A professionally tuned grand on performance audio (d ~ 1.4 cents,
including ~1.5 cents of partial-measurement noise) scores about 90.  target='railsback' uses the fast reference
stretch curve instead of the entropy solve.
"""
from __future__ import annotations
import math

import numpy as np

from .noteref import (
    midi_to_freq, reference_B, build_reference_stretch_cents,
    K_FIRST_PIANO_MIDI, K_NUM_PIANO_KEYS,
)
from .entropy import EntropySolver

# Score mapping: score = 100·exp(-median_residual_cents / TAU).  Calibrated
# (2026-07) so a professionally-tuned grand on polyphonic performance audio
# (~1.4¢ residual, incl. ~1.5¢ partial-measurement noise) reads ~90; entropy
# target.  Provisional — final human-facing scaling happens in combiner D.
# Isolated-note material scores higher.
_SCORE_TAU_CENTS = 12.0
_MIN_NOTES = 4


def _weighted_median(vals: np.ndarray, wts: np.ndarray) -> float:
    if vals.size == 0:
        return float("nan")
    order = np.argsort(vals)
    v, w = vals[order], wts[order]
    cw = np.cumsum(w)
    if cw[-1] <= 0:
        return float(np.median(vals))
    return float(v[np.searchsorted(cw, cw[-1] / 2.0)])


def build_b_curve(notes: list[dict]) -> np.ndarray:
    """88-key inharmonicity curve: energy-weighted median measured B where notes
    exist, physical reference_B elsewhere (index 0 = A0)."""
    b = np.array([reference_B(midi_to_freq(K_FIRST_PIANO_MIDI + k))
                  for k in range(K_NUM_PIANO_KEYS)], dtype=np.float64)
    by_key: dict[int, list[tuple[float, float]]] = {}
    for nt in notes:
        key = int(nt["midi"]) - K_FIRST_PIANO_MIDI
        if 0 <= key < K_NUM_PIANO_KEYS and nt["B"] > 0:
            by_key.setdefault(key, []).append((float(nt["B"]), float(nt["energy"])))
    for key, items in by_key.items():
        vals = np.array([v for v, _ in items])
        wts = np.array([w for _, w in items])
        b[key] = _weighted_median(vals, wts)
    return b


def ideal_stretch_cents(b_curve: np.ndarray, a4_hz: float = 440.0,
                        target: str = "entropy") -> np.ndarray:
    """Ideal per-key stretch (cents from 12-TET, A4=0), index 0 = A0.
    target='entropy' → instrument-specific (Hinrichsen solve on b_curve);
    target='railsback' → fast physical reference curve."""
    if target == "railsback":
        ref = build_reference_stretch_cents()
        return np.array([ref[K_FIRST_PIANO_MIDI + k]
                         for k in range(K_NUM_PIANO_KEYS)], dtype=np.float64)
    if target == "entropy":
        return EntropySolver().solve(b_curve, a4_hz)
    raise ValueError(f"unknown target {target!r}")


def score_from_notes(notes: list[dict], target: str = "entropy") -> dict:
    """Tuning score from per-note measurements.  Each note: {midi, f0, B, energy}.

    Returns dict with: score (0-100 or nan), a4_est (fitted pitch standard, Hz),
    median_dev_cents (internal residual), n_notes, target.
    """
    pts = [nt for nt in notes
           if K_FIRST_PIANO_MIDI <= int(nt["midi"]) <= K_FIRST_PIANO_MIDI + K_NUM_PIANO_KEYS - 1
           and nt["f0"] > 0]
    if len(pts) < _MIN_NOTES:
        return {"score": float("nan"), "a4_est": float("nan"),
                "median_dev_cents": float("nan"), "n_notes": len(pts),
                "target": target}

    b_curve = build_b_curve(pts)
    stretch = ideal_stretch_cents(b_curve, 440.0, target)

    raw, wts = [], []
    for nt in pts:
        midi = int(nt["midi"])
        key = midi - K_FIRST_PIANO_MIDI
        meas_cents = 1200.0 * math.log2(nt["f0"] / midi_to_freq(midi, 440.0))
        raw.append(meas_cents - stretch[key])
        wts.append(float(nt["energy"]))
    raw = np.asarray(raw)
    wts = np.asarray(wts)

    # Global pitch standard fit out (energy-weighted median of the raw offset).
    g = _weighted_median(raw, wts)
    a4_est = 440.0 * (2.0 ** (g / 1200.0))
    resid = np.abs(raw - g)
    d = _weighted_median(resid, wts)

    score = 100.0 * math.exp(-d / _SCORE_TAU_CENTS)
    return {"score": float(score), "a4_est": float(a4_est),
            "median_dev_cents": float(d), "n_notes": len(pts), "target": target}


def measure_notes(wav_path: str, max_notes: int = 200) -> list[dict]:
    """Segment a recording into per-note {midi, f0, B, energy} via detune."""
    from detector.segment import load_mono, build_segments
    x, sr = load_mono(wav_path)
    segs, _ = build_segments(x, sr, max_notes=max_notes)
    return [{"midi": s.midi, "f0": s.f0, "B": s.B, "energy": s.energy}
            for s in segs]


def tuning_score(wav_path: str, target: str = "entropy",
                 max_notes: int = 200) -> dict:
    """End-to-end tuning score for a recording (0-100 + diagnostics)."""
    notes = measure_notes(wav_path, max_notes=max_notes)
    out = score_from_notes(notes, target=target)
    out["path"] = wav_path
    return out


def _main(argv: list[str]) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Tuning axis A: recording → 0-100.")
    ap.add_argument("wav", nargs="+", help="WAV file(s) to score")
    ap.add_argument("--target", choices=("entropy", "railsback"), default="entropy",
                    help="ideal-stretch source (entropy = instrument-specific, default; "
                         "railsback = fast reference)")
    a = ap.parse_args(argv)
    print(f"{'file':40s} {'score':>6s} {'a4':>7s} {'dev¢':>6s} {'n':>4s}")
    for w in a.wav:
        try:
            r = tuning_score(w, target=a.target)
            import os
            print(f"{os.path.basename(w):40s} {r['score']:6.1f} {r['a4_est']:7.2f} "
                  f"{r['median_dev_cents']:6.2f} {r['n_notes']:4d}")
        except Exception as e:  # noqa: BLE001
            print(f"{w}: ERR {type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(_main(sys.argv[1:]))
