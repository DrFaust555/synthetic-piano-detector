"""Loss / objective functions for tuning the hybrid synth (Otey knobs + neural
polisher) against the quality discriminator — RICHER than the scalar rating.

Why not the XGBoost scalar: it is piecewise-constant (tree output) → a flat, jagged
landscape that Bayesian optimisation can't follow, and it is 1-D.  Instead we use the
45 channel-robust features directly:

  feature_gap(wav)          → signed z-gap per feature vs the real-good cluster
  mahalanobis_to_real(wav)  → smooth scalar distance to the real-good cluster (BO)
  axis_gaps(wav)            → grouped objectives {timbre, decay, tuning, bass, treble}
  objectives(wav)           → dict ready for (multi-objective) BO  (all: lower = better)
  feature_match_loss(...)   → DIFFERENTIABLE torch loss (for the neural polisher)

"Real-good" target cluster = our acoustic files with rating_cal ≥ REAL_MIN (synth sits
at ~0-6 so the threshold excludes it; single-note CCMusic anchors excluded too).

    python -m quality.loss <wav> [<wav>...]   # print gap diagnosis (Otey/Pianoteq…)
"""

from __future__ import annotations
import os
import csv
import json
import sys
import glob
import numpy as np

from quality.pipeline import (note_raw_descriptors, aggregate_file, FEATURE_COLS,
                              TUNING, BASS_FEATURE_COLS)
from quality.effects import phaser_index as _phaser_index
from quality.modulation import modulation_features as _mod_features
from quality.aversiveness import aversiveness_features as _aver_features
from quality.attack import attack_percussive_ratio as _attack_ratio
from detune.segment import load_mono

MOD_KEYS = ['beat_mod', 'beat_flatness', 'fm_wander', 'slow_coherent', 'beat_to_slow']
MOD_REF_CACHE = None
ATTACK_REF_CACHE = None

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FEATURES_CAL = os.path.join(HERE, 'cache', 'features_cal.csv')
TRENDS = os.path.join(HERE, 'cache', 'register_trends_cal.json')
REAL_MIN = 60.0          # rating_cal threshold for the "real-good" target cluster

# Phaser/flanger gate: no acoustic piano has an LFO notch-sweep, so a high
# phaser_index is a deterministic red flag (training data has no phaser positives,
# so the learned model can't penalise it — this gate does).  Reals ≤1, Pianoteq ≤3,
# a phasered render ≈6+.
PHASER_THRESH = 3.5
PHASER_K = 6.0           # penalty (rating_cal points) per unit above threshold

# ---- feature axes (grouped objectives for multi-objective BO) ----
_TIMBRE = ['centroid_h', 'bright_decay', 'spectral_slope', 'even_odd_db',
           'high_partial_frac', 'n_partials_above', 'log_inharmB', 'tristim2',
           'spec_irregularity', 'centroid_var']
_DECAY = ['log_t60', 'attack_time', 'sustain_db_1s', 'hf_decay_excess', 'partial_persist']


def _cols(bases):
    return [f'{b}_{s}' for b in bases for s in ('med', 'iqr') if f'{b}_{s}' in FEATURE_COLS]


AXES = {
    'timbre': _cols(_TIMBRE),
    'decay':  _cols(_DECAY),
    'tuning': list(TUNING),
    'bass':   [c for c in BASS_FEATURE_COLS if c.startswith(('bass_', 'deepbass_'))],
    'treble': [c for c in BASS_FEATURE_COLS if c.startswith('upper_')],
}


class _RealRef:
    """Mean / std / shrunk inverse-covariance of the 45 features over real-good files."""
    def __init__(self):
        rows = list(csv.DictReader(open(FEATURES_CAL, encoding='utf-8')))
        X, keep = [], []
        for r in rows:
            if int(r.get('is_anchor', 0) or 0):
                continue
            try:
                rc = float(r['rating_cal'])
            except ValueError:
                continue
            if rc < REAL_MIN:
                continue
            X.append([float(r[c]) if r.get(c, '') not in ('', 'nan') else np.nan
                      for c in FEATURE_COLS])
            keep.append(r['file'])
        X = np.array(X, dtype=float)
        self.n = len(X)
        self.median = np.nanmedian(X, axis=0)
        X = np.where(np.isnan(X), self.median, X)               # impute
        self.mean = X.mean(0)
        self.std = X.std(0) + 1e-6
        try:
            from sklearn.covariance import LedoitWolf
            self.prec = LedoitWolf().fit(X).precision_
        except Exception:                                       # diagonal fallback
            self.prec = np.diag(1.0 / self.std ** 2)
        self.idx = {c: i for i, c in enumerate(FEATURE_COLS)}


_REF = None
_TRENDS = None


def _ref():
    global _REF
    if _REF is None:
        _REF = _RealRef()
    return _REF


def file_features(path: str) -> np.ndarray:
    """The 45 analytic features for a WAV, using the SAME register trends as training."""
    global _TRENDS
    if _TRENDS is None:
        _TRENDS = json.load(open(TRENDS, encoding='utf-8'))
    meta, notes, tuning = note_raw_descriptors(path)
    row = aggregate_file(notes, tuning, _TRENDS)
    return np.array([row.get(c, np.nan) for c in FEATURE_COLS], dtype=float)


def _impute(x: np.ndarray) -> np.ndarray:
    r = _ref()
    return np.where(np.isnan(x), r.median, x)


def feature_gap(path: str) -> dict:
    """Signed z-gap per feature: (synth − real_mean)/real_std.  + → above real-good."""
    r = _ref()
    x = _impute(file_features(path))
    z = (x - r.mean) / r.std
    return {c: float(z[i]) for i, c in enumerate(FEATURE_COLS)}


def mahalanobis_to_real(path: str, x: np.ndarray | None = None) -> float:
    """Smooth, covariance-aware distance to the real-good cluster (0 = at centre).
    The main single objective for Bayesian optimisation (minimise)."""
    r = _ref()
    if x is None:
        x = file_features(path)
    d = _impute(x) - r.mean
    return float(np.sqrt(max(0.0, d @ r.prec @ d)))


def axis_gaps(path: str, x: np.ndarray | None = None) -> dict:
    """RMS |z| per feature axis — grouped objectives for multi-objective BO (minimise)."""
    r = _ref()
    if x is None:
        x = file_features(path)
    z = (_impute(x) - r.mean) / r.std
    out = {}
    for axis, cols in AXES.items():
        ii = [r.idx[c] for c in cols if c in r.idx]
        out[axis] = float(np.sqrt(np.mean(z[ii] ** 2))) if ii else float('nan')
    return out


def _mod_ref(n_sample: int = 40):
    """Mean/std of the 5 modulation descriptors over a sample of real-good files
    (cached to cache/mod_ref.json).  Used to z-score a candidate's modulation."""
    global MOD_REF_CACHE
    if MOD_REF_CACHE is not None:
        return MOD_REF_CACHE
    cache = os.path.join(HERE, 'cache', 'mod_ref.json')
    if os.path.exists(cache):
        MOD_REF_CACHE = json.load(open(cache, encoding='utf-8'))
        return MOD_REF_CACHE
    # build: real-good acoustic files from features_cal
    files = []
    for r in csv.DictReader(open(FEATURES_CAL, encoding='utf-8')):
        if int(r.get('is_anchor', 0) or 0):
            continue
        try:
            if float(r['rating_cal']) >= REAL_MIN:
                files.append(r['file'])
        except ValueError:
            pass
    files = files[::max(1, len(files) // n_sample)][:n_sample]
    rows = []
    for fn in files:
        p = os.path.join(ROOT, 'audio', fn)
        if not os.path.exists(p):
            continue
        try:
            x, sr = load_mono(p)
            m = _mod_features(x, sr)
            rows.append([m[k] for k in MOD_KEYS])
        except Exception:
            pass
    a = np.array(rows, dtype=float)
    MOD_REF_CACHE = {'mean': a.mean(0).tolist(), 'std': (a.std(0) + 1e-6).tolist()}
    json.dump(MOD_REF_CACHE, open(cache, 'w'))
    return MOD_REF_CACHE


def modulation_gap(path: str) -> dict:
    """Per-descriptor z-gap + summary of a file's modulation vs the real-good cluster.
    High = unreal modulation (e.g. Otey's dead, coherent, low-wander modulation)."""
    ref = _mod_ref()
    x, sr = load_mono(path)
    m = _mod_features(x, sr)
    mean, std = np.array(ref['mean']), np.array(ref['std'])
    z = (np.array([m[k] for k in MOD_KEYS]) - mean) / std
    out = {f'mod_{k}': float(m[k]) for k in MOD_KEYS}
    out['modulation_gap'] = float(np.sqrt(np.mean(z ** 2)))
    return out


def iq_synth(path: str) -> float:
    """File-level P(synth) from the TRAINED I/Q modulation critic (quality/iq_critic):
    median over the file's notes of p_synth_iq.  0 = real-like modulation patterns,
    ~1 = dead synthetic modulation (Otey ≈0.99, real/Pianoteq/samplers ≈0.0).

    Its native domain is ISOLATED notes (exactly what the synthesis optimizer renders),
    and it is constrained to the I/Q modulation domain only — timbre/decay/level/channel
    cannot enter — so lowering it means "the modulation came alive", not "the model
    learned to fool one more critic".  This makes it a GAMING-RESISTANT life target,
    unlike the hand-crafted modulation_gap (which string_fm gamed → sweet-spot
    regression, hence P_MOD=0 by default).  NaN if no scorable notes."""
    from detune.segment import build_segments
    from quality.iq_critic import p_synth_iq
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


def _attack_ref(n_sample: int = 40):
    """Mean/std of attack_percussive_ratio over real-good files (cached to
    cache/attack_ref.json).  Used to z-score a candidate's attack transient."""
    global ATTACK_REF_CACHE
    if ATTACK_REF_CACHE is not None:
        return ATTACK_REF_CACHE
    cache = os.path.join(HERE, 'cache', 'attack_ref.json')
    if os.path.exists(cache):
        ATTACK_REF_CACHE = json.load(open(cache, encoding='utf-8'))
        return ATTACK_REF_CACHE
    files = []
    for r in csv.DictReader(open(FEATURES_CAL, encoding='utf-8')):
        if int(r.get('is_anchor', 0) or 0):
            continue
        try:
            if float(r['rating_cal']) >= REAL_MIN:
                files.append(r['file'])
        except ValueError:
            pass
    files = files[::max(1, len(files) // n_sample)][:n_sample]
    vals = []
    for fn in files:
        p = os.path.join(ROOT, 'audio', fn)
        if not os.path.exists(p):
            continue
        try:
            x, sr = load_mono(p)
            v = _attack_ratio(x, sr)
            if np.isfinite(v):
                vals.append(v)
        except Exception:
            pass
    a = np.array(vals, dtype=float)
    ATTACK_REF_CACHE = {'mean': float(a.mean()), 'std': float(a.std() + 1e-6),
                        'n': int(len(a))}
    json.dump(ATTACK_REF_CACHE, open(cache, 'w'))
    return ATTACK_REF_CACHE


def attack_gap(path: str) -> dict:
    """Attack-transient (hammer knock) percussiveness vs the real-good cluster.
    High gap = onset far from real — the model's dead, tone-only attack (low
    percussive fraction) sits well BELOW real, closed by commuted-synthesis knock."""
    ref = _attack_ref()
    x, sr = load_mono(path)
    r = _attack_ratio(x, sr)
    z = (r - ref['mean']) / ref['std'] if np.isfinite(r) else float('nan')
    return {'attack_percussive_ratio': float(r),
            'attack_gap': float(abs(z)) if np.isfinite(z) else float('nan')}


def phaser(path: str) -> float:
    """LFO notch-sweep index (phaser/flanger). ~0 real, ≥~4 phasered. See effects.py."""
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


def objectives(path: str) -> dict:
    """Everything BO needs in one call (all 'lower = closer to real-good')."""
    x = file_features(path)
    out = {'mahalanobis': mahalanobis_to_real(path, x)}
    out.update(axis_gaps(path, x))
    out['phaser'] = phaser(path)        # extra objective: BO must keep this ~0
    out['modulation'] = modulation_gap(path)['modulation_gap']   # real-like LF "life"
    out['iq'] = iq_synth(path)          # gaming-resistant I/Q life target (0=real, ~1=dead)
    out['aversiveness'] = aversiveness(path)   # absolute psychoacoustic unpleasantness
    out['attack'] = attack_gap(path)['attack_gap']   # ADVISORY: hammer-knock gap (not in J)
    return out


# ----------------------------- differentiable (neural polisher) -----------------
def reference_descriptors(real_path: str, f0: float, n_notes: int = 8) -> dict:
    """Target differentiable descriptors (median over notes of a real reference)."""
    import torch
    from detune.segment import load_mono, build_segments
    from quality.torch_critic import diff_descriptors, DIFF_KEYS
    x, sr = load_mono(real_path)
    segs, _ = build_segments(x, sr)
    segs = sorted(segs, key=lambda s: -s.energy)[:n_notes]
    acc = {k: [] for k in DIFF_KEYS}
    for s in segs:
        if not (24.0 < s.f0 < 4500.0):
            continue
        d = diff_descriptors(torch.tensor(s.x, dtype=torch.float32), sr, s.f0)
        for k in DIFF_KEYS:
            acc[k].append(float(d[k].detach()))
    return {k: float(np.median(v)) for k, v in acc.items() if v}


def feature_match_loss(wave, sr, f0, target: dict, weights: dict | None = None):
    """DIFFERENTIABLE timbre feature-matching loss for the neural polisher.

    `wave` is a torch tensor (requires_grad); matches the channel-robust differentiable
    descriptors (quality/torch_critic) of the polished note to `target` (from
    reference_descriptors on a real-good reference).  Returns a differentiable scalar.
    """
    from quality.torch_critic import diff_descriptors, DIFF_KEYS
    d = diff_descriptors(wave, sr, f0)
    loss = 0.0
    for k in DIFF_KEYS:
        if k not in target:
            continue
        w = (weights or {}).get(k, 1.0)
        loss = loss + w * (d[k] - target[k]).abs()
    return loss


# ----------------------------------- CLI diagnosis ------------------------------
def main():
    args = sys.argv[1:]
    files = []
    for a in args:
        p = a if os.path.isabs(a) else os.path.join(ROOT, a)
        files += sorted(glob.glob(os.path.join(p, '*.wav'))) if os.path.isdir(p) \
            else sorted(glob.glob(p))
    if not files:
        print("Вкажи WAV або теку."); return
    r = _ref()
    print(f"Еталон «реальних 80+»: {r.n} файлів (rating_cal ≥ {REAL_MIN:.0f})\n")
    for f in files:
        try:
            x = file_features(f)
            mah = mahalanobis_to_real(f, x)
            ax = axis_gaps(f, x)
            z = (_impute(x) - r.mean) / r.std
            worst = sorted(range(len(z)), key=lambda i: -abs(z[i]))[:6]
            print(f"=== {os.path.basename(f)} ===")
            print(f"  Mahalanobis={mah:.1f}   осі(RMS|z|): " +
                  "  ".join(f"{k}={v:.2f}" for k, v in ax.items()))
            print(f"  phaser={phaser(f):.1f}   противність={aversiveness(f):.0f}/100")
            print("  найбільші розриви (z, + = вище за реал): " +
                  ", ".join(f"{FEATURE_COLS[i]} {z[i]:+.1f}" for i in worst))
        except Exception as e:
            print(f"=== {os.path.basename(f)}: ERR {e} ===")


if __name__ == '__main__':
    main()
