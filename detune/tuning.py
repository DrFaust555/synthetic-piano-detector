"""Global tuning consistency.

Measures how consistently the note pitches sit on a *self-consistent* scale.  The
reference pitch is NOT assumed (not fixed 440) — it is the robust consensus the
notes themselves agree on (440, 445, anything).  A piano uniformly shifted to 445
is "in tune with itself" (residuals ≈ 0); only notes that deviate from the
consensus (e.g. a bass string at 58 Hz where ~55 was expected) are penalized.

All confidently-detected notes feed this — including single-string basses, which
have no unison to beat against but whose pitch can still drift from the rest.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass

from detune.rigaud import fit_b_model, expected_stretch_curve

# The Rigaud parametric stretch curve is faithful but needs clean isolated-note
# data; on noisy polyphonic audio it hurts (see note in global_consistency).
USE_RIGAUD_STRETCH = False


@dataclass
class GlobalResult:
    scatter_cents: float      # robust RMS of residuals from the fitted scale
    ref_offset_cents: float   # consensus reference offset vs A=440 (info)
    n_distinct: int           # distinct pitch classes observed
    diversity: float          # 0..1 weight (how much to trust the global metric)
    available: bool
    stretch_model: str = 'none'   # 'rigaud' | 'cubic' | 'linear' | 'none'


def _weighted_median(vals: np.ndarray, w: np.ndarray) -> float:
    order = np.argsort(vals)
    v = vals[order]
    cw = np.cumsum(w[order])
    if cw[-1] <= 0:
        return float(np.median(v))
    return float(v[np.searchsorted(cw, cw[-1] / 2.0)])


def _robust_scatter(res_cents: np.ndarray, w: np.ndarray) -> float:
    """Weighted RMS of residuals with mild outlier clipping."""
    med = np.median(res_cents)
    mad = np.median(np.abs(res_cents - med)) + 1e-9
    clip = 5.0 * 1.4826 * mad
    rc = np.clip(res_cents, med - clip, med + clip)
    return float(np.sqrt(np.sum(w * rc ** 2) / np.sum(w)))


def global_consistency(notes: list[tuple]) -> GlobalResult:
    """notes: list of (f0_hz, weight[, B]).  weight ~ note energy * confidence.

    The expected stretched-tuning curve is predicted with the Rigaud (2013)
    parametric model from the measured inharmonicity (so the natural Railsback
    stretch is NOT counted as detuning); the residual scatter is the genuine
    note-to-note inconsistency.  Falls back to a robust polynomial stretch fit when
    inharmonicity isn't available / the model can't be fit.
    """
    clean = []
    for n in notes:
        f, w = n[0], n[1]
        B = n[2] if len(n) > 2 else None
        if f and f > 20.0 and w > 0:
            clean.append((f, w, B))
    if len(clean) < 2:
        return GlobalResult(0.0, 0.0, len(clean), 0.0, available=False)

    f0 = np.array([c[0] for c in clean])
    w = np.array([c[1] for c in clean])
    midf = 69.0 + 12.0 * np.log2(f0 / 440.0)
    nearest = np.round(midf)
    dev_cents = (midf - nearest) * 100.0           # vs nearest ET semitone

    n_distinct = int(len(np.unique(nearest)))
    if n_distinct < 2:
        return GlobalResult(0.0, 0.0, n_distinct, 0.0, available=False)

    model = 'none'
    d_expect = np.zeros_like(dev_cents)

    # --- Rigaud parametric stretch curve from measured inharmonicity ---
    # NOTE: tried and disabled by default.  The Rigaud (2013) model is validated on
    # clean isolated-note recordings; here the per-note B estimates from polyphonic
    # performance audio are too noisy to fit it, and the octave-chaining amplifies
    # that error into a wildly over-steep bass stretch (≈−90 c predicted vs ≈−5 c
    # measured), INCREASING the residual.  The flexible robust cubic below adapts to
    # whatever smooth stretch (and register-dependent measurement bias) is present
    # and empirically yields much smaller residuals.  Kept for reference / future
    # use on clean isolated-note inputs.
    if USE_RIGAUD_STRETCH and n_distinct >= 10 and any(c[2] is not None for c in clean):
        ms = np.array([nearest[i] for i, c in enumerate(clean) if c[2] is not None])
        Bs = np.array([c[2] for c in clean if c[2] is not None])
        fit = fit_b_model(ms, Bs)
        if fit is not None:
            d_expect = expected_stretch_curve(nearest, fit[0], fit[1])
            model = 'rigaud'

    # --- robust polynomial stretch fit (primary) ---
    if model == 'none':
        octave = (nearest - 60.0) / 12.0
        if n_distinct >= 10:
            A = np.vander(octave, 4)
            sol, *_ = np.linalg.lstsq(A * w[:, None], dev_cents * w, rcond=None)
            fitc = A @ sol
            resid = dev_cents - fitc
            mad = np.median(np.abs(resid - np.median(resid))) + 1e-9
            keep = np.abs(resid - np.median(resid)) <= 3.5 * 1.4826 * mad
            if keep.sum() >= 8:
                sol, *_ = np.linalg.lstsq(A[keep] * w[keep, None],
                                          dev_cents[keep] * w[keep], rcond=None)
                fitc = A @ sol
            d_expect = fitc
            model = 'cubic'
        elif n_distinct >= 5:
            A = np.vstack([octave, np.ones_like(octave)]).T
            sol, *_ = np.linalg.lstsq(A * w[:, None], dev_cents * w, rcond=None)
            d_expect = A @ sol
            model = 'linear'

    # Consensus reference offset (robust): detuned outliers don't drag it, they
    # stand out as residuals instead.  This is the self-consistent reference pitch.
    resid = dev_cents - d_expect
    off = _weighted_median(resid, w)
    res_cents = resid - off

    scatter = _robust_scatter(res_cents, w)
    diversity = float(n_distinct / (n_distinct + 12.0))
    return GlobalResult(scatter_cents=scatter, ref_offset_cents=float(off),
                        n_distinct=n_distinct, diversity=diversity,
                        available=True, stretch_model=model)
