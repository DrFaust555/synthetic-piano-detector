"""Parametric inharmonicity + tuning model of Rigaud, David & Daudet (2013).

Used to predict the EXPECTED stretched-tuning (Railsback) curve of a piano from
its measured inharmonicity, so that the global-tuning residual measures genuine
note-to-note inconsistency rather than the natural, physically-mandated stretch.

  * Inharmonicity law (eqs 7-8):  B_ξ(m) = exp(s_B·m + y_B) + exp(s_T·m + y_T)
    with the treble asymptote (s_T, y_T) fixed (standardized string design), so
    only ξ = {s_B, y_B} is fit (L1, eq 29).
  * Octave-type ρ_φ(m) = (κ/2)(1 − erf((m−m0)/α)) + 1  (eq 9), typical φ.
  * Expected deviation curve: octave-chain the A notes from A4 via eq (6), i.e.
    accumulate the per-octave stretch, then interpolate across the compass.
"""

from __future__ import annotations
import numpy as np

try:
    from scipy.special import erf
    from scipy.optimize import minimize
    _HAVE_SCIPY = True
except Exception:                       # pragma: no cover
    _HAVE_SCIPY = False

# Treble-bridge asymptote — Young (1952) / Rigaud: nearly identical for all pianos.
S_T, Y_T = 9.26e-2, -13.64
# A-note MIDI numbers across the compass (A0 … A7), the octave-chaining anchors.
A_MIDIS = np.array([21, 33, 45, 57, 69, 81, 93, 105], dtype=float)


def b_model(m, sB: float, yB: float) -> np.ndarray:
    m = np.asarray(m, dtype=float)
    return np.exp(sB * m + yB) + np.exp(S_T * m + Y_T)


def fit_b_model(midis, b_meas) -> tuple[float, float] | None:
    """Fit ξ = {s_B, y_B} to measured (m, B) by least absolute deviation (eq 29)."""
    if not _HAVE_SCIPY:
        return None
    m = np.asarray(midis, dtype=float)
    b = np.asarray(b_meas, dtype=float)
    ok = (b > 0) & np.isfinite(b)
    m, b = m[ok], b[ok]
    if len(m) < 5:
        return None
    logb = np.log(b)

    def cost(p):
        sB, yB = p
        return float(np.sum(np.abs(np.log(b_model(m, sB, yB) + 1e-30) - logb)))

    try:
        r = minimize(cost, [0.089, -7.0], method='Nelder-Mead',
                     options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4})
        sB, yB = float(r.x[0]), float(r.x[1])
        if not (np.isfinite(sB) and np.isfinite(yB)):
            return None
        return sB, yB
    except Exception:
        return None


def _rho(m, kappa: float = 3.5, m0: float = 60.0, alpha: float = 25.0):
    return (kappa / 2.0) * (1.0 - erf((m - m0) / alpha)) + 1.0


def expected_stretch_curve(midis_query, sB: float, yB: float) -> np.ndarray:
    """Expected deviation-from-ET (cents) at each queried MIDI note.

    The A notes are tuned by octave intervals from A4 (eq 6); each octave adds a
    stretch of 600·log2((1+B(m)·4ρ²)/(1+B(m+12)·ρ²)) cents.  d(A4)=0 (the absolute
    reference offset is handled separately).  The curve is then interpolated to all
    queried notes.
    """
    if not _HAVE_SCIPY:
        return np.zeros_like(np.asarray(midis_query, dtype=float))
    d = {69.0: 0.0}
    for m in (69.0, 81.0, 93.0):                 # chain up
        r = _rho(m)
        bm, bu = b_model(m, sB, yB), b_model(m + 12, sB, yB)
        d[m + 12] = d[m] + 600.0 * np.log2((1 + bm * 4 * r * r) / (1 + bu * r * r))
    for m in (69.0, 57.0, 45.0, 33.0):           # chain down
        r = _rho(m - 12)
        bl, bm = b_model(m - 12, sB, yB), b_model(m, sB, yB)
        d[m - 12] = d[m] - 600.0 * np.log2((1 + bl * 4 * r * r) / (1 + bm * r * r))
    anchors = np.array(sorted(d))
    vals = np.array([d[a] for a in anchors])
    return np.interp(np.asarray(midis_query, dtype=float), anchors, vals)
