"""Per-note physical parameters for 88-key piano (MIDI 21–108)."""

import warnings
import numpy as np
from calib.store import get_calibration


def _clip_warn(name: str, m: int, raw: float, lo: float, hi: float) -> float:
    """Clip `raw` to [lo, hi]; warn if the calibrated value actually hit the
    bound.  calib/calibrator.py searches per-note multipliers over a much
    wider log-space box than these physical runtime clamps, so a fit can
    silently push a parameter past its clamp and waste gradient budget in a
    flat region without anyone noticing (see
    PHYSICAL_PIANO_MODELS_AUDIT_UA.md item #3).  Surfacing it here means it
    shows up during calibration runs too, since fit_anchor renders through
    this same code path.
    """
    v = float(np.clip(raw, lo, hi))
    if raw != v:
        warnings.warn(
            f'{name} clamped at MIDI {m}: requested {raw:.4g}, '
            f'runtime range is [{lo:g}, {hi:g}]', stacklevel=3)
    return v


def midi_to_freq(m: int) -> float:
    return 440.0 * 2.0 ** ((m - 69) / 12.0)


def num_strings(m: int) -> int:
    """Number of strings per note — standard acoustic 1/2/3-string scaling.

    Real pianos use 1 (bass, thick wound), 2 (low bichord), or 3 (trichord)
    strings per note.  A previous 4th "sub-octave" string here was never
    actually tuned an octave down (detune_cents had no n==4 special case)
    and was muted at the output (voice_amp_3=0), so it silently cost CPU
    and diluted the scattering-junction denominator for nothing — see
    PHYSICAL_PIANO_MODELS_AUDIT_UA.md item #9.  If sub-octave/duplex string
    coupling is wanted, it belongs as a separate sympathetic resonator with
    its own length, not a 4th struck string sharing the hammer excitation.
    """
    if m <= 30: return 1   # bass: single thick wound string
    if m <= 41: return 2   # low: bichord
    return 3                # trichord


# Per-string detunes (signed, scaled by 'base' factor per note).
_VOICE_MAIN_BASE = (-0.6, +0.4, +1.5)


def detune_cents(m: int, idx: int, n: int) -> float:
    """Per-string detune in cents.

    - n=1: zero detune (single string)
    - n=2: bichord at ±base
    - n=3: trichord with asymmetric −0.6 / +0.4 / +1.5 base
    """
    if n == 1: return 0.0
    base = max(0.15, 1.5 - 1.2 * (m - 21) / 87.0)
    cal = get_calibration()
    mw = [float(np.clip(cal.get_global(f'voice_main_{i}_scale'), 0.3, 2.0))
          for i in range(3)]
    main = [_VOICE_MAIN_BASE[i] * mw[i] for i in range(3)]
    if n == 2:
        return [main[0], main[2]][idx] * base
    if n == 3:
        return main[idx] * base
    return 0.0


def voice_amplitude(voice_idx: int) -> float:
    """Per-string output gain multiplier."""
    cal = get_calibration()
    return float(cal.get_global(f'voice_amp_{voice_idx}'))


def loss_params(m: int, sr: float):
    """Returns (loop_gain, lp1_coeff, lp2_coeff).  Calibration adjusts T60 and LP cutoff."""
    cal = get_calibration()
    f0 = midi_to_freq(m)
    t = (m - 21) / 87.0
    # Base T60 (seconds): bass ~18s, mid ~8s, treble ~1.5s
    if m <= 40:   t60 = 18 - 8 * (m - 21) / 19
    elif m <= 72: t60 = 10 - 7 * (m - 40) / 32
    else:         t60 = 3 - 1.5 * (m - 72) / 36
    t60 = max(t60, 0.5) * cal.get(m, 't60_scale')
    loops = f0 * t60
    g = 0.001 ** (1.0 / max(loops, 1))
    g = float(np.clip(g, 0.9, 0.99999))

    lp1_raw = np.clip(0.45 - 0.35 * t, 0.05, 0.5) * cal.get(m, 'lp1_scale')
    lp1 = _clip_warn('lp1', m, lp1_raw, 0.0, 0.85)
    lp2 = lp1 * 0.25
    return g, lp1, lp2


def dispersion_coeffs(m: int, sr: float):
    """Allpass coefficients for stiffness dispersion (driven by inharmonicity B)."""
    B = inharmonicity_B(m)
    f0 = midi_to_freq(m)
    n_sec = int(np.clip(2 + 3 * np.log10(max(B, 1e-6) / 1e-4), 2, 6))
    coeffs = []
    for k in range(n_sec):
        n_ref = 2 + k * max(1, 12 // n_sec)
        f_ref = n_ref * f0
        if f_ref > sr * 0.35:
            coeffs.append(0.0)
            continue
        c = np.clip(B * n_ref ** 2 * 0.3, -0.5, 0.5)
        coeffs.append(float(c))
    return coeffs


# Chabassier-Chaigne-Joly (2013) string parameter table for cabinet grand piano.
# Each entry: (f0_Hz, L_m, d_mm, F, T0_N) — diameter is core wire for wound strings.
# Steel constants: E = 2.0e11 Pa, rho = 7850 kg/m^3.
_STRING_TABLE = [
    (38.9,   1.945, 1.48,  5.73, 1781),  # D1
    (65.4,   1.600, 0.95,  3.55, 865),   # C2
    (174.6,  0.961, 1.05,  1.0,  774),   # F3
    (555.6,  0.326, 0.92,  1.0,  684),   # C5
    (1571.0, 0.124, 0.79,  1.0,  587),   # G6
]
_E_STEEL = 2.0e11
_RHO_STEEL = 7850.0


def _interp_string(f0: float):
    """Log-frequency interpolation of (L, d, F, T0) from Chabassier table.

    Returns dict with derived physical quantities.
    """
    lf = np.log10(max(f0, 20.0))
    fts = [np.log10(x[0]) for x in _STRING_TABLE]
    if lf <= fts[0]:
        L, d, F, T0 = _STRING_TABLE[0][1:]
    elif lf >= fts[-1]:
        L, d, F, T0 = _STRING_TABLE[-1][1:]
    else:
        for i in range(len(fts) - 1):
            if fts[i] <= lf <= fts[i + 1]:
                r = (lf - fts[i]) / (fts[i + 1] - fts[i])
                a, b = _STRING_TABLE[i], _STRING_TABLE[i + 1]
                # L, d, F, T0 — log-interpolate L and T0 (vary by orders),
                # linear interpolate d and F (smaller variation)
                L = 10 ** ((1 - r) * np.log10(a[1]) + r * np.log10(b[1]))
                d = (1 - r) * a[2] + r * b[2]
                F = (1 - r) * a[3] + r * b[3]
                T0 = 10 ** ((1 - r) * np.log10(a[4]) + r * np.log10(b[4]))
                break
    d_m = d * 1e-3
    A = np.pi * d_m ** 2 / 4
    I = np.pi * d_m ** 4 / 64
    mu = _RHO_STEEL * A * F  # mass per unit length
    Z = float(np.sqrt(T0 * mu))
    B = float(np.pi ** 2 * _E_STEEL * I / (T0 * L ** 2))
    return {'L': L, 'd_m': d_m, 'F': F, 'T0': T0,
            'mu': mu, 'Z': Z, 'B': B, 'A': A, 'I': I}


def inharmonicity_B(m: int) -> float:
    """Physical inharmonicity from Chabassier string table:
        B = pi^2 * E * I / (T0 * L^2)
    Multiplied by `B_scale` calibration.
    """
    cal = get_calibration()
    f0 = midi_to_freq(m)
    B_raw = _interp_string(f0)['B'] * cal.get(m, 'B_scale')
    return _clip_warn('inharmonicity_B', m, B_raw, 1e-6, 0.1)


def string_impedance(m: int) -> float:
    """Physical impedance Z = sqrt(T0 * mu) from Chabassier table.
    Calibration's bridge_adm_scale is applied INDIRECTLY via bridge_admittance
    (which uses Z to derive the admittance ratio).  Z itself stays physical.
    """
    f0 = midi_to_freq(m)
    return _interp_string(f0)['Z']


# Chabassier-Chaigne-Joly (2013) reference table for cabinet grand piano.
# Anchors: (f0_Hz, K_H N·m⁻ᵖ, p, M_kg, xH/L).
# We interpolate linearly in log(f0) between these.
_HAMMER_TABLE = [
    (38.9,   4.0e8,  2.40, 0.01200, 0.129),  # D1
    (65.4,   2.0e9,  2.27, 0.01020, 0.125),  # C2
    (174.6,  1.0e9,  2.40, 0.00900, 0.120),  # F3
    (555.6,  2.8e10, 2.60, 0.00790, 0.120),  # C5
    (1571.0, 2.3e11, 3.00, 0.00677, 0.121),  # G6
]


def _interp_hammer(f0: float):
    """Log-frequency interpolation of Chabassier table values."""
    lf = np.log10(max(f0, 20.0))
    ft = [np.log10(x[0]) for x in _HAMMER_TABLE]
    if lf <= ft[0]:
        K, p, M, xr = _HAMMER_TABLE[0][1:]
        return K, p, M, xr
    if lf >= ft[-1]:
        K, p, M, xr = _HAMMER_TABLE[-1][1:]
        return K, p, M, xr
    # find segment
    for i in range(len(ft) - 1):
        if ft[i] <= lf <= ft[i + 1]:
            r = (lf - ft[i]) / (ft[i + 1] - ft[i])
            K = 10 ** ((1 - r) * np.log10(_HAMMER_TABLE[i][1]) +
                       r * np.log10(_HAMMER_TABLE[i + 1][1]))
            p = (1 - r) * _HAMMER_TABLE[i][2] + r * _HAMMER_TABLE[i + 1][2]
            M = (1 - r) * _HAMMER_TABLE[i][3] + r * _HAMMER_TABLE[i + 1][3]
            xr = (1 - r) * _HAMMER_TABLE[i][4] + r * _HAMMER_TABLE[i + 1][4]
            return K, p, M, xr
    return _HAMMER_TABLE[-1][1:]


def hammer_params(m: int):
    """Return hammer parameters following the Chabassier 2013 reference table.

    K_H (stiffness), p (Hunt-Crossley exponent), M_H (mass), and strike ratio
    are log-interpolated between table anchors.  Calibration multipliers are
    applied on top.
    """
    cal = get_calibration()
    f0 = midi_to_freq(m)
    K_base, p_base, mass_base, strike_base = _interp_hammer(f0)
    t = (m - 21) / 87.0
    return {
        'mass':         mass_base * cal.get(m, 'hammer_mass_scale'),
        'K':            K_base * cal.get(m, 'hammer_K_scale'),
        'p':            _clip_warn('hammer_p', m, p_base * cal.get(m, 'hammer_p_scale'), 1.5, 4.0),
        'mu':           0.4 * (1 - t) + 0.1 * t,
        'strike_ratio': _clip_warn('strike_ratio', m,
                                    strike_base * cal.get(m, 'strike_pos_scale'), 0.07, 0.18),
        'max_vel':      3.0 + 2.0 * t,
    }


def soundboard_modes():
    """Fixed set of soundboard resonance modes: (freq, Q, base_gain).

    Real piano soundboard has strong body resonances in the 40-200 Hz region
    that give the instrument its "wood" character; without them the model
    sounds like a struck wire (toy piano).  The lowest octave below 100 Hz
    therefore gets the bulk of the gain.
    """
    cal = get_calibration()
    gmul = cal.get_global('soundboard_gain')
    base = [
        # --- low-end body resonances (the missing piece) ---
        (46, 12,  .060), (62, 14,  .055), (78, 16,  .050),
        (92, 18,  .045), (110, 22, .040), (132, 25, .035),
        # --- mid-low ---
        (158, 30, .030), (188, 35, .025), (225, 40, .020),
        (268, 42, .017), (320, 45, .015), (380, 47, .012),
        # --- mid (already prominent in strings, modest support) ---
        (455, 45, .009), (540, 42, .008), (645, 40, .007),
        (770, 35, .006), (920, 30, .005), (1100, 28, .004),
        # --- upper body ---
        (1320, 25, .003), (1580, 22, .003), (1900, 20, .002),
        (2280, 18, .002), (2750, 16, .002), (3300, 14, .001),
        (4000, 12, .001), (4800, 10, .001),
    ]
    out = []
    for i, (f, q, g) in enumerate(base):
        per_mode = cal.get_global(f'sb_mode_{i}')
        out.append((f, q, g * gmul * per_mode))
    return out


def damper_params(m: int):
    if m >= 90: return None  # upper notes undamped
    t = (m - 21) / 69.0
    return {
        'attack_time': 0.05 + 0.03 * (1 - t),
        'max_damp': 0.95 + 0.04 * t,
    }


def bridge_admittance(m: int) -> float:
    cal = get_calibration()
    t = (m - 21) / 87.0
    return float((0.003 + 0.007 * t) * cal.get(m, 'bridge_adm_scale'))


def velocity_gain(m: int) -> float:
    """Per-note amplitude multiplier (applied on render side)."""
    cal = get_calibration()
    return cal.get(m, 'vel_gain') * cal.get_global('output_gain')


def longitudinal_params(m: int, sr: float):
    """
    Nonlinear (Kirchhoff-Carrier) string adds longitudinal modes via quadratic
    coupling.  The longitudinal wavespeed of a piano string (~4000-5000 m/s for
    steel) is much higher than the transverse one (sqrt(T/mu) ~ 100-300 m/s),
    so the longitudinal fundamental sits around 20*f0 for bass and ~8*f0 in
    treble.

    Returns:
        gain  (linear)  — coupling factor
        freq  (Hz)      — first longitudinal mode
        Q               — resonance sharpness
    """
    cal = get_calibration()
    f0 = midi_to_freq(m)
    t = (m - 21) / 87.0
    base_ratio = 22.0 - 14.0 * t
    base_freq = max(80.0, min(sr * 0.45, f0 * base_ratio))
    base_Q = 8.0
    base_gain = 1.5e-4 * (1.0 - 0.7 * t)
    gain = base_gain * cal.get(m, 'long_gain_scale')
    freq = _clip_warn('long_freq', m, base_freq * cal.get(m, 'long_freq_scale'), 80.0, sr * 0.45)
    # Cap Q at 12: above that the longitudinal mode rings long enough to
    # produce audible metallic flare-ups when re-excited during sustain.
    Q = _clip_warn('long_Q', m, base_Q * cal.get(m, 'long_q_scale'), 2.0, 12.0)
    return gain, freq, Q
