"""Unison detuning estimators — how far apart the 2-3 strings of one note are.

Two complementary methods (see plan):

  U2  beat-rate via partial-envelope modulation, with a k-scaling consistency fit.
      Robust, works on short notes.  PRIMARY.
      A true unison split makes partial k beat at ~k*Δf (the split widens with k),
      so fitting beat_rate(k) ≈ k*Δf both estimates Δf and rejects stray amplitude
      modulation (mic movement, room) that does NOT scale with k.

  U1  high-resolution sinusoidal split fit on sustained notes.  REFINEMENT.
      Resolves the individual string lines per partial → recovers the *cluster*
      structure (e.g. "2 strings together + 1 off" vs "all 3 spread").
"""

from __future__ import annotations
import numpy as np
from numpy.fft import rfft, rfftfreq
from dataclasses import dataclass

from scipy.signal import butter, sosfiltfilt, hilbert, find_peaks


# 1200 / ln(2): small-interval Hz→cents conversion (cents ≈ K * Δf / f0).
_CENTS_K = 1200.0 / np.log(2.0)


@dataclass
class UnisonResult:
    spread_cents: float        # estimated max string spread (cents)
    beat_hz: float             # dominant beat rate at the fundamental (Hz)
    confidence: float          # 0..1
    method: str                # 'U1' | 'U2' | 'none'
    n_partials: int            # partials that contributed
    config: str = ''           # cluster description (U1 only)


def _partial_centers(f0: float, B: float, sr: int, n: int) -> list[tuple[int, float]]:
    out = []
    nyq = sr * 0.45
    for k in range(1, n + 1):
        fk = k * f0 * np.sqrt(1.0 + B * k * k)
        if fk > nyq:
            break
        out.append((k, fk))
    return out


def _bandpass(x: np.ndarray, sr: int, f_lo: float, f_hi: float, order: int = 4):
    nyq = sr / 2.0
    lo = max(f_lo, 1.0) / nyq
    hi = min(f_hi, nyq * 0.999) / nyq
    if hi <= lo:
        return None
    sos = butter(order, [lo, hi], btype='band', output='sos')
    padlen = 3 * (sos.shape[0] * 2)
    if len(x) <= padlen + 2:
        return None
    return sosfiltfilt(sos, x)


def _modulation(env: np.ndarray, sr: int, f_min: float = 0.7, f_max: float = 45.0
                ) -> tuple[float, float, float]:
    """Dominant modulation frequency of an envelope.

    Returns (beat_hz, strength, depth).  The slow exponential decay is removed with
    a low-order polynomial detrend before taking the modulation spectrum.
    """
    n = len(env)
    if n < 64:
        return 0.0, 0.0, 0.0
    t = np.arange(n) / sr
    mean_env = float(np.mean(env)) + 1e-12
    deg = 4 if n > 8 * (sr // 100) else 2
    try:
        coef = np.polyfit(t, env, deg)
        trend = np.polyval(coef, t)
    except Exception:
        trend = np.full(n, mean_env)
    res = env - trend

    w = np.hanning(n)
    M = np.abs(rfft(res * w))
    mf = rfftfreq(n, 1.0 / sr)
    band = (mf >= f_min) & (mf <= f_max)
    if not band.any() or M[band].max() <= 0:
        return 0.0, 0.0, 0.0
    idx = np.where(band)[0]
    j = idx[int(np.argmax(M[band]))]
    # parabolic refine of the modulation peak
    if 1 <= j < len(M) - 1:
        a, b, c = M[j - 1], M[j], M[j + 1]
        den = a - 2 * b + c
        off = 0.5 * (a - c) / den if abs(den) > 1e-12 else 0.0
    else:
        off = 0.0
    beat = float((j + off) * (mf[1] - mf[0]))
    # strength: peak vs in-band median (how clearly a beat stands out)
    med = float(np.median(M[band])) + 1e-12
    strength = float(M[j] / med)
    depth = float(np.std(res) / mean_env)
    return beat, strength, depth


def _weighted_median(vals: np.ndarray, w: np.ndarray) -> float:
    if len(vals) == 0:
        return 0.0
    order = np.argsort(vals)
    v = vals[order]
    cw = np.cumsum(w[order])
    if cw[-1] <= 0:
        return float(np.median(v))
    cutoff = cw[-1] / 2.0
    return float(v[np.searchsorted(cw, cutoff)])


def estimate_u2(seg_x: np.ndarray, sr: int, f0: float, B: float,
                n_partials: int = 10) -> UnisonResult:
    """Beat-rate / k-scaling unison estimate (primary).

    Distinguishes three outcomes:
      * too weak / short to analyze            → method 'none' (unknown)
      * partials clearly present, no confirmed beat → spread 0 (well tuned)
      * beat confirmed across ≥2 partials with k-scaling → measured spread
    The k-scaling confirmation (a real unison beat shows in several partials at
    rate ~k*Δf) rejects single-partial blips and non-scaling AM (mic/room).
    """
    centers = _partial_centers(f0, B, sr, n_partials)
    if not centers:
        return UnisonResult(0.0, 0.0, 0.0, 'none', 0)

    rows = []
    max_e = 0.0
    for k, fk in centers:
        # band half-width: wide enough for the k*Δf split, narrow enough to avoid
        # the neighbouring partials (spaced ~f0 apart).
        hw = float(np.clip(0.03 * fk, 6.0, 0.40 * f0))
        bp = _bandpass(seg_x, sr, fk - hw, fk + hw)
        if bp is None:
            continue
        env = np.abs(hilbert(bp))
        p_energy = float(np.mean(env ** 2))
        rows.append((k, env, p_energy))
        max_e = max(max_e, p_energy)

    if max_e <= 0:
        return UnisonResult(0.0, 0.0, 0.0, 'none', 0)

    df_per_k = []     # Δf candidates  = beat_k / k
    weights = []
    n_analyzable = 0
    for k, env, p_energy in rows:
        if p_energy < 0.03 * max_e:
            continue              # partial too weak to trust
        n_analyzable += 1
        beat, strength, depth = _modulation(env, sr)
        if beat <= 0 or strength < 3.0 or depth < 0.03:
            continue
        df_per_k.append(beat / k)
        weights.append(p_energy * min(strength, 20.0) * min(depth, 1.0))

    if n_analyzable < 2:
        # couldn't reliably analyze → unknown, let caller drop it
        return UnisonResult(0.0, 0.0, 0.0, 'none', n_analyzable)

    tuned_conf = float(np.clip(0.40 + 0.06 * n_analyzable, 0.0, 0.9))
    if len(df_per_k) == 0:
        # partials present but no beating → well tuned unison
        return UnisonResult(0.0, 0.0, tuned_conf, 'U2', n_analyzable)

    df_per_k = np.array(df_per_k)
    weights = np.array(weights)
    # Robust Δf: weighted median of per-partial estimates rejects partials whose
    # beat doesn't follow the k-scaling (i.e. non-unison AM).
    delta_f = _weighted_median(df_per_k, weights)
    consistent = np.abs(df_per_k - delta_f) <= max(0.25, 0.15 * delta_f)
    n_agree = int(np.sum(consistent))

    # Require the beat to be confirmed across ≥3 partials following k-scaling.  A
    # true unison split shows in many partials at rate ~k*Δf; sporadic beats from
    # partial collisions with OTHER sounding notes (polyphony, pedal, sympathetic
    # resonance) do not, so they fail this and read as tuned.
    if n_agree < 3:
        return UnisonResult(0.0, 0.0, tuned_conf, 'U2', n_analyzable)

    frac = n_agree / len(df_per_k)
    conf = float(np.clip(0.35 + 0.55 * frac * (1.0 - np.exp(-n_agree / 3.0)), 0.0, 1.0))
    spread_cents = _CENTS_K * delta_f / f0
    return UnisonResult(spread_cents=float(spread_cents), beat_hz=float(delta_f),
                        confidence=conf, method='U2', n_partials=n_agree)


def estimate_u1(seg_x: np.ndarray, sr: int, f0: float, B: float,
                n_partials: int = 6) -> UnisonResult | None:
    """High-resolution sinusoidal split fit (refinement; needs a long ring).

    Resolves up to 3 string lines per partial when the window is long enough that
    1/T is smaller than the split.  Recovers the per-note string cluster.
    """
    T = len(seg_x) / sr
    if T < 0.6:
        return None
    res_hz = 1.0 / T  # spectral resolution; we can separate peaks > ~1.5*res_hz

    nfft = 1 << int(np.ceil(np.log2(len(seg_x) * 4)))
    nfft = min(nfft, 1 << 20)
    win = np.hanning(len(seg_x))
    mag = np.abs(rfft(seg_x * win, nfft))
    if mag.max() <= 0:
        return None
    bin_hz = sr / nfft

    centers = _partial_centers(f0, B, sr, n_partials)
    string_freqs: list[float] = []   # per-partial string freq / k  (≈ string f0)
    string_amps: list[float] = []
    contributing = 0
    for k, fk in centers:
        hw = float(np.clip(0.03 * fk, 5.0, 0.40 * f0))
        lo = int((fk - hw) / bin_hz)
        hi = int((fk + hw) / bin_hz) + 1
        if hi - lo < 5 or lo < 1 or hi >= len(mag):
            continue
        seg = mag[lo:hi]
        min_sep = max(2, int(1.5 * res_hz / bin_hz))
        pk, props = find_peaks(seg, distance=min_sep,
                               height=seg.max() * 0.12, prominence=seg.max() * 0.08)
        if len(pk) == 0:
            continue
        # keep the strongest up to 3 lines
        heights = props['peak_heights']
        order = np.argsort(heights)[::-1][:3]
        found = 0
        for pi in pk[order]:
            loc = lo + pi
            if 1 <= loc < len(mag) - 1:
                a, b, c = mag[loc - 1], mag[loc], mag[loc + 1]
                den = a - 2 * b + c
                off = 0.5 * (a - c) / den if abs(den) > 1e-12 else 0.0
            else:
                off = 0.0
            f_line = (loc + off) * bin_hz
            string_freqs.append(f_line / k)
            string_amps.append(float(mag[loc]))
            found += 1
        if found >= 2:
            contributing += 1

    # Need at least two partials that actually resolved >=2 lines to trust U1.
    if contributing < 2 or len(string_freqs) < 4:
        return None

    sf_arr = np.array(string_freqs)
    sa_arr = np.array(string_amps)
    # Cluster the per-partial string-f0 estimates into distinct strings.
    clusters = _cluster_freqs(sf_arr, sa_arr, tol_cents=4.0)
    # Drop weak clusters (leakage from neighbouring notes / noise) — keep only
    # lines with real amplitude relative to the strongest.
    if clusters:
        amax = max(c[1] for c in clusters)
        clusters = [c for c in clusters if c[1] >= 0.20 * amax]
    if len(clusters) < 2:
        return None
    cfreqs = np.array([c[0] for c in clusters])
    camps = np.array([c[1] for c in clusters])
    spread_cents = 1200.0 * np.log2(cfreqs.max() / cfreqs.min())
    beat_hz = float(cfreqs.max() - cfreqs.min())

    config = _describe_config(cfreqs, camps)
    conf = float(np.clip(0.5 + 0.15 * contributing, 0.0, 1.0))
    return UnisonResult(spread_cents=float(spread_cents), beat_hz=beat_hz,
                        confidence=conf, method='U1', n_partials=contributing,
                        config=config)


def _cluster_freqs(freqs: np.ndarray, amps: np.ndarray, tol_cents: float = 4.0):
    """Greedy 1-D clustering of string-f0 estimates within tol_cents.

    Returns list of (center_freq, total_amp, count) sorted by frequency.
    """
    order = np.argsort(freqs)
    f = freqs[order]
    a = amps[order]
    clusters = []
    cur_f = [f[0]]
    cur_a = [a[0]]
    for i in range(1, len(f)):
        cents = 1200.0 * np.log2(f[i] / np.mean(cur_f))
        if cents <= tol_cents:
            cur_f.append(f[i])
            cur_a.append(a[i])
        else:
            clusters.append((float(np.average(cur_f, weights=cur_a)),
                             float(np.sum(cur_a)), len(cur_f)))
            cur_f = [f[i]]
            cur_a = [a[i]]
    clusters.append((float(np.average(cur_f, weights=cur_a)),
                     float(np.sum(cur_a)), len(cur_f)))
    return clusters


def _describe_config(cfreqs: np.ndarray, camps: np.ndarray) -> str:
    """Human-readable string cluster (answers '1 off / 2 together' vs spread)."""
    n = len(cfreqs)
    if n <= 1:
        return 'unison'
    amax = camps.max()
    rel = camps / amax
    parts = []
    for f, r in zip(cfreqs, rel):
        parts.append(f"{f:.2f}Hz({r:.2f})")
    return f"{n} lines: " + ", ".join(parts)


def estimate_unison_spectral(seg_x: np.ndarray, sr: int, f0: float, B: float,
                             n_partials: int = 6, skip_attack_s: float = 0.12,
                             band_cents: float = 55.0, min_offset_cents: float = 3.5
                             ) -> UnisonResult:
    """Spectral split estimator (primary, robust to polyphony).

    On the sustain (attack skipped) with a high-resolution FFT, each partial band
    is examined for a SECONDARY peak — the spectral line of a string that has
    drifted off the unison.  Key robustness facts:
      * Resolution (≈1/T) sets a natural floor: an intentional <~3-5 cent unison
        ("warmth") is not resolved → reads as tuned.  Real detuning is.
      * The secondary's offset in cents is k-INDEPENDENT (both lines scale with k),
        so a true split shows the SAME offset on every partial; collisions with
        other sounding notes give scattered offsets → rejected by consistency.

    Returns spread_cents = string offset (cents), beat_hz = beat at the fundamental,
    and `config` carries the secondary's relative amplitude (the energy split, which
    distinguishes "1 string off" from "2 strings off").
    """
    a = min(int(skip_attack_s * sr), max(0, len(seg_x) - int(0.2 * sr)))
    xs = seg_x[a:]
    if len(xs) < int(0.3 * sr):
        xs = seg_x
    T = len(xs) / sr
    if T < 0.22:
        return UnisonResult(0.0, 0.0, 0.0, 'none', 0)

    nfft = 1 << int(np.ceil(np.log2(len(xs) * 4)))
    nfft = min(nfft, 1 << 20)
    mag = np.abs(rfft(xs * np.hanning(len(xs)), nfft))
    if mag.max() <= 0:
        return UnisonResult(0.0, 0.0, 0.0, 'none', 0)
    bin_hz = sr / nfft
    min_sep = max(2, int(1.5 / T / bin_hz))

    centers = _partial_centers(f0, B, sr, n_partials)
    rows = []          # (offset_cents_signed, rel_amp, energy)
    n_analyzable = 0
    band_energies = []
    for k, fk in centers:
        hw = fk * (2 ** (band_cents / 1200.0) - 1.0)
        lo = int((fk - hw) / bin_hz)
        hi = int((fk + hw) / bin_hz) + 1
        if hi - lo < 5 or lo < 1 or hi >= len(mag):
            continue
        band = mag[lo:hi].astype(float)
        energy = float(np.sum(band ** 2))
        band_energies.append(energy)
    if not band_energies:
        return UnisonResult(0.0, 0.0, 0.0, 'none', 0)
    e_floor = 0.03 * max(band_energies)

    for k, fk in centers:
        hw = fk * (2 ** (band_cents / 1200.0) - 1.0)
        lo = int((fk - hw) / bin_hz)
        hi = int((fk + hw) / bin_hz) + 1
        if hi - lo < 5 or lo < 1 or hi >= len(mag):
            continue
        band = mag[lo:hi].astype(float)
        energy = float(np.sum(band ** 2))
        if energy < e_floor:
            continue
        n_analyzable += 1
        pk, props = find_peaks(band, distance=min_sep,
                               height=band.max() * 0.12, prominence=band.max() * 0.08)
        if len(pk) < 2:
            continue
        heights = props['peak_heights']
        order = np.argsort(heights)[::-1]
        main_i = pk[order[0]]
        f_main = _parabolic_freq(mag, lo + main_i, bin_hz)
        h_main = heights[order[0]]
        # strongest secondary at least min_offset_cents away from the main line
        for oi in order[1:]:
            f_sec = _parabolic_freq(mag, lo + pk[oi], bin_hz)
            off = 1200.0 * np.log2(f_sec / f_main)
            if abs(off) >= min_offset_cents:
                rows.append((off, heights[oi] / h_main, energy))
                break

    if n_analyzable < 2:
        return UnisonResult(0.0, 0.0, 0.0, 'none', n_analyzable)

    tuned_conf = float(np.clip(0.40 + 0.06 * n_analyzable, 0.0, 0.9))
    if len(rows) < 2:
        # no consistent secondary line → unison is tight (well tuned)
        return UnisonResult(0.0, 0.0, tuned_conf, 'SP', n_analyzable)

    offs = np.array([abs(r[0]) for r in rows])
    rels = np.array([r[1] for r in rows])
    ens = np.array([r[2] for r in rows])
    w = ens * np.clip(rels, 0, 2)
    o_med = _weighted_median(offs, w)
    # Consistency: the split offset is k-independent, so genuine partials cluster.
    keep = np.abs(offs - o_med) <= max(3.0, 0.35 * o_med)
    if int(np.sum(keep)) < 2:
        return UnisonResult(0.0, 0.0, tuned_conf, 'SP', n_analyzable)

    detune_cents = float(o_med)
    rel_amp = float(_weighted_median(rels[keep], ens[keep]))
    beat_hz = float(f0 * (2 ** (detune_cents / 1200.0) - 1.0))
    n_keep = int(np.sum(keep))
    conf = float(np.clip(0.45 + 0.5 * (1.0 - np.exp(-n_keep / 3.0)), 0.0, 1.0))
    config = f"secondary line {rel_amp:.2f}× main @ {detune_cents:.1f}c"
    return UnisonResult(spread_cents=detune_cents, beat_hz=beat_hz, confidence=conf,
                        method='SP', n_partials=n_keep, config=config)


def _parabolic_freq(mag: np.ndarray, loc: int, bin_hz: float) -> float:
    if 1 <= loc < len(mag) - 1:
        a, b, c = mag[loc - 1], mag[loc], mag[loc + 1]
        den = a - 2 * b + c
        off = 0.5 * (a - c) / den if abs(den) > 1e-12 else 0.0
        return (loc + off) * bin_hz
    return loc * bin_hz


def estimate_unison(seg_x: np.ndarray, sr: int, f0: float, B: float,
                    n_partials: int = 10, allow_u1: bool = True) -> UnisonResult:
    """Unison estimate.  The spectral split method (robust to polyphony, proven to
    separate the same piano tuned vs detuned) is primary.  On isolated inputs
    (single note / triad) U1 additionally recovers the full string cluster, so we
    prefer it there when it resolves ≥2 lines confidently.
    """
    if allow_u1:
        u1 = estimate_u1(seg_x, sr, f0, B, n_partials=min(n_partials, 6))
        if u1 is not None and u1.confidence >= 0.6:
            return u1
    return estimate_unison_spectral(seg_x, sr, f0, B, n_partials=min(n_partials, 6))
