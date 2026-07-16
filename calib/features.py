"""
Timbral feature extraction for piano sample / model output comparison.

Features captured (all designed to be scale-invariant where appropriate):

  amp_env      — RMS envelope in dB on log-spaced time points
  spec_frames  — log-magnitude mel-like spectrum at several times
  partials     — first 12 harmonic amplitudes (dB) over time
  attack_time  — seconds from onset to peak RMS
  centroid_t   — spectral centroid trajectory (Hz)
  t60          — estimated decay time to -60 dB (seconds)
  inharm_B     — estimated inharmonicity coefficient from partial frequencies
"""

from __future__ import annotations
import numpy as np
from numpy.fft import rfft

# ---------------------------------------------------------------------------
# Time anchors (seconds after onset) at which we sample spectral frames.
# Log-spaced so attack + early decay get more weight than long tail.
TIME_ANCHORS = np.array([0.02, 0.05, 0.10, 0.20, 0.40, 0.80, 1.60])

# Boolean mask selecting the onset-only anchors (<=80ms) — used when fitting
# hammer parameters in isolation so the loss can't be satisfied by cheating
# on the sustain/tail instead of the actual attack transient.
ONSET_MASK = TIME_ANCHORS <= 0.08

# Mel-like log frequency band edges (Hz) for spectral envelope comparison.
# 24 bands from 60 Hz to 16 kHz.
def _band_edges(n=24, fmin=60.0, fmax=16000.0):
    return np.geomspace(fmin, fmax, n + 1)

BAND_EDGES = _band_edges()

# ---------------------------------------------------------------------------
def _trim_onset(x: np.ndarray, sr: int, thresh_db: float = -50.0) -> tuple[np.ndarray, int]:
    """Cut samples before audible onset.  Returns (trimmed, onset_idx_in_orig)."""
    win = max(1, sr // 1000)
    pad = (-len(x)) % win
    if pad:
        x = np.concatenate([x, np.zeros(pad)])
    frames = x.reshape(-1, win)
    rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-30)
    peak = rms.max()
    thr = peak * (10.0 ** (thresh_db / 20.0))
    above = np.where(rms > thr)[0]
    onset_frame = int(above[0]) if len(above) else 0
    onset_sample = onset_frame * win
    return x[onset_sample:], onset_sample


def _stft_mag(x: np.ndarray, sr: int, center: int, nfft: int) -> tuple[np.ndarray, np.ndarray]:
    """Magnitude spectrum centred at sample 'center' using a Hann window."""
    half = nfft // 2
    a = max(0, center - half)
    b = min(len(x), center + half)
    seg = np.zeros(nfft)
    seg[: b - a] = x[a:b]
    win = np.hanning(nfft)
    seg = seg * win
    mag = np.abs(rfft(seg))
    freqs = np.fft.rfftfreq(nfft, 1.0 / sr)
    return freqs, mag


def _mel_band_db(freqs: np.ndarray, mag: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Integrate magnitude² into log-spaced bands → dB."""
    bands = np.zeros(len(edges) - 1)
    p = mag * mag
    for i in range(len(bands)):
        lo, hi = edges[i], edges[i + 1]
        mask = (freqs >= lo) & (freqs < hi)
        if mask.any():
            bands[i] = p[mask].sum()
    bands = np.maximum(bands, 1e-12)
    return 10.0 * np.log10(bands)


def _spectral_centroid(freqs: np.ndarray, mag: np.ndarray) -> float:
    w = mag
    if w.sum() < 1e-12:
        return 0.0
    return float((freqs * w).sum() / w.sum())


def _amplitude_envelope_db(x: np.ndarray, sr: int, n_pts: int = 64) -> np.ndarray:
    """RMS envelope (dB rel. peak) at log-spaced time points up to 4 seconds."""
    times = np.geomspace(0.005, min(4.0, len(x) / sr - 0.005), n_pts)
    win = max(1, sr // 200)  # 5 ms window
    out = np.zeros(n_pts)
    for i, t in enumerate(times):
        c = int(t * sr)
        a = max(0, c - win // 2)
        b = min(len(x), c + win // 2)
        if b - a > 1:
            out[i] = np.sqrt(np.mean(x[a:b] ** 2))
        else:
            out[i] = 1e-9
    peak = out.max() + 1e-12
    return 20.0 * np.log10(out / peak + 1e-12), times


def _estimate_t60(env_db: np.ndarray, times: np.ndarray) -> float:
    """Fit -60 dB time from amplitude envelope after the peak."""
    if len(env_db) < 5:
        return 0.5
    peak_idx = int(np.argmax(env_db))
    if peak_idx >= len(env_db) - 3:
        return 0.5
    t = times[peak_idx:]
    e = env_db[peak_idx:]
    # Fit linear slope (dB per second).  Robust to noise floor: only use 0..-40 dB
    e_rel = e - e[0]
    mask = e_rel > -40.0
    if mask.sum() < 3:
        return 0.5
    slope, intercept = np.polyfit(t[mask], e_rel[mask], 1)
    if slope >= -0.5:  # essentially flat
        return 10.0
    return float((-60.0 - intercept) / slope)


def _estimate_attack_v2(x: np.ndarray, sr: int,
                         onset_thresholds_db=(-35.0, -40.0, -50.0),
                         early_window_ms: float = 30.0) -> dict:
    """Robust early-attack descriptors, replacing the single scalar
    `attack_time` (= time-to-GLOBAL-peak of a log-spaced envelope).  That
    metric was found to be hijackable by a later room/body resonance or by
    how low the onset threshold is set — the C3 "78ms body bloom"
    investigation turned out to be filter-settling artifact, not physical,
    but the global-argmax `attack_time` metric is exactly the kind of thing
    that WOULD get fooled by a genuine late resonance if one existed. See
    PHYSICAL_PIANO_MODELS_AUDIT_UA.md follow-up discussion.

    Design:
      - onset time is the median across several thresholds (-35/-40/-50dB);
        `onset_threshold_spread_ms` flags disagreement between them instead
        of silently picking one.
      - early_peak_ms / rise_10_90_ms are found ONLY within
        `early_window_ms` of onset, so a later resonance can't be picked up
        as "the attack" the way a global argmax could.
      - crest_20ms and energy_0_10/energy_10_30 are scale-free (ratios),
        comparable across differently-normalised recordings.

    x should be minimally trimmed (coarse leading-silence removal only —
    NOT the aggressive -50dB trim used elsewhere in this module) so the
    onset thresholds below still have something meaningful to disagree on.
    """
    win = max(1, int(sr * 0.0015))   # ~1.5ms envelope window
    hop = max(1, int(sr * 0.001))    # 1ms hop
    n = (len(x) - win) // hop + 1
    empty = {
        'onset_ms': 0.0, 'onset_threshold_spread_ms': float('nan'),
        'early_peak_ms': float('nan'), 'rise_10_90_ms': float('nan'),
        'crest_20ms': float('nan'), 'energy_0_10': 0.0, 'energy_10_30': 0.0,
    }
    if n < 10:
        return empty
    env = np.array([np.sqrt(np.mean(x[i * hop: i * hop + win] ** 2) + 1e-30)
                     for i in range(n)])
    t_ms = np.arange(n) * (hop / sr) * 1000.0
    peak_all = env.max() + 1e-12

    onsets = []
    for thr_db in onset_thresholds_db:
        thr = peak_all * 10.0 ** (thr_db / 20.0)
        above = np.where(env > thr)[0]
        onsets.append(float(t_ms[above[0]]) if len(above) else 0.0)
    onset_ms = float(np.median(onsets))
    onset_spread_ms = float(max(onsets) - min(onsets))

    win_mask = (t_ms >= onset_ms) & (t_ms <= onset_ms + early_window_ms)
    if not win_mask.any():
        return {**empty, 'onset_ms': onset_ms, 'onset_threshold_spread_ms': onset_spread_ms}
    local_t, local_e = t_ms[win_mask], env[win_mask]
    pk_idx = int(np.argmax(local_e))
    early_peak_ms = float(local_t[pk_idx] - onset_ms)
    early_peak_amp = float(local_e[pk_idx])

    rise_mask = local_t <= local_t[pk_idx]
    rt, re = local_t[rise_mask], local_e[rise_mask]
    lo_thr, hi_thr = 0.10 * early_peak_amp, 0.90 * early_peak_amp
    above_lo = np.where(re >= lo_thr)[0]
    above_hi = np.where(re >= hi_thr)[0]
    rise_10_90_ms = (float(rt[above_hi[0]] - rt[above_lo[0]])
                      if len(above_lo) and len(above_hi) else float('nan'))

    crest_mask = (t_ms >= onset_ms) & (t_ms <= onset_ms + 20.0)
    seg20 = env[crest_mask]
    crest_20ms = (float(seg20.max() / (np.sqrt(np.mean(seg20 ** 2)) + 1e-12))
                   if len(seg20) else float('nan'))

    e0_10_mask = (t_ms >= onset_ms) & (t_ms < onset_ms + 10.0)
    e10_30_mask = (t_ms >= onset_ms + 10.0) & (t_ms < onset_ms + 30.0)
    energy_0_10 = (float(np.mean(env[e0_10_mask] ** 2) / (peak_all ** 2))
                    if e0_10_mask.any() else 0.0)
    energy_10_30 = (float(np.mean(env[e10_30_mask] ** 2) / (peak_all ** 2))
                     if e10_30_mask.any() else 0.0)

    return {
        'onset_ms': onset_ms,
        'onset_threshold_spread_ms': onset_spread_ms,
        'early_peak_ms': early_peak_ms,
        'rise_10_90_ms': rise_10_90_ms,
        'crest_20ms': crest_20ms,
        'energy_0_10': energy_0_10,
        'energy_10_30': energy_10_30,
    }


def _estimate_partial_t30(part_db: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Per-partial decay time to -30dB, fit independently for each harmonic
    from its own peak onward (mirrors _estimate_t60 but per-partial and at
    -30dB since slow bass partials rarely reach -60dB within TIME_ANCHORS).

    This is what makes the string loss filter (t60_scale / lp1_scale)
    identifiable: a broadband T60 can't tell "uniform loss" from "upper
    partials die faster than the fundamental" apart, but comparing T30 per
    harmonic can — see PHYSICAL_PIANO_MODELS_AUDIT_UA.md item #7.

    Returns shape (n_part,); NaN where a partial never rises above the
    noise floor or a usable decay slope can't be fit.
    """
    n_part = part_db.shape[1]
    out = np.full(n_part, np.nan)
    for k in range(n_part):
        e = part_db[:, k]
        if np.max(e) < -55.0:
            continue
        peak_idx = int(np.argmax(e))
        if peak_idx >= len(e) - 1:
            continue
        t = times[peak_idx:]
        ee = e[peak_idx:] - e[peak_idx]
        mask = ee > -30.0
        if mask.sum() >= 3:
            slope, _ = np.polyfit(t[mask], ee[mask], 1)
        elif len(ee) >= 2:
            # decayed past -30dB before the next time anchor: coarse 2-point slope
            slope = (ee[1] - ee[0]) / max(t[1] - t[0], 1e-6)
        else:
            continue
        if slope >= -0.5:
            continue  # flat/rising — no usable decay estimate
        out[k] = -30.0 / slope
    return out


def _estimate_inharmonicity(x: np.ndarray, sr: int, f0: float, n_partials: int = 8) -> float:
    """Estimate stiffness B from partial frequencies: f_k = k*f0*sqrt(1 + B*k²)."""
    nfft = 1 << int(np.ceil(np.log2(sr * 0.5)))  # ~0.5 s window
    nfft = min(nfft, 1 << 16)
    centre = min(len(x) - 1, int(sr * 0.2))
    freqs, mag = _stft_mag(x, sr, centre, nfft)
    df = freqs[1] - freqs[0]
    peaks_k = []
    peaks_f = []
    for k in range(1, n_partials + 1):
        f_nom = k * f0
        if f_nom > sr * 0.45:
            break
        lo = max(0, int((f_nom - 0.04 * f_nom) / df))
        hi = min(len(mag), int((f_nom + 0.10 * f_nom) / df) + 1)
        if hi - lo < 3:
            continue
        loc = lo + int(np.argmax(mag[lo:hi]))
        # Parabolic interpolation for sub-bin accuracy
        if 1 <= loc < len(mag) - 1 and mag[loc] > 0:
            a, b, c = mag[loc - 1], mag[loc], mag[loc + 1]
            denom = a - 2 * b + c
            offset = 0.5 * (a - c) / denom if abs(denom) > 1e-12 else 0.0
            fp = (loc + offset) * df
        else:
            fp = loc * df
        # ignore if peak is weak (likely noise)
        if mag[loc] < mag.max() * 0.01:
            continue
        peaks_k.append(k)
        peaks_f.append(fp)
    if len(peaks_k) < 3:
        return 1e-4
    ks = np.array(peaks_k, dtype=float)
    fs = np.array(peaks_f, dtype=float)
    # (fs/(k*f0))² = 1 + B*k² → y = B*k² + 1
    y = (fs / (ks * f0)) ** 2
    A = np.vstack([ks ** 2, np.ones_like(ks)]).T
    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    B_est = float(max(0.0, sol[0]))
    return min(B_est, 0.5)


# ---------------------------------------------------------------------------
def extract(x: np.ndarray, sr: int, f0: float) -> dict:
    """Compute the full feature dict for a single mono signal."""
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = x.astype(np.float64)
    # Strip leading silence
    x_t, _ = _trim_onset(x, sr)
    if len(x_t) < int(0.05 * sr):
        x_t = x
    peak = np.max(np.abs(x_t)) + 1e-12
    x_n = x_t / peak

    # Coarse (-60dB, looser than the -50dB trim above) leading-silence strip
    # for the attack_v2 descriptors, which need room to compare -35/-40/-50dB
    # onset thresholds against each other rather than having them pre-collapsed.
    x_coarse, _ = _trim_onset(x, sr, thresh_db=-60.0)
    if len(x_coarse) < int(0.05 * sr):
        x_coarse = x
    x_coarse_n = x_coarse / (np.max(np.abs(x_coarse)) + 1e-12)
    attack_v2 = _estimate_attack_v2(x_coarse_n, sr)

    # Amplitude envelope and T60
    env_db, env_t = _amplitude_envelope_db(x_n, sr)
    t60 = _estimate_t60(env_db, env_t)
    attack_idx = int(np.argmax(env_db))
    attack_time = float(env_t[attack_idx])

    # Spectral frames + centroid trajectory + mel bands
    nfft = 4096 if sr <= 48000 else 8192
    spec_bands = []
    centroids = []
    for t_anchor in TIME_ANCHORS:
        c = int(t_anchor * sr)
        if c >= len(x_n) - 16:
            spec_bands.append(np.full(len(BAND_EDGES) - 1, -60.0))
            centroids.append(0.0)
            continue
        freqs, mag = _stft_mag(x_n, sr, c, nfft)
        spec_bands.append(_mel_band_db(freqs, mag, BAND_EDGES))
        centroids.append(_spectral_centroid(freqs, mag))
    spec_bands = np.stack(spec_bands)  # (T, bands)

    # Per-partial amplitudes over time
    n_part = 12
    part_db = np.zeros((len(TIME_ANCHORS), n_part))
    for ti, t_anchor in enumerate(TIME_ANCHORS):
        c = int(t_anchor * sr)
        if c >= len(x_n) - 16:
            part_db[ti] = -60.0
            continue
        freqs, mag = _stft_mag(x_n, sr, c, nfft)
        df = freqs[1] - freqs[0]
        for k in range(1, n_part + 1):
            f_k = k * f0
            if f_k > sr * 0.45:
                part_db[ti, k - 1] = -60.0
                continue
            tol = max(df * 4, 0.05 * f_k)
            lo = max(0, int((f_k - tol) / df))
            hi = min(len(mag), int((f_k + tol * 1.5) / df) + 1)
            if hi - lo < 1:
                part_db[ti, k - 1] = -60.0
                continue
            m_peak = float(np.max(mag[lo:hi]))
            part_db[ti, k - 1] = 20.0 * np.log10(m_peak + 1e-9)

    inharm = _estimate_inharmonicity(x_n, sr, f0)
    partial_t30 = _estimate_partial_t30(part_db, TIME_ANCHORS)

    return {
        'amp_env_db': env_db,
        'amp_env_t': env_t,
        'spec_bands': spec_bands,
        'centroid_t': np.array(centroids),
        'part_db': part_db,
        'partial_t30': partial_t30,
        'attack_time': attack_time,
        'attack_v2': attack_v2,
        't60': t60,
        'inharm_B': inharm,
    }


def feature_loss(ref: dict, est: dict, weights: dict | None = None,
                  time_mask: np.ndarray | None = None) -> dict:
    """
    Weighted scalar loss + breakdown.  Returns dict with 'total' and components.

    All features are normalised so each component has comparable scale.

    time_mask: optional boolean array over TIME_ANCHORS restricting which
    time frames the spec/partials/centroid terms average over (e.g.
    features.ONSET_MASK for a hammer-only, attack-window fit — see
    PHYSICAL_PIANO_MODELS_AUDIT_UA.md's recommended fit-by-register,
    fit-by-parameter-group sequencing).  Doesn't affect env/t60/inharm,
    which aren't per-time-anchor; zero their weight explicitly if a
    fit phase should ignore them.
    """
    # Weights tuned for piano-like timbre: spectral envelope dominates because
    # that's what determines bright-vs-warm character; attack matters because
    # short impulsive attacks are the toy-piano / plucked-instrument signature.
    w = {
        'env': 0.6,
        'spec': 2.5,
        'partials': 2.0,
        'centroid': 0.8,
        't60': 0.7,
        'attack': 2.0,
        'inharm': 0.4,
        'partial_t30': 0.0,  # off by default; loss-filter fit phase turns it on
        'partial_t30_overring': 0.0,  # off by default; penalizes the WORST
                                       # over-long partial specifically (a
                                       # mean-based partial_t30 loss can be
                                       # satisfied by one partial ringing far
                                       # past its reference target — audible
                                       # as an isolated resonant "tail"/
                                       # "tunnel" even though the average
                                       # improved; see C4 register fit notes
                                       # in PHYSICAL_PIANO_MODELS_AUDIT_UA.md)
    }
    if weights:
        w.update(weights)

    # Envelope loss: L1 on dB envelope (already peak-normalised)
    env_loss = float(np.mean(np.abs(ref['amp_env_db'] - est['amp_env_db']))) / 20.0

    # Spectral envelope: align in dB then L1; subtract per-frame mean to remove
    # overall loudness so it focuses on timbre, not amplitude.
    a = ref['spec_bands'] - ref['spec_bands'].mean(axis=1, keepdims=True)
    b = est['spec_bands'] - est['spec_bands'].mean(axis=1, keepdims=True)
    if time_mask is not None:
        a, b = a[time_mask], b[time_mask]
    spec_loss = float(np.mean(np.abs(a - b))) / 20.0

    # Partials in dB — normalize each time slice
    a = ref['part_db'] - ref['part_db'].max(axis=1, keepdims=True)
    b = est['part_db'] - est['part_db'].max(axis=1, keepdims=True)
    if time_mask is not None:
        a, b = a[time_mask], b[time_mask]
    part_loss = float(np.mean(np.abs(a - b))) / 30.0

    # Centroid in log Hz (only where both are finite)
    rc, ec = ref['centroid_t'], est['centroid_t']
    if time_mask is not None:
        rc, ec = rc[time_mask], ec[time_mask]
    mask = (rc > 50) & (ec > 50)
    if mask.any():
        cent_loss = float(np.mean(np.abs(np.log2(rc[mask] / ec[mask]))))
    else:
        cent_loss = 1.0

    # T60 in log seconds
    t60_loss = abs(np.log2(max(est['t60'], 0.1) / max(ref['t60'], 0.1)))

    # Attack — log-ratio with 5ms floor so 5ms vs 18ms isn't tiny in linear terms
    floor = 0.005
    att_loss = abs(np.log2((est['attack_time'] + floor) / (ref['attack_time'] + floor)))

    # Inharmonicity in log
    inharm_loss = abs(np.log2(max(est['inharm_B'], 1e-6) / max(ref['inharm_B'], 1e-6)))

    # Per-partial T30 (log-ratio, only over harmonics where both sides got a
    # usable decay estimate) — identifies the loss filter's frequency shape,
    # not just overall T60.  See _estimate_partial_t30.
    rt30, et30 = ref.get('partial_t30'), est.get('partial_t30')
    if rt30 is not None and et30 is not None:
        valid = np.isfinite(rt30) & np.isfinite(et30) & (rt30 > 0) & (et30 > 0)
        if valid.any():
            log_ratio = np.log2(et30[valid] / rt30[valid])
            t30_loss = float(np.mean(np.abs(log_ratio)))
            # Worst single over-ringing partial (positive log_ratio only —
            # a partial decaying too FAST isn't the "tunnel" failure mode,
            # only one ringing past its reference target is).
            overring = np.maximum(log_ratio, 0.0)
            t30_overring_loss = float(np.max(overring)) if overring.size else 0.0
        else:
            t30_loss = 1.0
            t30_overring_loss = 1.0
    else:
        t30_loss = 0.0
        t30_overring_loss = 0.0

    parts = {
        'env': w['env'] * env_loss,
        'spec': w['spec'] * spec_loss,
        'partials': w['partials'] * part_loss,
        'centroid': w['centroid'] * cent_loss,
        't60': w['t60'] * t60_loss,
        'attack': w['attack'] * att_loss,
        'inharm': w['inharm'] * inharm_loss,
        'partial_t30': w['partial_t30'] * t30_loss,
        'partial_t30_overring': w['partial_t30_overring'] * t30_overring_loss,
    }
    parts['total'] = sum(parts.values())
    return parts
