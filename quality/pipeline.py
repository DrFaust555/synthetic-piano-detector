"""Per-file intrinsic-feature pipeline shared by extraction and prediction.

Two stages so the *same* register normalisation is applied at train and predict
time:

  1. ``note_raw_descriptors(path)`` — segment a recording (``detune.build_segments``),
     run ``calib.features.extract`` per note, reduce to recording-stable scalar
     descriptors, and run ``detune.analyze`` once for file-level tuning features.
  2. ``fit_trends`` (dataset only) learns each descriptor's smooth dependence on
     pitch (MIDI); ``aggregate_file`` subtracts that trend (register normalisation)
     and reduces the per-note residuals to a file-level vector.

Only *intrinsic* traits are used (timbre + tuning); no loudness/stereo/noise-floor
recording descriptors.  Brightness is expressed relative to the fundamental
(centroid / f0) and partial structure as level-invariant ratios/slopes so the
mic/EQ confound is reduced (not eliminated — see report caveats).
"""

from __future__ import annotations
import warnings
import numpy as np

from detune.segment import load_mono, build_segments
from detune.analyze import analyze
from calib.features import extract, TIME_ANCHORS
from quality.spectral import nonharm_ratio, phantom_index
from quality.waveform import inflection_density

# --- per-note scalar descriptors (all register-normalised before aggregation) ---
DESCRIPTORS = [
    'log_t60',            # log10 decay-to-60dB time  (sustain)
    'attack_time',        # onset->peak seconds
    'log_inharmB',        # log10 inharmonicity coefficient B (string stiffness)
    'centroid_h',         # spectral centroid in units of f0 (brightness, EQ-robust)
    'bright_decay',       # change in centroid_h from early to late (darkening rate)
    'spectral_slope',     # dB per partial across first 12 harmonics (level-invariant)
    'even_odd_db',        # mean(even partials) - mean(odd partials), dB
    'high_partial_frac',  # energy fraction in partials 5..12 (harmonic richness)
    'n_partials_above',   # count of partials within 40 dB of the strongest
    'sustain_db_1s',      # envelope level (dB rel peak) 1 s after onset
    # --- added channel-robust descriptors (ratios / temporal — EQ/level-invariant) ---
    'tristim2',           # energy fraction in partials 2..4 (tristimulus T2)
    'spec_irregularity',  # mean |ΔdB| between adjacent partials (envelope jaggedness)
    'hf_decay_excess',    # (HF − LF) energy change first→last valid frame, dB
    'centroid_var',       # std of centroid/f0 over decay (brightness stability)
    'partial_persist',    # # partials surviving to last valid frame / first frame
    'infl_density',       # waveform inflections per period (shape complexity; model too smooth)
]

# --- file-level tuning features (already register-corrected inside detune) ------
TUNING = [
    'detune_final',       # integral detuning 0..100 (0 = perfect)
    'unison_index',       # unison-spread index 0..100
    'global_index',       # note-to-note tuning-consistency index 0..100
    'med_spread_cents',   # typical resolved unison split (cents)
    'frac_split',         # fraction of notes with a resolved unison split
    'global_scatter',     # residual pitch scatter (cents)
    'ref_offset_abs',     # |reference offset vs A4=440| (cents)
]

# Register-localised descriptors, NOT register-normalised (the effect is register-
# specific). Fundamental-centric structure aggregated over the bass AND, as a
# control, over the upper register; phantom/non-harmonic content over the DEEP bass
# where longitudinal modes live.
FUND_DESCRIPTORS = ['fund_strength', 'h2_over_h1', 'body_no_fund']  # per note
DEEP_DESCRIPTORS = ['nonharm_ratio', 'phantom_idx']                 # per bass note
BASS_MAX_MIDI = 48     # bass aggregation cutoff (fundamental-centric)
DEEP_MAX_MIDI = 40     # deep-bass aggregation (phantom / non-harmonic)
UPPER_MIN_MIDI = 60    # upper-register aggregation (C4+), as a control

BASS_FEATURE_COLS = ([f'bass_{d}' for d in FUND_DESCRIPTORS]
                     + [f'upper_{d}' for d in FUND_DESCRIPTORS]
                     + [f'deepbass_{d}' for d in DEEP_DESCRIPTORS])

FEATURE_COLS = ([f'{d}_med' for d in DESCRIPTORS]
                + [f'{d}_iqr' for d in DESCRIPTORS]
                + BASS_FEATURE_COLS
                + TUNING)

META_COLS = ['file', 'rating', 'type', 'source', 'category', 'mic_moving',
             'n_notes', 'median_midi', 'duration_s', 'mode']

# How many (highest-energy) notes per file feed the timbre extractor.  Bounds
# runtime and down-weights faint mush; tuning still uses all notes via detune.
MAX_TIMBRE_NOTES = 80

# How many notes per file feed the NEURAL branch.  Far fewer than the analytic
# branch: the file label is shared across notes, so neighbouring notes are highly
# redundant — a register-spread handful carries almost all the information while
# cutting the training set ~5x (the data-reduction lever).
N_NEURO_NOTES = 15


# ---------------------------------------------------------------------------
def select_diverse_notes(segs, n_target: int = N_NEURO_NOTES) -> list[int]:
    """Pick ~n_target note indices spread across the pitch range (not just the
    loudest).  Bins notes by octave and round-robins the loudest per bin, so a
    bright-treble vs warm-bass file is represented across registers rather than by
    a clump of similar mid notes.  Returns indices into `segs`.
    """
    n = len(segs)
    if n <= n_target:
        return list(range(n))
    # group note indices by octave bin, each sorted loudest-first
    bins: dict[int, list[int]] = {}
    for i, s in enumerate(segs):
        bins.setdefault(int(s.midi) // 12, []).append(i)
    for b in bins.values():
        b.sort(key=lambda i: -segs[i].energy)
    # round-robin across bins until we have n_target
    keep: list[int] = []
    order = sorted(bins.keys())
    pos = {b: 0 for b in order}
    while len(keep) < n_target:
        progressed = False
        for b in order:
            if pos[b] < len(bins[b]):
                keep.append(bins[b][pos[b]])
                pos[b] += 1
                progressed = True
                if len(keep) >= n_target:
                    break
        if not progressed:
            break
    return sorted(keep)


# ---------------------------------------------------------------------------
def _first_valid(valid: np.ndarray, idxs: list[int]):
    for i in idxs:
        if i < len(valid) and valid[i]:
            return i
    return None


def _partial_stats(prof: np.ndarray):
    """From a 12-vector of partial amplitudes (dB; -60 = absent) derive
    level-invariant structure descriptors."""
    present = prof > -55.0
    ks = np.arange(1, 13)
    slope = even_odd = high_frac = np.nan
    n_above = np.nan
    if present.sum() >= 3:
        kk = ks[present].astype(float)
        vv = prof[present]
        slope = float(np.polyfit(kk, vv, 1)[0])
        n_above = int(np.sum(vv > (vv.max() - 40.0)))
        ev = present & (ks % 2 == 0)
        od = present & (ks % 2 == 1)
        if ev.any() and od.any():
            even_odd = float(prof[ev].mean() - prof[od].mean())
        pw = np.where(present, 10.0 ** (prof / 10.0), 0.0)
        total = pw.sum()
        if total > 0:
            high_frac = float(pw[4:].sum() / total)   # partials 5..12
    return slope, even_odd, high_frac, n_above


def _note_desc(feat: dict, f0: float) -> dict:
    """Reduce a calib.features.extract() dict to scalar intrinsic descriptors."""
    cen = feat['centroid_t']
    valid = cen > 50.0
    cen_h = cen / max(f0, 1e-6)
    # anchors: early ~0.10/0.05/0.02 s, late ~0.80/0.40 s
    ei = _first_valid(valid, [2, 1, 0])
    li = _first_valid(valid, [5, 4])
    centroid_h = float(cen_h[ei]) if ei is not None else np.nan
    bright_decay = (float(cen_h[li] - cen_h[ei])
                    if (ei is not None and li is not None) else np.nan)

    # representative partial profile: mean over valid early-mid frames
    pf = feat['part_db']
    frames = [i for i in (1, 2, 3) if i < len(valid) and valid[i]]
    if not frames:
        frames = [i for i in range(len(valid)) if valid[i]]
    fund_strength = h2_over_h1 = body_no_fund = np.nan
    tristim2 = spec_irreg = np.nan
    if frames:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            prof = np.nanmean(pf[frames], axis=0)
        slope, even_odd, high_frac, n_above = _partial_stats(prof)
        # fundamental-centric structure (the strong bass discriminator) — computed
        # per note; aggregated over the BASS register only (see aggregate_file).
        pw = np.where(prof > -55.0, 10.0 ** (prof / 10.0), 0.0)
        tot = pw[:12].sum()
        fund_strength = float(pw[0] / tot) if tot > 0 else np.nan
        h26, h712 = pw[1:6].sum(), pw[6:12].sum()
        body_no_fund = float(np.log10(h26 / h712)) if (h26 > 0 and h712 > 0) else np.nan
        h2_over_h1 = float(np.log10(pw[1] / pw[0])) if (pw[0] > 0 and pw[1] > 0) else np.nan
        # tristimulus T2 (partials 2..4) and harmonic-envelope jaggedness (both EQ-robust)
        tristim2 = float(pw[1:4].sum() / tot) if tot > 0 else np.nan
        pres = prof > -55.0
        spec_irreg = (float(np.mean(np.abs(np.diff(prof[pres]))))
                      if pres.sum() >= 3 else np.nan)
    else:
        slope = even_odd = high_frac = n_above = np.nan

    # sustain level 1 s after onset (interp on the log-spaced envelope)
    t, e = feat['amp_env_t'], feat['amp_env_db']
    sustain_1s = float(np.interp(1.0, t, e)) if (t[0] <= 1.0 <= t[-1]) else np.nan

    # brightness stability over the decay (std of centroid/f0 on valid frames)
    centroid_var = float(np.std(cen_h[valid])) if valid.sum() >= 3 else np.nan

    # HF-vs-LF relative decay + partial persistence over the AVAILABLE decay span
    # (first→last valid frame — adaptive, so it works for short performance notes too).
    def _band_db(fr, lo, hi):
        seg = pf[fr, lo:hi]
        return 10.0 * np.log10(np.where(seg > -55.0, 10.0 ** (seg / 10.0), 0.0).sum() + 1e-9)
    def _n_pres(fr):
        seg = pf[fr]
        return int(np.sum(seg > seg.max() - 40.0))
    hf_decay_excess = partial_persist = np.nan
    vidx = np.where(valid)[0]
    if len(vidx) >= 2:
        ef, lf = int(vidx[0]), int(vidx[-1])
        lf_dec = _band_db(lf, 0, 4) - _band_db(ef, 0, 4)
        hf_dec = _band_db(lf, 4, 12) - _band_db(ef, 4, 12)
        hf_decay_excess = float(hf_dec - lf_dec)
        partial_persist = float(_n_pres(lf) / max(1, _n_pres(ef)))

    return {
        'log_t60': float(np.log10(np.clip(feat['t60'], 0.05, 30.0))),
        'attack_time': float(feat['attack_time']),
        'log_inharmB': float(np.log10(np.clip(feat['inharm_B'], 1e-6, 0.5))),
        'centroid_h': centroid_h,
        'bright_decay': bright_decay,
        'spectral_slope': slope,
        'even_odd_db': even_odd,
        'high_partial_frac': high_frac,
        'n_partials_above': float(n_above) if n_above is not None else np.nan,
        'sustain_db_1s': sustain_1s,
        'tristim2': tristim2,
        'spec_irregularity': spec_irreg,
        'hf_decay_excess': hf_decay_excess,
        'centroid_var': centroid_var,
        'partial_persist': partial_persist,
        'fund_strength': fund_strength,
        'h2_over_h1': h2_over_h1,
        'body_no_fund': body_no_fund,
    }


def note_raw_descriptors(path: str) -> tuple[dict, list[dict], dict]:
    """Return (meta, per-note raw descriptor dicts, tuning-feature dict) for a file.

    Per-note dicts carry 'midi' and 'energy' for register fitting / weighting.
    """
    x, sr = load_mono(path)
    segs, info = build_segments(x, sr)
    if len(segs) > MAX_TIMBRE_NOTES:
        order = np.argsort([s.energy for s in segs])[-MAX_TIMBRE_NOTES:]
        keep = set(int(i) for i in order)
        # force-include the loudest bass notes so the bass aggregation has coverage
        bass_i = [i for i in range(len(segs)) if segs[i].midi <= BASS_MAX_MIDI]
        bass_i.sort(key=lambda i: -segs[i].energy)
        keep.update(bass_i[:40])
        segs_t = [segs[i] for i in sorted(keep)]
    else:
        segs_t = segs

    notes = []
    midis = []
    for s in segs_t:
        if s.f0 < 24.0 or s.f0 > 4500.0:
            continue
        try:
            feat = extract(s.x, sr, s.f0)
        except Exception:
            continue
        d = _note_desc(feat, s.f0)
        d['midi'] = int(s.midi)
        d['energy'] = float(s.energy)
        try:
            d['infl_density'] = inflection_density(s.x, sr, s.f0)
        except Exception:
            d['infl_density'] = np.nan
        # phantom / non-harmonic content only matters (and is only used) in the
        # bass; skip the extra FFTs elsewhere.
        if s.midi <= BASS_MAX_MIDI:
            B = float(feat['inharm_B'])
            d['nonharm_ratio'] = nonharm_ratio(s.x, sr, s.f0, B)
            d['phantom_idx'] = phantom_index(s.x, sr, s.f0, B)
        notes.append(d)
        midis.append(int(s.midi))

    # tuning (uses all notes internally; one extra segmentation pass)
    try:
        r = analyze(path)
        g = r['global']
        tuning = {
            'detune_final': r['final'],
            'unison_index': r['unison_index'],
            'global_index': r['global_index'],
            'med_spread_cents': r['med_spread_cents'],
            'frac_split': r['frac_split'],
            'global_scatter': g.scatter_cents if g.available else None,
            'ref_offset_abs': abs(g.ref_offset_cents) if g.available else None,
        }
    except Exception:
        tuning = {k: None for k in TUNING}
    tuning = {k: (np.nan if v is None else float(v)) for k, v in tuning.items()}

    meta = {
        'n_notes': len(notes),
        'median_midi': float(np.median(midis)) if midis else np.nan,
        'duration_s': float(info.get('duration_s', len(x) / sr)),
        'mode': info.get('mode', ''),
    }
    return meta, notes, tuning


# ---------------------------------------------------------------------------
def fit_trends(all_notes: list[dict], bin_w: int = 6, min_per_bin: int = 4) -> dict:
    """Learn each descriptor's median trend vs MIDI (non-parametric, robust).

    Returns {desc: {'bc': bin_centers, 'bv': bin_medians}} for np.interp at apply
    time.  Recordings differ in which notes are played, so subtracting this trend
    removes the trivial bass-vs-treble artefact before cross-file comparison.
    """
    trends = {}
    midis = np.array([n['midi'] for n in all_notes], dtype=float)
    for d in DESCRIPTORS:
        v = np.array([n.get(d, np.nan) for n in all_notes], dtype=float)
        ok = np.isfinite(v) & np.isfinite(midis)
        if ok.sum() < min_per_bin:
            trends[d] = {'bc': [60.0], 'bv': [float(np.nanmedian(v)) if ok.any() else 0.0]}
            continue
        m, vv = midis[ok], v[ok]
        lo, hi = int(np.floor(m.min())), int(np.ceil(m.max()))
        edges = list(range(lo, hi + bin_w, bin_w))
        bc, bv = [], []
        for a, b in zip(edges[:-1], edges[1:]):
            sel = (m >= a) & (m < b)
            if sel.sum() >= min_per_bin:
                bc.append(float(m[sel].mean()))
                bv.append(float(np.median(vv[sel])))
        if len(bc) < 2:                       # fall back to a global constant
            bc, bv = [60.0], [float(np.median(vv))]
        trends[d] = {'bc': bc, 'bv': bv}
    return trends


def _detrend(desc: str, midi: float, val: float, trends: dict) -> float:
    if not np.isfinite(val):
        return np.nan
    t = trends[desc]
    bc, bv = t['bc'], t['bv']
    base = bv[0] if len(bc) < 2 else float(np.interp(midi, bc, bv))
    return val - base


def _med_iqr(vals: list[float]):
    a = np.array(vals, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return np.nan, np.nan
    med = float(np.median(a))
    iqr = float(np.percentile(a, 75) - np.percentile(a, 25)) if a.size >= 2 else 0.0
    return med, iqr


def aggregate_file(notes: list[dict], tuning: dict, trends: dict) -> dict:
    """Register-normalise per-note descriptors and reduce to a file feature row."""
    row = {}
    for d in DESCRIPTORS:
        res = [_detrend(d, n['midi'], n.get(d, np.nan), trends) for n in notes]
        med, iqr = _med_iqr(res)
        row[f'{d}_med'] = med
        row[f'{d}_iqr'] = iqr
    # register-localised medians, NOT register-normalised
    def grp_med(key, lo=None, hi=None):
        sel = [n for n in notes
               if (lo is None or n.get('midi', 999) >= lo)
               and (hi is None or n.get('midi', 999) <= hi)]
        a = np.array([n.get(key, np.nan) for n in sel], dtype=float)
        a = a[np.isfinite(a)]
        return float(np.median(a)) if a.size >= 1 else np.nan

    for d in FUND_DESCRIPTORS:
        row[f'bass_{d}'] = grp_med(d, hi=BASS_MAX_MIDI)
        row[f'upper_{d}'] = grp_med(d, lo=UPPER_MIN_MIDI)
    for d in DEEP_DESCRIPTORS:
        row[f'deepbass_{d}'] = grp_med(d, hi=DEEP_MAX_MIDI)
    for t in TUNING:
        row[t] = float(tuning.get(t, np.nan))
    return row
