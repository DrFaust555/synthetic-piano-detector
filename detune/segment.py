"""Audio loading, onset segmentation and per-note pitch / inharmonicity estimation.

The detuning analyzer needs, for each struck note, a clean "ring" segment plus a
reliable fundamental f0 and inharmonicity coefficient B.  We don't attempt full
transcription — only a robust dominant-pitch estimate per onset (and up to a few
pitches for a chord/triad input), which is enough to place partial bands.
"""

from __future__ import annotations
import numpy as np
from numpy.fft import rfft, rfftfreq
from dataclasses import dataclass, field

import soundfile as sf

# Reuse the project's vetted helpers (numpy-only, no calibration coupling).
from calib.features import _estimate_inharmonicity, _trim_onset


# ---------------------------------------------------------------------------
def midi_to_freq(m: float) -> float:
    return 440.0 * 2.0 ** ((m - 69) / 12.0)


def freq_to_midi(f: float) -> float:
    return 69.0 + 12.0 * np.log2(max(f, 1e-9) / 440.0)


def num_strings_real(midi: int) -> int:
    """Real acoustic string count per note (not the synth model's 4-voice trick).

    Lowest octave is single wound (monochord), then bichords, then trichords.
    Boundaries are approximate and vary by instrument; this is the typical layout.
    """
    if midi <= 27:   # ~ up to A0..D#1: single thick wound string
        return 1
    if midi <= 40:   # ~ E1..E2: bichord
        return 2
    return 3         # trichord


def _next_pow2(n: int) -> int:
    return 1 << int(np.ceil(np.log2(max(n, 2))))


def load_mono(path: str) -> tuple[np.ndarray, int]:
    """Load a WAV as mono float64 at its native sample rate (project convention)."""
    x, sr = sf.read(path)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x.astype(np.float64), int(sr)


# ---------------------------------------------------------------------------
def detect_onsets(x: np.ndarray, sr: int, nfft: int = 2048, hop: int = 512,
                  min_gap_s: float = 0.06) -> np.ndarray:
    """Spectral-flux onset detection.  Returns onset sample indices.

    Flux is accumulated online (one magnitude frame held at a time) so memory
    stays bounded for multi-minute files.
    """
    from scipy.signal import find_peaks

    if len(x) < nfft * 2:
        return np.array([0], dtype=int)

    win = np.hanning(nfft)
    n_frames = 1 + (len(x) - nfft) // hop
    flux = np.zeros(max(n_frames - 1, 1))
    prev = None
    for i in range(n_frames):
        seg = x[i * hop: i * hop + nfft] * win
        m = np.log1p(np.abs(rfft(seg)))
        if prev is not None:
            flux[i - 1] = np.sum(np.maximum(m - prev, 0.0))
        prev = m

    if flux.max() <= 0:
        return np.array([0], dtype=int)
    flux = flux / flux.max()

    # Adaptive threshold: local mean (≈0.3 s) plus a margin.
    w = max(3, int(0.3 * sr / hop))
    kernel = np.ones(w) / w
    local = np.convolve(flux, kernel, mode='same')
    height = np.maximum(0.08, local + 0.06)
    dist = max(1, int(min_gap_s * sr / hop))
    peaks, _ = find_peaks(flux, height=height, distance=dist, prominence=0.04)

    if len(peaks) == 0:
        return np.array([0], dtype=int)
    # flux[i] corresponds to the transition into frame i+1 → onset near (i+1)*hop.
    onsets = (peaks + 1) * hop
    return onsets.astype(int)


# ---------------------------------------------------------------------------
def _harmonic_salience(logmag: np.ndarray, df: float, sr: int,
                       midi_lo: int = 21, midi_hi: int = 96,
                       n_harm: int = 6) -> tuple[np.ndarray, np.ndarray]:
    """Harmonic-sum pitch salience over candidate MIDI notes.

    Returns (midis, salience).  Salience(m) = mean over first n_harm harmonics of
    the peak log-magnitude in a ±1.5%% window around k*f0.
    """
    midis = np.arange(midi_lo, midi_hi + 1)
    sal = np.zeros(len(midis), dtype=float)
    nyq = sr * 0.45
    for i, m in enumerate(midis):
        f0 = midi_to_freq(m)
        s = 0.0
        cnt = 0
        for k in range(1, n_harm + 1):
            fk = k * f0
            if fk > nyq:
                break
            lo = int((fk * 0.985) / df)
            hi = int((fk * 1.015) / df) + 1
            if hi <= lo or lo >= len(logmag):
                continue
            s += float(logmag[lo:min(hi, len(logmag))].max())
            cnt += 1
        sal[i] = s / cnt if cnt else 0.0
    return midis, sal


def _refine_f0(mag: np.ndarray, df: float, f_guess: float, tol: float = 0.03) -> float:
    """Parabolic-interpolated spectral peak near f_guess (±tol fraction)."""
    lo = max(1, int(f_guess * (1 - tol) / df))
    hi = min(len(mag) - 1, int(f_guess * (1 + tol) / df) + 1)
    if hi <= lo:
        return f_guess
    loc = lo + int(np.argmax(mag[lo:hi]))
    if 1 <= loc < len(mag) - 1 and mag[loc] > 0:
        a, b, c = mag[loc - 1], mag[loc], mag[loc + 1]
        denom = a - 2 * b + c
        off = 0.5 * (a - c) / denom if abs(denom) > 1e-12 else 0.0
        return (loc + off) * df
    return loc * df


def _is_harmonic_ghost(f_cand: float, kept_freqs: list[float],
                       tol_cents: float = 25.0) -> bool:
    """True if f_cand is an integer-multiple harmonic (2x, 3x, ...) of a kept pitch
    (or vice-versa) — i.e. an octave/12th/double-octave ghost, not a real interval."""
    for fp in kept_freqs:
        hi, lo = max(f_cand, fp), min(f_cand, fp)
        ratio = hi / lo
        n = round(ratio)
        if n >= 2 and abs(1200.0 * np.log2(ratio / n)) <= tol_cents:
            return True
    return False


def estimate_pitches(x_win: np.ndarray, sr: int, max_pitches: int = 1
                     ) -> list[tuple[float, float]]:
    """Estimate up to `max_pitches` dominant fundamentals in a window.

    Returns list of (f0_hz, salience) sorted by salience desc.  Uses a harmonic-sum
    salience with a fundamental-energy check to suppress sub-octave (octave-down)
    errors, then refines f0 on the real spectral peak.
    """
    if len(x_win) < 256:
        return []
    nfft = int(np.clip(_next_pow2(len(x_win)), 1 << 14, 1 << 18))
    win = np.hanning(len(x_win))
    mag = np.abs(rfft(x_win * win, nfft))
    if mag.max() <= 0:
        return []
    df = sr / nfft
    logmag = np.log1p(mag)

    midis, sal = _harmonic_salience(logmag, df, sr)
    # Local maxima of the salience curve are pitch candidates.
    cand = []
    for i in range(len(midis)):
        left = sal[i - 1] if i > 0 else -1
        right = sal[i + 1] if i < len(midis) - 1 else -1
        if sal[i] >= left and sal[i] >= right and sal[i] > 0:
            cand.append((sal[i], int(midis[i])))
    cand.sort(reverse=True)

    picked: list[tuple[float, float]] = []
    used_midi: list[int] = []
    for s, m in cand:
        if len(picked) >= max_pitches:
            break
        # Skip if within a semitone of an already-picked (stronger) pitch.
        if any(abs(m - um) <= 1 for um in used_midi):
            continue
        # Skip integer-ratio harmonic ghosts (octave 2:1, 12th 3:1, ...) of an
        # already-picked pitch — but NOT real intervals like a fifth (3:2), whose
        # ratio is not near an integer, so genuine chord tones survive.
        if picked and _is_harmonic_ghost(midi_to_freq(m), [p[0] for p in picked]):
            continue
        f0 = midi_to_freq(m)
        # Sub-octave guard: if the true fundamental bin is weak but the octave-up
        # is strong, the salience likely latched onto a sub-harmonic → bump octave.
        f0r = _refine_f0(mag, df, f0)
        b_f0 = int(round(f0r / df))
        b_2f0 = int(round(2 * f0r / df))
        if (b_f0 < len(mag) and b_2f0 < len(mag)
                and mag[b_f0] < 0.20 * mag[b_2f0] and m + 12 <= 96):
            f0r = _refine_f0(mag, df, 2 * f0r)
            m = m + 12
        picked.append((f0r, float(s)))
        used_midi.append(m)
    return picked


# ---------------------------------------------------------------------------
def refine_f0_multipartial(x_win: np.ndarray, sr: int, f0: float, B: float,
                           n_partials: int = 8) -> float:
    """Accurate f0 from several inharmonicity-corrected partials.

    Each partial k sits at fk = k*f0*sqrt(1+B*k²); inverting gives an f0 estimate
    per partial.  The magnitude-weighted MEDIAN over partials is used (not the
    least-squares estimator of Rigaud eq. 20): on polyphonic performance audio
    individual partials are frequently contaminated by other sounding notes, and
    the median is far more robust to such outliers than a least-squares mean —
    empirically this halves the global-tuning residual on real recordings.
    """
    nfft = int(np.clip(_next_pow2(len(x_win)), 1 << 15, 1 << 18))
    mag = np.abs(rfft(x_win * np.hanning(len(x_win)), nfft))
    if mag.max() <= 0:
        return f0
    df = sr / nfft
    ests = []
    wts = []
    nyq = sr * 0.45
    for k in range(1, n_partials + 1):
        fk = k * f0 * np.sqrt(1.0 + B * k * k)
        if fk > nyq:
            break
        fp = _refine_f0(mag, df, fk, tol=0.02)
        if fp <= 0:
            continue
        loc = int(round(fp / df))
        if loc >= len(mag):
            continue
        ests.append(fp / (k * np.sqrt(1.0 + B * k * k)))
        wts.append(float(mag[loc]))
    if not ests:
        return f0
    ests = np.array(ests)
    wts = np.array(wts)
    order = np.argsort(ests)
    cw = np.cumsum(wts[order])
    if cw[-1] <= 0:
        return float(np.median(ests))
    return float(ests[order][np.searchsorted(cw, cw[-1] / 2.0)])


@dataclass
class NoteSegment:
    onset: int            # sample index of onset
    x: np.ndarray         # ring window (mono float64), onset-trimmed
    sr: int
    f0: float             # refined fundamental (Hz)
    midi: int             # nearest MIDI note
    B: float              # inharmonicity coefficient
    energy: float         # RMS energy of the ring (weight)
    n_strings: int        # expected real string count
    salience: float = 0.0
    extra_f0: list = field(default_factory=list)  # other pitches (chord/triad)


def _segment_energy(seg: np.ndarray) -> float:
    return float(np.sqrt(np.mean(seg ** 2) + 1e-20))


def _make_segment(segments, onset, ring, sr, skip_attack_s, max_pitches,
                  known_f0=None):
    """Build NoteSegment(s) from a ring window and append to `segments`.

    If known_f0 is given (piece mode), the pitch is taken as-is (just refined),
    rather than re-detected from the ring — so overlapping chord tones don't all
    collapse onto the loudest pitch.
    """
    if len(ring) < int(0.08 * sr):
        return
    rt, _ = _trim_onset(ring, sr, thresh_db=-45.0)
    if len(rt) < int(0.06 * sr):
        rt = ring
    a = min(int(skip_attack_s * sr), max(0, len(rt) - 1))
    win = rt[a:]
    if len(win) < int(0.08 * sr):
        return
    if known_f0 is not None:
        kept = [(float(known_f0), 1.0)]
    else:
        pitches = estimate_pitches(win, sr, max_pitches=max_pitches)
        if not pitches:
            return
        # Drop weak spurious extra pitches (keep clear chord tones, ditch ghosts).
        top_sal = pitches[0][1]
        kept = [pitches[0]] + [(f, s) for f, s in pitches[1:] if s >= 0.55 * top_sal]
    energy = _segment_energy(rt)
    norm = win / (np.max(np.abs(win)) + 1e-12)
    primary = kept[0][0]
    for f0, sal in kept:
        if f0 < 24.0 or f0 > 4500.0:
            continue
        try:
            B = _estimate_inharmonicity(norm, sr, f0)
        except Exception:
            B = 1e-4
        f0 = refine_f0_multipartial(win, sr, f0, B)   # accurate f0 for global tuning
        midi = int(round(freq_to_midi(f0)))
        segments.append(NoteSegment(
            onset=int(onset), x=win, sr=sr, f0=f0, midi=midi, B=B,
            energy=energy if f0 == primary else energy * 0.7,
            n_strings=num_strings_real(midi), salience=sal,
            extra_f0=[],
        ))


def _quick_pitch(x: np.ndarray, sr: int, o: int) -> float | None:
    w = x[o + int(0.02 * sr): o + int(0.27 * sr)]
    if len(w) < int(0.1 * sr):
        return None
    p = estimate_pitches(w, sr, max_pitches=1)
    return p[0][0] if p else None


def build_segments(x: np.ndarray, sr: int,
                   ring_min_s: float = 0.12, ring_max_s: float = 2.5,
                   skip_attack_s: float = 0.03, max_notes: int = 400
                   ) -> tuple[list[NoteSegment], dict]:
    """Segment audio into per-note ring windows with pitch + B estimates.

    The regime is decided by how many distinct STRIKE TIMES there are (not by
    duration — unison beating is audible in 2-4 s, so short clips must work):
      * one strike (a single note or a chord struck once, then sustained) → analyze
        that one window with up to 3 dominant pitches.  Best case for the high-res
        split fit.
      * several strikes (a performance, even a 2-3 s phrase) → onset-segment each
        note.  Consecutive SAME-pitch onsets are merged so a held/beating note isn't
        chopped into useless sub-windows; each note's known pitch is carried in so
        overlapping chord tones aren't collapsed.
    """
    info: dict = {'truncated': False, 'mode': 'single/chord'}
    dur = len(x) / sr
    segments: list[NoteSegment] = []
    ring_max = int(ring_max_s * sr)

    onsets = detect_onsets(x, sr)
    onsets = onsets[onsets < len(x) - int(0.05 * sr)]
    if len(onsets) == 0:
        onsets = np.array([0], dtype=int)
    pit = [_quick_pitch(x, sr, int(o)) for o in onsets]

    # Count distinct strike times (onsets separated by > 120 ms).
    gap = int(0.12 * sr)
    n_strikes = 1 + int(np.sum(np.diff(onsets) > gap)) if len(onsets) > 1 else 1

    # --- single note / chord: one window, up to 3 pitches ---
    if n_strikes <= 1:
        _make_segment(segments, int(onsets[0]), x[int(onsets[0]):], sr,
                      skip_attack_s, max_pitches=3)
        info.update(mode='single/chord', n_onsets=int(len(onsets)),
                    n_segments=len(segments), duration_s=dur)
        return segments, info

    # --- performance: merge same-pitch runs, then per-note rings ---
    info['mode'] = 'piece'
    events = []  # (start_sample, pitch)
    i = 0
    while i < len(onsets):
        j = i
        while (j + 1 < len(onsets) and pit[i] and pit[j + 1] and pit[i] > 0
               and pit[j + 1] > 0
               and abs(1200.0 * np.log2(pit[j + 1] / pit[i])) <= 40.0
               and (onsets[j + 1] - onsets[i]) < ring_max):
            j += 1
        events.append((int(onsets[i]), pit[i]))
        i = j + 1

    if len(events) > max_notes:
        ens = [np.sqrt(np.mean(x[s:min(len(x), s + int(0.1 * sr))] ** 2) + 1e-20)
               for s, _ in events]
        keep = sorted(np.argsort(ens)[-max_notes:])
        events = [events[k] for k in keep]
        info['truncated'] = True

    starts = [e[0] for e in events]
    for k, (s, p) in enumerate(events):
        # ring end = next strike that is meaningfully later (skip simultaneous
        # chord tones so each gets a proper sustain to analyze)
        end = len(x)
        for m in range(k + 1, len(events)):
            if starts[m] - s > gap:
                end = starts[m]
                break
        ring_len = min(ring_max, end - s)
        if ring_len < int(ring_min_s * sr):
            continue
        _make_segment(segments, s, x[s: s + ring_len], sr, skip_attack_s,
                      max_pitches=1, known_f0=p)

    info.update(n_onsets=int(len(onsets)), n_events=len(events),
                n_segments=len(segments), duration_s=dur)
    return segments, info
