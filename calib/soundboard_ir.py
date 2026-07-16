"""
Extract a data-driven soundboard frequency / impulse-response profile
from the bank of OGG reference samples.

Idea
----
Each recorded sample is roughly
    record_n(t) = string_n(t) * board_n(t) * room
The string content has harmonics at k·f0(n) — these positions *shift with
the note*.  The soundboard resonances and the room sit at *fixed
frequencies*.  Averaging log-magnitude spectra across many notes (with
different f0) smears the moving harmonics into a smooth background while
the fixed body modes reinforce — peaks survive averaging only if they're
common to every note.

This is the same trick astronomers use to extract continuum from many
emission-line spectra and what operational modal analysis uses to extract
structural modes from random excitation.

What we output
--------------
- An "averaged log-magnitude spectrum" of the soundboard+room aggregate
- A list of modal peaks (f, Q estimate, gain) found in that spectrum
- A 4096-tap FIR impulse response derived from the spectrum (random
  spectral phase, magnitude from data)
- A "per-note coupling" matrix: for each available midi note, regress
  the post-attack spectrum onto the modal basis to find how strongly
  that note excites each mode.

Run as a script to produce all artefacts under `calib/` and plots
under `out/`:

    python -m calib.soundboard_ir
"""

from __future__ import annotations
import json
import os
import re
import sys
import numpy as np
import soundfile as sf
from numpy.fft import rfft, irfft, rfftfreq
from scipy.signal import resample_poly

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

# ---------------------------------------------------------------------------
# Sample-pack file naming: C4_v127_p0.ogg, B0_v75_p1.ogg, etc.
_NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
def name_to_midi(name: str) -> int:
    m = re.match(r'^([A-G]#?)(-?\d+)$', name)
    if not m:
        raise ValueError(name)
    letter, octave = m.group(1), int(m.group(2))
    return (octave + 1) * 12 + _NOTE_NAMES.index(letter)


def list_samples(samples_dir: str):
    """Yield (midi, vel, variant, path) for every .ogg in the pack."""
    pat = re.compile(r'^([A-G]#?\d+)_v(\d+)_p(\d+)\.ogg$')
    for fn in sorted(os.listdir(samples_dir)):
        m = pat.match(fn)
        if not m:
            continue
        midi = name_to_midi(m.group(1))
        vel = int(m.group(2))
        variant = int(m.group(3))
        yield midi, vel, variant, os.path.join(samples_dir, fn)


# ---------------------------------------------------------------------------
SR_TARGET = 48000
NFFT = 8192        # ~5.5 Hz resolution at 48 kHz
HOP = NFFT // 4
WIN_AFTER_ATTACK_S = 0.060   # window starts 60 ms after detected onset
WIN_LENGTH_S = 0.500         # 500 ms window — enough to resolve sub-100 Hz


def _load_mono(path: str) -> tuple[np.ndarray, int]:
    a, sr = sf.read(path)
    if a.ndim > 1:
        a = a.mean(axis=1)
    if sr != SR_TARGET:
        a = resample_poly(a, SR_TARGET, sr)
        sr = SR_TARGET
    return a.astype(np.float64), sr


def _detect_onset(x: np.ndarray, sr: int) -> int:
    """Return sample index of audible onset."""
    win = max(1, sr // 1000)
    rms = np.sqrt(np.convolve(x * x, np.ones(win) / win, mode='same'))
    peak = rms.max() + 1e-30
    thr = peak * 10 ** (-30 / 20)
    idx = np.where(rms > thr)[0]
    return int(idx[0]) if len(idx) else 0


def _post_attack_spectrum(x: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Take WIN_LENGTH starting WIN_AFTER_ATTACK_S past the onset and
    return (freqs, log_mag in dB)."""
    onset = _detect_onset(x, sr)
    start = onset + int(WIN_AFTER_ATTACK_S * sr)
    end = start + int(WIN_LENGTH_S * sr)
    if end > len(x):
        end = len(x)
        start = max(0, end - int(WIN_LENGTH_S * sr))
    seg = x[start:end]
    if len(seg) < 128:
        return None, None
    # Pad / trim to NFFT
    if len(seg) < NFFT:
        seg = np.concatenate([seg, np.zeros(NFFT - len(seg))])
    else:
        seg = seg[:NFFT]
    seg = seg * np.hanning(NFFT)
    mag = np.abs(rfft(seg)) + 1e-12
    return rfftfreq(NFFT, 1.0 / sr), 20.0 * np.log10(mag)


# ---------------------------------------------------------------------------
def collect_spectra(samples_dir: str, vel_floor: int = 50) -> dict:
    """Build (n_samples, nfft//2+1) log-magnitude matrix and metadata."""
    rows = []
    metas = []
    for midi, vel, variant, path in list_samples(samples_dir):
        if vel < vel_floor:
            continue
        try:
            x, sr = _load_mono(path)
        except Exception as e:
            print(f'  skip {path}: {e}')
            continue
        freqs, log_mag = _post_attack_spectrum(x, sr)
        if log_mag is None:
            continue
        # Normalise so each sample contributes equally to the average
        log_mag = log_mag - log_mag.max()
        rows.append(log_mag)
        metas.append({'midi': midi, 'vel': vel, 'variant': variant, 'file': os.path.basename(path)})
    if not rows:
        raise RuntimeError('no samples loaded')
    M = np.stack(rows)
    return {'freqs': freqs, 'log_mag': M, 'meta': metas, 'sr': SR_TARGET}


# ---------------------------------------------------------------------------
def averaged_profile(spectra: dict) -> np.ndarray:
    """Mean log-magnitude across all samples — soundboard backbone."""
    return spectra['log_mag'].mean(axis=0)


def pick_peaks(freqs: np.ndarray, prof_db: np.ndarray, *,
               n_max: int = 40, min_sep_hz: float = 8.0,
               prominence_db: float = 1.5) -> list[tuple[float, float, float]]:
    """Find prominent peaks in the averaged profile.  Returns [(f, Q, gain_db)]."""
    # Subtract a smooth background (median filter in log-frequency)
    log_f = np.log10(np.maximum(freqs, 1.0))
    # Resample to uniform log-freq, take running median for "continuum"
    bins = np.linspace(np.log10(40), np.log10(SR_TARGET / 2), 400)
    # Naive: per-bin median over neighbours
    bg = np.copy(prof_db)
    for i in range(len(prof_db)):
        lo = max(0, i - 64)
        hi = min(len(prof_db), i + 64)
        bg[i] = np.median(prof_db[lo:hi])
    detrended = prof_db - bg

    # Pick peaks with minimum prominence
    cand = []
    df = freqs[1] - freqs[0]
    for i in range(1, len(detrended) - 1):
        if (detrended[i] > detrended[i - 1] and
            detrended[i] > detrended[i + 1] and
            detrended[i] > prominence_db and
            freqs[i] > 30 and freqs[i] < SR_TARGET * 0.45):
            cand.append((freqs[i], detrended[i], prof_db[i]))
    # Sort by detrended prominence, enforce minimum separation
    cand.sort(key=lambda t: -t[1])
    chosen = []
    for f, p, raw_db in cand:
        if all(abs(f - cf) >= min_sep_hz for cf, _, _ in chosen):
            # Estimate Q from -3 dB width around peak in raw profile
            i = int(round(f / df))
            i_lo = i; i_hi = i
            while i_lo > 0 and prof_db[i_lo] > prof_db[i] - 3.0:
                i_lo -= 1
            while i_hi < len(prof_db) - 1 and prof_db[i_hi] > prof_db[i] - 3.0:
                i_hi += 1
            bw = max(df, (i_hi - i_lo) * df)
            Q = f / bw
            chosen.append((float(f), float(Q), float(raw_db)))
            if len(chosen) >= n_max:
                break
    chosen.sort(key=lambda t: t[0])
    return chosen


# ---------------------------------------------------------------------------
def build_fir(freqs: np.ndarray, prof_db: np.ndarray, n_taps: int = 4096,
              max_gain_db: float = 0.0) -> np.ndarray:
    """
    Inverse FFT of a magnitude spectrum (with random phase) → FIR.
    The spectrum is normalised so peak == 0 dB.
    """
    # Resample prof_db to length n_taps // 2 + 1
    target_len = n_taps // 2 + 1
    src_freqs = freqs
    tgt_freqs = np.linspace(0, SR_TARGET / 2, target_len)
    mag_db = np.interp(tgt_freqs, src_freqs, prof_db)
    mag_db -= mag_db.max()
    mag_db = np.clip(mag_db, -60.0, max_gain_db)
    mag = 10.0 ** (mag_db / 20.0)
    # Minimum-phase reconstruction: use Hilbert envelope on log-magnitude
    log_mag = np.log(np.maximum(mag, 1e-9))
    # Mirror to full spectrum, compute analytic signal log, take imag = phase
    full_log = np.concatenate([log_mag, log_mag[-2:0:-1]])
    cep = np.real(np.fft.ifft(full_log))
    n_full = len(cep)
    # Window for minimum-phase: keep first half, zero second
    win = np.zeros(n_full)
    win[0] = 1
    win[1: n_full // 2] = 2
    if n_full % 2 == 0:
        win[n_full // 2] = 1
    min_phase_cep = cep * win
    min_phase_log = np.fft.fft(min_phase_cep)
    H = np.exp(min_phase_log)[:target_len]
    ir = np.fft.irfft(H, n=n_taps)
    # Normalise peak amplitude
    pk = np.max(np.abs(ir)) + 1e-12
    return ir / pk * 0.95


# ---------------------------------------------------------------------------
def per_note_coupling(spectra: dict, modes: list, *,
                       smooth_window: int = 5) -> dict:
    """
    For every (midi) observed in the sample set, regress that note's
    average log-magnitude onto a Lorentzian basis built from `modes`.

    Returns dict midi -> list of mode coupling amplitudes (length len(modes)).
    """
    freqs = spectra['freqs']
    # Build basis: each mode is a Lorentzian in dB centred at f_k with width f_k/Q_k
    n_modes = len(modes)
    basis = np.zeros((n_modes, len(freqs)))
    for k, (f, Q, _) in enumerate(modes):
        bw = max(1.0, f / max(Q, 1.0))
        # Lorentzian magnitude squared response, in dB
        H2 = 1.0 / (1.0 + ((freqs - f) / (bw / 2)) ** 2)
        basis[k] = 10 * np.log10(H2 + 1e-9)
    # Group rows by midi (avg across velocities/variants)
    by_midi: dict[int, list[int]] = {}
    for idx, m in enumerate(spectra['meta']):
        by_midi.setdefault(m['midi'], []).append(idx)
    coupling = {}
    A = basis.T  # (F, n_modes)
    # Non-negative least squares would be cleaner; use unconstrained lstsq
    for midi, idxs in by_midi.items():
        y = spectra['log_mag'][idxs].mean(axis=0)
        # Subtract smooth background so we fit only the resonant peaks
        bg = np.zeros_like(y)
        for i in range(len(y)):
            lo = max(0, i - 64); hi = min(len(y), i + 64)
            bg[i] = np.median(y[lo:hi])
        y_det = y - bg
        sol, *_ = np.linalg.lstsq(A, y_det, rcond=None)
        coupling[midi] = sol.tolist()
    return coupling


# ---------------------------------------------------------------------------
def save_artifacts(spectra, prof, modes, coupling, ir, out_json: str, out_ir: str):
    payload = {
        'sr': spectra['sr'],
        'nfft': NFFT,
        'win_length_s': WIN_LENGTH_S,
        'win_after_attack_s': WIN_AFTER_ATTACK_S,
        'profile_freqs': spectra['freqs'].tolist(),
        'profile_db': prof.tolist(),
        'modes': [{'f': f, 'Q': Q, 'gain_db': g} for (f, Q, g) in modes],
        'coupling_per_midi': {str(k): v for k, v in coupling.items()},
        'ir_path': os.path.basename(out_ir),
    }
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=1)
    np.save(out_ir, ir.astype(np.float32))


def plot_summary(spectra, prof, modes, coupling, out_dir: str):
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)

    freqs = spectra['freqs']
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.semilogx(freqs, prof, color='steelblue', lw=1, label='avg log-mag (all notes)')
    for f, Q, g in modes:
        ax.axvline(f, color='red', alpha=0.3, lw=0.7)
    ax.set_xlim(40, 8000)
    ax.set_ylim(prof.max() - 60, prof.max() + 2)
    ax.set_xlabel('Hz')
    ax.set_ylabel('dB')
    ax.set_title(f'Soundboard backbone (mean of {len(spectra["meta"])} samples) — '
                 f'{len(modes)} modes picked')
    ax.legend(loc='lower left')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'soundboard_profile.png'), dpi=120)
    plt.close(fig)

    # Per-note coupling matrix as image
    midis = sorted(coupling.keys())
    M = np.array([coupling[m] for m in midis])
    fig, ax = plt.subplots(figsize=(11, 5))
    im = ax.imshow(M.T, aspect='auto', cmap='magma', origin='lower',
                   extent=[midis[0], midis[-1], 0, len(modes)])
    ax.set_xlabel('MIDI note')
    ax.set_ylabel('mode index')
    ax.set_title('Per-note × per-mode coupling (regressed from data)')
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'soundboard_coupling.png'), dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    samples_dir = os.path.join(_ROOT, 'oggSamples')
    print(f'Scanning {samples_dir}...')
    spectra = collect_spectra(samples_dir)
    print(f'  loaded {len(spectra["meta"])} samples')
    prof = averaged_profile(spectra)
    print(f'  profile freq range: {spectra["freqs"][0]:.1f} - {spectra["freqs"][-1]:.1f} Hz')
    modes = pick_peaks(spectra['freqs'], prof, n_max=40, prominence_db=1.2)
    print(f'  picked {len(modes)} modes')
    for f, Q, g in modes[:10]:
        print(f'    f={f:7.1f} Hz  Q={Q:5.1f}  gain={g:+.1f} dB')
    if len(modes) > 10:
        print(f'    ... ({len(modes) - 10} more)')
    coupling = per_note_coupling(spectra, modes)
    print(f'  coupling computed for {len(coupling)} midi notes')
    ir = build_fir(spectra['freqs'], prof, n_taps=4096)
    print(f'  built {len(ir)}-tap FIR  (peak {np.max(np.abs(ir)):.3f})')

    out_json = os.path.join(_HERE, 'soundboard_ir.json')
    out_ir = os.path.join(_HERE, 'soundboard_ir.npy')
    save_artifacts(spectra, prof, modes, coupling, ir, out_json, out_ir)
    print(f'  wrote {out_json}\n        {out_ir}')

    try:
        plot_summary(spectra, prof, modes, coupling, os.path.join(_ROOT, 'out'))
        print('  wrote out/soundboard_profile.png')
        print('  wrote out/soundboard_coupling.png')
    except Exception as e:
        print(f'  (plotting skipped: {e})')
