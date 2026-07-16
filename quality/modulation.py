"""Per-band modulation descriptors — capture the LOW-FREQUENCY "life" of a real piano.

Hypothesis (user): real pianos carry rich low-frequency modulation in the BEAT band
(~1-12 Hz: unison beating, soundboard coupling) that is INCOHERENT (many beats, differs
per register) and does NOT collapse into a single coherent <1 Hz sweep.  Synthetic is
either dead (no modulation) or a coherent phaser (<1 Hz peak, same across bands).

For 8 log-frequency bands we compute, over the whole file (STFT 2048/512):
  AM(t) = band energy over time            (amplitude modulation → beating)
  FM(t) = intensity-weighted band frequency over time, in cents  (frequency modulation →
          two detuned unison strings make the partial's apparent pitch wobble)
and their modulation spectra, split into SLOW (phaser/wander), BEAT (life), ROUGH bands.

`modulation_features(x, sr)` → flat dict (summary, usable as discriminator features):
  beat_mod        AM+FM energy in 1-12 Hz, mean over bands   (real: high; dead synth: low)
  beat_flatness   spectral flatness of beat-band AM          (real: ~flat/incoherent; phaser/single-beat: low)
  fm_wander       median per-band inst-frequency std (cents) (real strings wander; synth: ~0)
  slow_coherent   <1 Hz modulation × cross-band coherence    (phaser: high; real: low)
  beat_to_slow    beat_mod / (slow modulation)               (real: high; phaser: low)
"""

from __future__ import annotations
import numpy as np
from numpy.fft import rfft, rfftfreq

N_BANDS = 8
FMIN, FMAX = 100.0, 8000.0
WIN, HOP = 2048, 512
SLOW = (0.2, 1.0)        # phaser / slow wander
BEAT = (1.0, 12.0)       # unison beating / soundboard — the "life"
ROUGH = (12.0, 35.0)


def _band_frac(F, mf, lo, hi):
    sel = (mf >= lo) & (mf <= hi)
    tot = F[(mf >= SLOW[0])].sum() + 1e-12
    return float(F[sel].sum() / tot)


def _flatness(F, mf, lo, hi):
    v = F[(mf >= lo) & (mf <= hi)] + 1e-12
    if len(v) < 3:
        return 0.0
    return float(np.exp(np.mean(np.log(v))) / np.mean(v))     # 0 peaky … 1 flat


def modulation_features(x: np.ndarray, sr: int) -> dict:
    x = np.asarray(x, dtype=float)
    if x.ndim > 1:
        x = x.mean(axis=1)
    nfr = 1 + (len(x) - WIN) // HOP if len(x) >= WIN else 0
    if nfr < 48:
        return {k: 0.0 for k in ('beat_mod', 'beat_flatness', 'fm_wander',
                                 'slow_coherent', 'beat_to_slow')}
    w = np.hanning(WIN)
    freqs = rfftfreq(WIN, 1.0 / sr)
    edges = np.geomspace(FMIN, min(FMAX, sr * 0.475), N_BANDS + 1)
    masks = [(freqs >= edges[b]) & (freqs < edges[b + 1]) for b in range(N_BANDS)]
    fb = [freqs[m] for m in masks]

    AM = np.empty((nfr, N_BANDS))
    FM = np.empty((nfr, N_BANDS))
    for i in range(nfr):
        p = np.abs(rfft(x[i * HOP:i * HOP + WIN] * w)) ** 2
        for b, m in enumerate(masks):
            pe = p[m]
            s = pe.sum()
            AM[i, b] = 10.0 * np.log10(s + 1e-9)
            FM[i, b] = (fb[b] @ pe) / (s + 1e-12) if s > 0 else 0.0

    lvl = AM.mean(axis=1)
    keep = lvl > (lvl.max() - 40.0)
    if keep.sum() < 48:
        return {k: 0.0 for k in ('beat_mod', 'beat_flatness', 'fm_wander',
                                 'slow_coherent', 'beat_to_slow')}
    AM, FM = AM[keep], FM[keep]
    nfr = AM.shape[0]
    fps = sr / HOP
    han = np.hanning(nfr)[:, None]
    mf = rfftfreq(nfr, 1.0 / fps)

    # FM → cents relative to each band's median frequency
    fmed = np.median(FM, axis=0, keepdims=True) + 1e-9
    FMc = 1200.0 * np.log2(np.clip(FM / fmed, 1e-3, 1e3))

    Fam = np.abs(rfft((AM - AM.mean(0)) * han, axis=0))       # (modfreq, band)
    Ffm = np.abs(rfft((FMc - FMc.mean(0)) * han, axis=0))

    beat, slow, flat, fmw = [], [], [], []
    slow_peakfreq = []
    for b in range(N_BANDS):
        beat.append(_band_frac(Fam[:, b], mf, *BEAT) + _band_frac(Ffm[:, b], mf, *BEAT))
        slow.append(_band_frac(Fam[:, b], mf, *SLOW))
        flat.append(_flatness(Fam[:, b], mf, *BEAT))
        fmw.append(float(np.std(FMc[:, b])))
        ssel = (mf >= SLOW[0]) & (mf <= SLOW[1])
        slow_peakfreq.append(mf[ssel][Fam[ssel, b].argmax()] if ssel.sum() else 0.0)

    # cross-band coherence of the slow peak (phaser → same slow freq in all bands)
    spk = np.array(slow_peakfreq)
    coh = float(np.mean(np.abs(spk - np.median(spk)) < 0.1)) if len(spk) else 0.0
    beat_mod = float(np.mean(beat))
    slow_mod = float(np.mean(slow))
    return {
        'beat_mod': beat_mod,
        'beat_flatness': float(np.mean(flat)),
        'fm_wander': float(np.median(fmw)),
        'slow_coherent': slow_mod * coh,
        'beat_to_slow': float(beat_mod / (slow_mod + 1e-6)),
    }


if __name__ == '__main__':
    import sys, os, glob
    from detune.segment import load_mono
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    files = []
    for a in sys.argv[1:]:
        p = a if os.path.isabs(a) else os.path.join(ROOT, a)
        files += sorted(glob.glob(os.path.join(p, '*'))) if os.path.isdir(p) else sorted(glob.glob(p))
    keys = ['beat_mod', 'beat_flatness', 'fm_wander', 'slow_coherent', 'beat_to_slow']
    print(f"{'file':32s}" + ''.join(f'{k:>14s}' for k in keys))
    for f in files:
        try:
            x, sr = load_mono(f)
            m = modulation_features(x, sr)
            print(f"{os.path.basename(f)[:32]:32s}" + ''.join(f'{m[k]:14.3f}' for k in keys))
        except Exception as e:
            print(f"{os.path.basename(f)[:32]:32s} ERR {e}")
