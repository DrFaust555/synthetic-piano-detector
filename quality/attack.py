"""Attack-transient descriptor: how percussive is the note onset.

The three trained listeners flagged the ATTACK (hammer knock) as the model's #1
realism gap.  This measures it directly: HPSS-percussive energy as a fraction of
total energy in the first `win_ms` after each note onset, aggregated over the file.

Real pianos have a broadband knock at the strike (high percussive fraction); the
Otey model's onset is almost pure spun-up tone (low fraction) — so real ≫ model.
Adding the commuted-synthesis knock (inference.knock) should close that gap.

    python -m quality.attack <wav> [<wav>...]

Advisory diagnostic only — NOT one of the 47 frozen features, NOT a term in the
tuning objective J (rule #4) until the ears confirm it.
"""

from __future__ import annotations
import os
import sys
import glob
import numpy as np

from detune.segment import load_mono, detect_onsets

WIN_MS = 50.0          # attack window measured from onset
PRE_MS = 5.0           # HPSS analysis pre-roll before onset
ANA_MS = 260.0         # HPSS analysis window (needs a few STFT frames)
HPSS_MARGIN = 1.0      # default split; higher margins over-strip the transient
                       # (empirically real<model above ~2.5) — 1.0 keeps real>model
BACK_MS = 80.0         # how far before a spectral-flux onset to hunt the true rise
THRESH = 0.02          # amplitude-onset threshold (fraction of local peak)


def _amp_onset_near(x: np.ndarray, o: int, sr: int) -> int:
    """Refine a spectral-flux onset `o` to the true AMPLITUDE rise (first sound).

    Spectral flux peaks ~40-50 ms AFTER the strike (once the tone is established),
    which would place the attack window past the knock entirely.  The knock lives at
    the amplitude onset, so we hunt backward to the first crossing of THRESH·peak."""
    back = int(BACK_MS * sr / 1000.0)
    fwd = int(WIN_MS * sr / 1000.0)
    lo = max(0, int(o) - back)
    r = x[lo: int(o) + fwd]
    if len(r) == 0:
        return int(o)
    pk = float(np.max(np.abs(r)))
    if pk <= 0:
        return int(o)
    idx = np.nonzero(np.abs(r) >= THRESH * pk)[0]
    return lo + int(idx[0]) if len(idx) else int(o)


def attack_percussive_ratio(x: np.ndarray, sr: int, win_ms: float = WIN_MS,
                            margin: float = HPSS_MARGIN) -> float:
    """Median over notes of (percussive energy / total energy) in the attack window.

    ~0 = pure tone onset (Otey model), higher = broadband knock (real piano).  The
    window is anchored at the AMPLITUDE onset, not the lagging spectral-flux onset —
    that is where the hammer knock sits (real ≫ model ~6x at C4 v127)."""
    import librosa
    x = np.asarray(x, dtype=np.float32)
    if len(x) < int(0.08 * sr):
        return float("nan")
    win = int(win_ms * sr / 1000.0)
    pre = int(PRE_MS * sr / 1000.0)
    ana = int(ANA_MS * sr / 1000.0)
    gpk = float(np.max(np.abs(x))) + 1e-20
    # Refine each spectral-flux onset to its amplitude rise, then keep only genuine
    # STRIKES: the attack must reach a sizeable fraction of the global peak.  This
    # drops the spurious mid-decay flux peaks that over-segment a single held note.
    cand = sorted(set(_amp_onset_near(x, int(o), sr) for o in detect_onsets(x, sr)))
    onsets = []
    for on in cand:
        if float(np.max(np.abs(x[on: on + win]))) < 0.25 * gpk:
            continue
        if onsets and on - onsets[-1] < win:      # merge near-duplicates
            continue
        onsets.append(on)
    if not onsets:
        onsets = cand[:1]
    ratios = []
    for on in onsets:
        lo = max(0, int(on) - pre)
        seg = x[lo: int(on) + ana]
        if len(seg) < 2048:
            continue
        _, perc = librosa.effects.hpss(seg.astype(np.float32), margin=margin)
        a = int(on) - lo
        pe = float(np.sum(perc[a: a + win] ** 2))
        te = float(np.sum(seg[a: a + win] ** 2)) + 1e-20
        ratios.append(pe / te)
    if not ratios:
        return float("nan")
    return float(np.median(ratios))


def main():
    files = []
    for a in sys.argv[1:]:
        files += sorted(glob.glob(a)) if any(c in a for c in "*?") else [a]
    if not files:
        print("Вкажи WAV/OGG файл(и).")
        return
    for f in files:
        try:
            x, sr = load_mono(f)
            r = attack_percussive_ratio(x, sr)
            print(f"{os.path.basename(f):40s} attack_percussive_ratio = {r:.4f}")
        except Exception as e:
            print(f"{os.path.basename(f):40s} ERR {e}")


if __name__ == "__main__":
    main()
