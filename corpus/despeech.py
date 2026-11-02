"""Remove speech segments from a recording using PANNs sliding-window tagging.

    python corpus/despeech.py <in.wav> <out.wav>

Slides a 1.0 s / 0.5 s-hop window over the file, tags each window with PANNs
(CNN14), and keeps only windows where Piano dominates and Speech is below a
threshold. Kept windows are merged into spans (small gaps bridged, tiny spans
dropped) and concatenated with short crossfades. If almost nothing survives
(e.g. speech over the whole clip) the original is kept and the file is flagged.

Output is 44.1 kHz / 16-bit mono PCM (dataset convention). Returns dict with
kept-fraction and span list.
"""
from __future__ import annotations
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..')))   # repository root
from detector.paths import DATA, RESULTS, AUDIO_ROOT, results_dir   # noqa: E402
OUT = results_dir('corpus')

WIN_S = 1.0          # analysis window
HOP_S = 0.5          # hop
SPEECH_TH = 0.35     # drop window if Speech prob >= this
GUARD_S = 2.0        # also drop this many seconds before/after any speech window
                     # (phrase fragments bleed past the detected window)
MIN_KEEP_S = 8.0     # keep ORIGINAL (flagged) only if less than this many seconds
                     # of clean piano survive (absolute, not fraction)
BRIDGE_S = 0.40      # merge kept spans separated by <= this
MIN_SPAN_S = 0.60    # drop kept spans shorter than this
FADE_S = 0.010       # crossfade/fade at span joins
OUT_SR = 44100


def _spans_from_keep(keep, starts, win, sr_an, sr_out, n_out):
    """keep[] per window -> merged sample spans in output sr."""
    mask = np.zeros(n_out, bool)
    for s, k in zip(starts, keep):
        if k:
            a = int(round(s / sr_an * sr_out))
            b = int(round((s + win) / sr_an * sr_out))
            mask[a:min(b, n_out)] = True
    # contiguous True runs -> spans
    spans = []
    i = 0
    N = len(mask)
    while i < N:
        if mask[i]:
            j = i
            while j < N and mask[j]:
                j += 1
            spans.append([i, j])
            i = j
        else:
            i += 1
    # bridge small gaps
    bridge = int(BRIDGE_S * sr_out)
    merged = []
    for sp in spans:
        if merged and sp[0] - merged[-1][1] <= bridge:
            merged[-1][1] = sp[1]
        else:
            merged.append(sp)
    # drop tiny
    minlen = int(MIN_SPAN_S * sr_out)
    merged = [sp for sp in merged if sp[1] - sp[0] >= minlen]
    return merged


def despeech_array(y, sr, at, lab):
    import librosa
    from corpus.pianoness import PANNS_SR, PIANO_TAGS
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y.astype(np.float64)
    y32 = librosa.resample(y, orig_sr=sr, target_sr=PANNS_SR) if sr != PANNS_SR else y
    w = int(WIN_S * PANNS_SR); h = int(HOP_S * PANNS_SR)
    starts = list(range(0, max(1, len(y32) - w + 1), h)) or [0]
    batch = np.stack([np.pad(y32[s:s + w], (0, max(0, w - len(y32[s:s + w]))))
                      for s in starts]).astype(np.float32)
    sp_i = lab.index('Speech')
    pi = [lab.index(t) for t in PIANO_TAGS if t in lab]
    probs = []
    B = 64
    for i in range(0, len(batch), B):
        clip, _ = at.inference(batch[i:i + B])
        probs.append(clip)
    probs = np.concatenate(probs, 0)
    speech = probs[:, sp_i]
    piano = probs[:, pi].max(axis=1)
    keep = (speech < SPEECH_TH) & (piano >= speech)
    # guard: expand speech windows by GUARD_S each side so phrase fragments at
    # the boundaries are also dropped
    speech_win = speech >= SPEECH_TH
    g = int(round(GUARD_S / HOP_S))
    if g > 0 and speech_win.any():
        dil = speech_win.copy()
        for k in range(1, g + 1):
            dil[k:] |= speech_win[:-k]
            dil[:-k] |= speech_win[k:]
        keep = keep & ~dil
    spans = _spans_from_keep(keep, starts, w, PANNS_SR, sr, len(y))
    kept_frac = sum(b - a for a, b in spans) / max(1, len(y))
    return spans, kept_frac, float(speech.max()), float(np.mean(speech >= SPEECH_TH))


def _concat_spans(y, sr, spans):
    fade = int(FADE_S * sr)
    out = []
    for a, b in spans:
        seg = y[a:b].astype(np.float64).copy()
        if len(seg) > 2 * fade:
            seg[:fade] *= np.linspace(0, 1, fade)
            seg[-fade:] *= np.linspace(1, 0, fade)
        out.append(seg)
    return np.concatenate(out) if out else y.astype(np.float64)


def despeech_file(path, out_path, at, lab):
    import soundfile as sf
    from detector.segment import load_mono
    y, sr = load_mono(path)
    spans, frac, smax, sfrac = despeech_array(y, sr, at, lab)
    kept_sec = sum(b - a for a, b in spans) / sr
    flagged = False
    if not spans or kept_sec < MIN_KEEP_S:   # truly almost nothing clean -> keep original (flagged)
        ys = y.astype(np.float64)
        flagged = True
    else:
        ys = _concat_spans(y, sr, spans)
    # peak-normalize to -1 dBFS (dataset convention)
    pk = np.max(np.abs(ys)) or 1.0
    ys = ys * (10 ** (-1.0 / 20) / pk)
    sf.write(out_path, ys.astype(np.float32), sr, subtype='PCM_16')
    return {'kept_frac': frac, 'speech_max': smax, 'speech_frac': sfrac,
            'n_spans': len(spans), 'flagged': flagged,
            'dur_in': len(y) / sr, 'dur_out': len(ys) / sr}


def main():
    import argparse
    from corpus.pianoness import load_model, _labels, _ensure_assets
    ap = argparse.ArgumentParser()
    ap.add_argument('inp'); ap.add_argument('outp')
    a = ap.parse_args()
    _ensure_assets()
    lab = _labels(); at = load_model()
    r = despeech_file(a.inp, a.outp, at, lab)
    print(f"{os.path.basename(a.inp)} -> {os.path.basename(a.outp)}")
    print(f"  kept={r['kept_frac']*100:.0f}%  spans={r['n_spans']}  "
          f"speech_max={r['speech_max']:.2f}  speech_win_frac={r['speech_frac']*100:.0f}%  "
          f"dur {r['dur_in']:.1f}->{r['dur_out']:.1f}s{'  [FLAGGED keep-orig]' if r['flagged'] else ''}")


if __name__ == '__main__':
    main()
