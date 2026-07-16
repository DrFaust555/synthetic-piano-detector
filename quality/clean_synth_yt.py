"""Clean the YouTube synth-piano demos: keep ONLY piano-dominant audio, drop speech,
drums and other non-piano sounds, with a ±MARGIN-second guard around anything rejected.

    python -m quality.clean_synth_yt [folder]   (default audio/SynthPianoSamples/yt)

PANNs over 1-s windows: a window is REJECTED if speech/drums/other beat piano (or piano
is weak).  The reject mask is then dilated by ±MARGIN s, so kept piano never sits within
MARGIN s of any rejected content.  Kept windows are glued with short fades to
<folder>/clean/<name>.wav (44.1 kHz).  Files with < MIN_KEEP_S of piano are skipped.
"""

from __future__ import annotations
import os
import sys
import numpy as np
import librosa
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT = os.path.join(HERE, '..', 'audio', 'SynthPianoSamples', 'yt')
SR_OUT = 44100
SR_PANNS = 32000
WIN = 1.0
MARGIN_S = 2.0                 # ±2 s guard around any rejected window
MIN_KEEP_S = 8.0
SPEECH_THRESH = 0.10           # ANY speech above this → reject (talk-over-piano too)

PIANO = ['Piano', 'Electric piano', 'Keyboard (musical)']
SPEECH = ['Speech', 'Male speech', 'Female speech', 'Conversation', 'Narration',
          'Monologue', 'Speech synthesizer', 'Child speech']
OTHER = ['Drum', 'Drum kit', 'Percussion', 'Snare drum', 'Bass drum', 'Cymbal',
         'Hi-hat', 'Singing', 'Guitar', 'Applause', 'Laughter', 'Crowd']


def _idx(labels, keys):
    return [i for i, l in enumerate(labels) if any(k.lower() in l.lower() for k in keys)]


def main():
    from panns_inference import AudioTagging, labels
    folder = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    files = [os.path.join(folder, f) for f in sorted(os.listdir(folder))
             if f.lower().endswith('.wav')]
    out_dir = os.path.join(folder, 'clean')
    os.makedirs(out_dir, exist_ok=True)
    pi = _idx(labels, PIANO)
    spi = _idx(labels, SPEECH)
    oi = _idx(labels, OTHER)
    try:
        import torch
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    except Exception:
        dev = 'cpu'
    at = AudioTagging(checkpoint_path=None, device=dev)
    margin = int(round(MARGIN_S / WIN))
    fade = np.linspace(0, 1, int(0.006 * SR_OUT))

    for f in files:
        y44, _ = librosa.load(f, sr=SR_OUT, mono=True)
        y32 = librosa.resample(y44, orig_sr=SR_OUT, target_sr=SR_PANNS)
        w32, w44 = int(WIN * SR_PANNS), int(WIN * SR_OUT)
        n = min(len(y32) // w32, len(y44) // w44)
        if n < 4:
            print(f"  {os.path.basename(f)[:40]:40s} закороткий"); continue
        wins = y32[:n * w32].reshape(n, w32)
        clip, _ = at.inference(wins)
        pn = clip[:, pi].max(1)
        speech = clip[:, spi].max(1)
        other = clip[:, oi].max(1)
        # KEEP only clean piano: piano present, NO speech (hard, catches talk-over),
        # no drums/other, and piano louder than any other non-piano sound.
        raw_keep = (pn > 0.10) & (speech < SPEECH_THRESH) & (other < 0.35) & (pn > other)
        # dilate the REJECT mask by ±margin → ±MARGIN s guard around any bad window
        reject = ~raw_keep
        rej_dil = reject.copy()
        for i in np.where(reject)[0]:
            rej_dil[max(0, i - margin):i + margin + 1] = True
        keep = ~rej_dil

        kept = []
        for i in np.where(keep)[0]:
            c = y44[i * w44:(i + 1) * w44].copy()
            c[:len(fade)] *= fade
            c[-len(fade):] *= fade[::-1]
            kept.append(c)
        sec = len(kept) * WIN
        print(f"  {os.path.basename(f)[:40]:40s} піано {sec:4.0f}с / {n*WIN:4.0f}с "
              f"({100*len(kept)//max(n,1):3d}%)", end='  ')
        if sec < MIN_KEEP_S:
            print("→ замало піано, пропуск"); continue
        out = np.concatenate(kept)
        dst = os.path.join(out_dir, os.path.splitext(os.path.basename(f))[0] + '.wav')
        sf.write(dst, out, SR_OUT, subtype='FLOAT')
        print(f"→ clean/{os.path.basename(dst)[:30]}")


if __name__ == '__main__':
    main()
