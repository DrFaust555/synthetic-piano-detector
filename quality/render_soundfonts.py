"""Render the SoundFont (.sf2) synth pianos to per-note WAVs — the synthetic-negative
class for the real-vs-synth critic.

    python -m quality.render_soundfonts

Each .sf2 in audio/SynthPianoSamples/ (zips auto-extracted) is loaded once and played
over a note x velocity grid; each note is written peak-normalised FLOAT to
audio/SynthPianoSamples/rendered/<instrument>/<NOTE>_v<vel>.wav.
"""

from __future__ import annotations
import os
import glob
import zipfile
import re
import numpy as np
import soundfile as sf
import tinysoundfont

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, 'audio', 'SynthPianoSamples')
OUT = os.path.join(SRC, 'rendered')
SR = 44100
NOTES = list(range(21, 97, 3))                 # A0..C7 every 3 semitones
VELS = [64, 100, 127]
NAME = {21: 'A0'}  # filled below
_PC = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
def note_name(m): return f"{_PC[m % 12]}{m // 12 - 1}".replace('#', 's')


def _extract_zips():
    for z in glob.glob(os.path.join(SRC, '*.zip')):
        try:
            with zipfile.ZipFile(z) as zf:
                for n in zf.namelist():
                    if n.lower().endswith('.sf2'):
                        zf.extract(n, SRC)
        except Exception as e:
            print(f"  zip {os.path.basename(z)}: {e}")


def _render_note(synth, midi, vel, dur=1.4, rel=0.4):
    synth.noteon(0, midi, vel)
    a1 = np.frombuffer(synth.generate(int(SR * dur)), dtype=np.float32)
    synth.noteoff(0, midi)
    a2 = np.frombuffer(synth.generate(int(SR * rel)), dtype=np.float32)
    a = np.concatenate([a1, a2]).reshape(-1, 2).mean(axis=1)   # stereo→mono
    return a


def main():
    _extract_zips()
    sf2s = sorted(glob.glob(os.path.join(SRC, '**', '*.sf2'), recursive=True))
    sf2s = [s for s in sf2s if os.sep + 'rendered' + os.sep not in s]
    print(f"SoundFonts: {len(sf2s)}")
    total = 0
    for path in sf2s:
        inst = re.sub(r'[^A-Za-z0-9]+', '_', os.path.splitext(os.path.basename(path))[0]).strip('_')
        odir = os.path.join(OUT, inst)
        os.makedirs(odir, exist_ok=True)
        try:
            synth = tinysoundfont.Synth(samplerate=SR)
            sfid = synth.sfload(path)
            synth.program_select(0, sfid, 0, 0)
        except Exception as e:
            print(f"  [load fail] {inst}: {e}")
            continue
        n_inst = 0
        for m in NOTES:
            for v in VELS:
                try:
                    a = _render_note(synth, m, v)
                except Exception:
                    continue
                pk = np.max(np.abs(a))
                if pk < 1e-3:                  # silent (wrong preset/out of range)
                    continue
                a = (a / pk) * 0.9
                sf.write(os.path.join(odir, f"{note_name(m)}_v{v}.wav"), a, SR,
                         subtype='FLOAT')
                n_inst += 1
        synth.sfunload(sfid)
        total += n_inst
        print(f"  {inst:34s} {n_inst:4d} notes")
    print(f"\nTotal rendered: {total} WAVs -> {OUT}")


if __name__ == '__main__':
    main()
