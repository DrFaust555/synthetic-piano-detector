"""Full piano instrument — multi-note rendering with mixing."""

import numpy as np
from note_engine import NoteEngine
from params import midi_to_freq


class Piano:
    def __init__(self, sr=44100):
        self.sr = sr

    def render_note(self, midi, vel, dur, pedal=False, release=None):
        eng = NoteEngine(midi, self.sr)
        return eng.render(dur, vel, pedal, release)

    def render_chord(self, notes, vel, dur, pedal=False, release=None):
        n = int(self.sr * dur)
        mix = np.zeros(n)
        for m in notes:
            print(f'  MIDI {m} ({midi_to_freq(m):.1f} Hz)')
            eng = NoteEngine(m, self.sr)
            a = eng.render(dur, vel, pedal, release)
            L = min(len(a), n)
            mix[:L] += a[:L]
        return mix

    def render_sequence(self, events, total_dur, pedal=False):
        n = int(self.sr * total_dur)
        mix = np.zeros(n)
        for e in events:
            m, vel = e['note'], e['velocity']
            start = int(self.sr * e['start'])
            dur = e['duration']
            rel = e.get('release', dur * 0.85)
            print(f'  MIDI {m} vel={vel} @{e["start"]:.2f}s')
            eng = NoteEngine(m, self.sr)
            a = eng.render(dur, vel, pedal, rel)
            end = min(start + len(a), n)
            mix[start:end] += a[:end - start]
        return mix


def normalize(audio, headroom_db=-1.0):
    pk = np.max(np.abs(audio))
    if pk < 1e-15:
        return audio
    return audio * (10.0 ** (headroom_db / 20.0) / pk)


def soft_clip(audio, thresh=0.9):
    return np.tanh(audio / thresh) * thresh
