"""Shared helpers for the performance renders from MAESTRO MIDI (SoundFont and SFZ sample-library play).

MIDI source: MAESTRO v3.0.0 MIDI (local copy, audio/maestro/maestro-v3.0.0/, 1276 files,
CC BY-NC-SA 4.0). Only the .midi files are used; MAESTRO audio is not present locally.

Excerpt plan (deterministic, MASTER_SEED):
  * performances with duration >= MIN_PERF_S, visited in a seeded random order;
  * at most ONE excerpt per canonical_title (so no composition is shared between
    SoundFont groups, nor between SoundFonts and Otey);
  * per accepted slot k: rng = default_rng(MASTER_SEED + 1000 + k); draw a start s ~ U(10, dur-70);
    the excerpt starts at the first note onset >= s and lasts exactly EXCERPT_S seconds;
  * a draw is rejected if fewer than MIN_NOTES onsets fall in the window (up to 5 draws,
    then the performance is skipped);
  * slots 0..99 -> 10 SoundFonts x 10 (assignment by a seeded permutation), slots 100..119 -> Otey.

Sustain pedal (CC64): value >= 64 = down (standard threshold). Half-pedalling, CC66 (sostenuto)
and CC67 (soft) are ignored. Notes whose onset precedes the window are not rendered; the
pedal state at the window start is taken from the last CC64 event before it.
"""
from __future__ import annotations
import os
import csv
import bisect
from dataclasses import dataclass, field

import numpy as np

from detector.paths import DATA, AUDIO_ROOT   # noqa: E402
ROOT = AUDIO_ROOT     # audio and the MAESTRO MIDI are named relative to this folder (SPD_AUDIO)
MAESTRO = os.path.join(ROOT, 'audio', 'maestro', 'maestro-v3.0.0')
MAESTRO_CSV = os.path.join(MAESTRO, 'maestro-v3.0.0.csv')
MAESTRO_CLIPS_MANIFEST = os.path.join(ROOT, 'audio', 'maestro_clips_manifest.csv')
DATA_V2 = os.path.join(DATA, 'performances')
PLAN_CSV = os.path.join(DATA_V2, 'perf_excerpt_plan.csv')
AUDIO_OUT = os.path.join(ROOT, 'audio', 'sf2_performances_v2')

MASTER_SEED = 20261001
EXCERPT_S = 60.0
MIN_PERF_S = 120.0
MIN_NOTES = 60
MAX_DRAWS = 5
N_SF2_SLOTS = 100
N_OTEY_SLOTS = 20
SR = 44100
PEDAL_THRESH = 64

# SoundFont group keys exactly as in the released synth_features.csv (instrument column),
# derived by corpus/render_soundfont_notes.py from the .sf2 basename.
SF2_FILES = {
    'CP80_16_44100': 'CP80 16_44100.sf2',
    'CTK_230_SoundFont': 'CTK-230_SoundFont.sf2',
    'CTK_533_Piano1_00_Casio': 'CTK-533_Piano1__00___Casio_.sf2',
    'Casio_CA_110_Piano_FM': 'Casio_CA-110_Piano__FM_.sf2',
    'Casio_PX_860_Concert_Grand_Piano': 'Casio PX-860 Concert Grand Piano.sf2',
    'JV1080_Nice_Piano_m': 'JV1080_Nice_Piano_m.sf2',
    'U20PIANO': 'U20PIANO.sf2',
    'YAMAHA_PSR400_500': 'YAMAHA_PSR400-500.sf2',
    'Yamaha_PSRF50': 'Yamaha PSRF50.sf2',
    'Yamaha_S90_ES_Natural_Grand': 'Yamaha S90 ES Natural Grand.sf2',
}
SF2_DIR = os.path.join(ROOT, 'audio', 'SynthPianoSamples')


@dataclass
class Note:
    onset: float
    offset: float
    pitch: int
    vel: int


@dataclass
class Perf:
    notes: list = field(default_factory=list)       # list[Note], sorted by onset
    pedal_t: list = field(default_factory=list)     # CC64 event times (sorted)
    pedal_v: list = field(default_factory=list)     # CC64 values
    end: float = 0.0


def load_midi(path: str) -> Perf:
    """Parse a MIDI file into notes (FIFO pairing per pitch) + CC64 events, times in s."""
    import mido
    mf = mido.MidiFile(path)
    t = 0.0
    open_: dict[int, list] = {}
    notes, pt, pv = [], [], []
    for msg in mf:                      # merged tracks, msg.time = delta seconds (tempo-aware)
        t += msg.time
        if msg.type == 'note_on' and msg.velocity > 0:
            open_.setdefault(msg.note, []).append((t, msg.velocity))
        elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
            st = open_.get(msg.note)
            if st:
                on, v = st.pop(0)
                notes.append(Note(on, t, msg.note, v))
        elif msg.type == 'control_change' and msg.control == 64:
            pt.append(t)
            pv.append(msg.value)
    for p, st in open_.items():         # unterminated notes end at file end
        for on, v in st:
            notes.append(Note(on, t, p, v))
    notes.sort(key=lambda n: (n.onset, n.pitch))
    return Perf(notes=notes, pedal_t=pt, pedal_v=pv, end=t)


def pedal_down_at(perf: Perf, t: float) -> bool:
    """Pedal state just before time t (last CC64 strictly before t)."""
    i = bisect.bisect_left(perf.pedal_t, t) - 1
    return i >= 0 and perf.pedal_v[i] >= PEDAL_THRESH


def pedal_down_at_incl(perf: Perf, t: float) -> bool:
    """Pedal state at time t including events at exactly t."""
    i = bisect.bisect_right(perf.pedal_t, t) - 1
    return i >= 0 and perf.pedal_v[i] >= PEDAL_THRESH


def next_pedal_up(perf: Perf, t: float) -> float:
    """First time >= t at which the pedal goes from down to up (inf if never)."""
    i = bisect.bisect_left(perf.pedal_t, t)
    down = pedal_down_at(perf, t)
    while i < len(perf.pedal_t):
        d = perf.pedal_v[i] >= PEDAL_THRESH
        if down and not d:
            return perf.pedal_t[i]
        down = d
        i += 1
    return float('inf')


def window_notes(perf: Perf, t0: float, t1: float) -> list:
    return [n for n in perf.notes if t0 <= n.onset < t1]


def pedal_events_in(perf: Perf, t0: float, t1: float):
    return [(t, v) for t, v in zip(perf.pedal_t, perf.pedal_v) if t0 <= t < t1]


def frac_pedal_down(perf: Perf, t0: float, t1: float, step: float = 0.01) -> float:
    ts = np.arange(t0, t1, step)
    return float(np.mean([pedal_down_at_incl(perf, t) for t in ts]))


def effective_releases(perf: Perf, notes: list) -> list:
    """Damper-engage time per note (absolute s), honouring CC64 and same-key re-strikes.

    own release = key-up if pedal up at key-up, else the next pedal-up after key-up;
    if the same pitch is struck again before that, the string stays undamped until the
    re-struck note's release (linear superposition on the same string)."""
    rel = []
    for n in notes:
        r = n.offset
        if pedal_down_at_incl(perf, n.offset):
            r = max(n.offset, next_pedal_up(perf, n.offset))
        rel.append(r)
    by_pitch: dict[int, list[int]] = {}
    for i, n in enumerate(notes):
        by_pitch.setdefault(n.pitch, []).append(i)
    for idx in by_pitch.values():
        idx.sort(key=lambda i: notes[i].onset)
        for a, b in zip(reversed(idx[:-1]), reversed(idx[1:])):
            if notes[b].onset < rel[a]:
                rel[a] = max(rel[a], rel[b])
    return rel


def read_plan() -> list[dict]:
    with open(PLAN_CSV, encoding='utf-8') as f:
        return list(csv.DictReader(f))
