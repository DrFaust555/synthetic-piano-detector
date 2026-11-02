"""Render the 100 SoundFont performance excerpts of perf_excerpt_plan.csv.

    python corpus/render_soundfont_performances.py

Same synthesizer, preset and output format as corpus/render_soundfont_notes.py (the 762 single-note
renders): python `tinysoundfont` (0.3.7 here; verified bit-identical to all 762 existing renders),
samplerate 44100, bank 0 / preset 0, stereo -> mono by channel mean, peak-normalised to 0.9,
FLOAT WAV. No reverb, no channel simulation, nothing else.

Sustain pedal: TinySoundFont ignores CC64 (verified: identical output with and without CC64=127).
It is therefore emulated in the event layer: while CC64 >= 64 a key release does not send a
note-off; the note-off is sent at pedal-up unless the key is physically down again (the voice
stays in the SoundFont envelope's sustain stage, as in GM synths with native sustain).
All MIDI channels are routed to channel 0; program changes and other controllers are ignored.

Writes <SPD_AUDIO>/audio/sf2_performances_v2/<instrument>/<instrument>_perfNN.wav and
results/corpus/sf2_perf_manifest.csv.
"""
from __future__ import annotations
import os
import sys
import csv
import time
import importlib.metadata as md

import numpy as np
import soundfile as sf
import tinysoundfont

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))   # repository root
from detector.paths import DATA, RESULTS, AUDIO_ROOT, results_dir   # noqa: E402
from corpus.perf_common import (ROOT, MAESTRO, DATA_V2, SR, EXCERPT_S, PEDAL_THRESH, SF2_FILES, SF2_DIR,
                         load_midi, window_notes, pedal_events_in, pedal_down_at, read_plan)

MANIFEST = os.path.join(results_dir('corpus'), 'sf2_perf_manifest.csv')
PEAK = 0.9


def audible_pitches(sf2_path: str) -> set:
    """MIDI pitches 21..108 for which preset 0 makes sound (peak >= 1e-3, vel 100, 0.5 s)."""
    s = tinysoundfont.Synth(samplerate=SR)
    i = s.sfload(sf2_path)
    s.program_select(0, i, 0, 0)
    out = set()
    for p in range(21, 109):
        s.noteon(0, p, 100)
        a = np.frombuffer(s.generate(int(SR * 0.5)), dtype=np.float32)
        s.noteoff(0, p)
        s.sounds_off()
        s.generate(int(SR * 0.05))
        if np.max(np.abs(a)) >= 1e-3:
            out.add(p)
    s.sfunload(i)
    return out


def build_events(perf, t0, t1):
    """(time, priority, kind, pitch, vel); priority: off 0 < pedal 1 < on 2 at equal times."""
    ev = []
    for n in window_notes(perf, t0, t1):
        ev.append((n.onset, 2, 'on', n.pitch, n.vel))
        if n.offset < t1:
            ev.append((n.offset, 0, 'off', n.pitch, 0))
    for t, v in pedal_events_in(perf, t0, t1):
        ev.append((t, 1, 'ped', 0, v))
    ev.sort(key=lambda e: (e[0], e[1]))
    return ev


def render_excerpt(sf2_path, perf, t0, t1):
    synth = tinysoundfont.Synth(samplerate=SR)
    sfid = synth.sfload(sf2_path)
    synth.program_select(0, sfid, 0, 0)
    N = int(round(EXCERPT_S * SR))
    buf = np.zeros((N, 2), dtype=np.float32)
    pos = 0
    held: dict[int, int] = {}
    sustained: set = set()
    ped = pedal_down_at(perf, t0)
    n_deferred = 0
    for t, _pr, kind, p, v in build_events(perf, t0, t1):
        s = min(max(int(round((t - t0) * SR)), pos), N)
        if s > pos:
            buf[pos:s] = np.frombuffer(synth.generate(s - pos), dtype=np.float32).reshape(-1, 2)
            pos = s
        if kind == 'on':
            synth.noteon(0, p, v)
            held[p] = held.get(p, 0) + 1
            sustained.discard(p)
        elif kind == 'off':
            if held.get(p, 0) > 0:
                held[p] -= 1
            if held.get(p, 0) == 0:
                if ped:
                    sustained.add(p)
                    n_deferred += 1
                else:
                    synth.noteoff(0, p)
        else:
            down = v >= PEDAL_THRESH
            if ped and not down:
                for q in sorted(sustained):
                    if held.get(q, 0) == 0:
                        synth.noteoff(0, q)
                sustained.clear()
            ped = down
    if pos < N:
        buf[pos:N] = np.frombuffer(synth.generate(N - pos), dtype=np.float32).reshape(-1, 2)
    synth.sfunload(sfid)
    return buf.mean(axis=1), n_deferred


def main():
    plan = [r for r in read_plan() if r['engine'] == 'tinysoundfont']
    ver = md.version('tinysoundfont')
    audible = {}
    names = {}
    for inst, fn in SF2_FILES.items():
        path = os.path.join(SF2_DIR, fn)
        audible[inst] = audible_pitches(path)
        s = tinysoundfont.Synth(samplerate=SR)
        i = s.sfload(path)
        names[inst] = s.sfpreset_name(i, 0, 0)
        print(f"  {inst:34s} preset0='{names[inst]}' audible pitches "
              f"{min(audible[inst])}-{max(audible[inst])} (n={len(audible[inst])})", flush=True)
    rows = []
    t_start = time.time()
    for r in plan:
        inst = r['instrument']
        sf2_path = os.path.join(SF2_DIR, SF2_FILES[inst])
        perf = load_midi(os.path.join(MAESTRO, r['maestro_midi_filename'].replace('/', os.sep)))
        t0, t1 = float(r['excerpt_start_s']), float(r['excerpt_end_s'])
        wn = window_notes(perf, t0, t1)
        tt = time.time()
        mono, n_def = render_excerpt(sf2_path, perf, t0, t1)
        pk = float(np.max(np.abs(mono)))
        if pk < 1e-3:
            print(f"  [silent] {r['file']}")
            continue
        out = (mono / pk) * PEAK
        dst = os.path.join(ROOT, r['file'].replace('/', os.sep))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        sf.write(dst, out.astype(np.float32), SR, subtype='FLOAT')
        n_inaud = sum(1 for n in wn if n.pitch not in audible[inst])
        rows.append({
            'file': r['file'], 'synth_type': 'rendered_perf', 'instrument': inst,
            'take': r['take'], 'sf2_file': SF2_FILES[inst], 'bank': 0, 'preset': 0,
            'preset_name': names[inst], 'engine': f'tinysoundfont {ver}', 'sr': SR,
            'channels': 1, 'subtype': 'FLOAT', 'duration_s': round(len(out) / SR, 4),
            'normalisation': 'mono=mean(L,R); peak->0.9', 'peak_before_norm': round(pk, 6),
            'midi_source': 'MAESTRO v3.0.0 MIDI (CC BY-NC-SA 4.0)',
            'maestro_midi_filename': r['maestro_midi_filename'],
            'canonical_composer': r['canonical_composer'], 'canonical_title': r['canonical_title'],
            'year': r['year'], 'maestro_split': r['maestro_split'],
            'excerpt_start_s': r['excerpt_start_s'], 'excerpt_end_s': r['excerpt_end_s'],
            'seed': r['seed'], 'attempt': r['attempt'], 'draw_start_s': r['draw_start_s'],
            'n_notes': len(wn), 'n_notes_outside_sf2_range': n_inaud,
            'pedal_down_at_start': r['pedal_down_at_start'], 'n_cc64_events': r['n_cc64_events'],
            'frac_time_pedal_down': r['frac_time_pedal_down'],
            'n_noteoffs_deferred_by_pedal': n_def,
            'perf_also_in_maestro_clips': r['perf_also_in_maestro_clips'],
        })
        print(f"  {r['file']:70s} notes={len(wn):4d} deferred={n_def:4d} "
              f"outside={n_inaud:3d} pk={pk:.3f} {time.time() - tt:5.1f}s", flush=True)
    with open(MANIFEST, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n{len(rows)} files in {time.time() - t_start:.0f}s -> {MANIFEST}")


if __name__ == '__main__':
    main()
