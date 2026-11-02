"""Render the perf_excerpt_plan.csv excerpts through SFZ sample libraries ("sampled play" class).

    python corpus/render_sfz_performances.py events                    # local: MIDI -> event JSON
    python corpus/render_sfz_performances.py render <root> <shard> <n_shards> [<lib> ...]
    python corpus/render_sfz_performances.py manifest                  # local: manifest CSV

Engine: sfizz 1.2.3 (official win64 release, tools/sfizz-1.2.3-win64/sfizz.dll) driven through its
C API with ctypes. Not sfizz_render.exe: that tool writes 16-bit PCM only, so the quantisation
floor would depend on each library's raw level (Iowa raw peak 0.036 -> about -67 dB re peak after
normalisation). The C API returns float blocks; note events are sent with sample-accurate delays.

Everything else as render_soundfont_performances.py: same excerpts, sustain pedal emulated in the event
layer (deferred note-offs, no CC64 sent; also keeps the libraries' pedal-noise regions silent),
mono = mean(L,R), exactly EXCERPT_S, peak -> 0.9, FLOAT WAV, 44.1 kHz. Freewheeling mode,
sample quality 10 (sfizz's highest; resampling only matters for transposed notes), block 1024,
polyphony 256.

The render step needs only numpy + soundfile + the event JSON (no mido), so it runs on the remote
machine; <root> is the directory holding events/, the libraries (paths as LIBS below, relative to
<root>) and receives audio/sfz_performances_v2/<lib>/<lib>_slotNNN.wav + render_log/.
"""
from __future__ import annotations
import os
import sys
import csv
import json
import time
import ctypes

import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..')))   # repository root
from detector.paths import DATA, RESULTS, AUDIO_ROOT, results_dir   # noqa: E402
OUT = results_dir('corpus')
SR = 44100
EXCERPT_S = 60.0
PEAK = 0.9
BLOCK = 1024
QUALITY = 10
POLYPHONY = 256
SFIZZ_PROCESS_FREEWHEELING = 1
SFIZZ_DLL = os.path.join('tools', 'sfizz-1.2.3-win64', 'sfizz.dll')
LIBS = {      # relative to the project root (or to <root> on the remote machine)
    'Salamander': os.path.join('audio', 'ref_bright_yamaha_salamander', 'SalamanderGrandPianoV3_OggVorbis', 'SalamanderGrandPianoV3.sfz'),
    'K18': os.path.join('audio', 'ref_dull_upright', 'K18-Upright-Piano.sfz'),
    'FreePats': os.path.join('audio', 'ref_bad_old_piano', 'PianoFB SFZ+WAV-20200401', 'PianoFB 20200401.sfz'),
    'Iowa': os.path.join('audio', 'ref_top_grand_iowa_steinway', 'IowaSteinwayB.sfz'),   # build_iowa_sfz.py
}
EVENTS_DIR = os.path.join('audio', 'sfz_performances_v2', '_events')
AUDIO_OUT = os.path.join('audio', 'sfz_performances_v2')
LOG_DIR = os.path.join('audio', 'sfz_performances_v2', '_render_log')


class Sfizz:
    def __init__(self, root: str):
        d = ctypes.CDLL(os.path.join(root, SFIZZ_DLL))
        d.sfizz_create_synth.restype = ctypes.c_void_p
        d.sfizz_load_file.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        d.sfizz_load_file.restype = ctypes.c_bool
        d.sfizz_set_sample_rate.argtypes = [ctypes.c_void_p, ctypes.c_float]
        for f in ('sfizz_set_samples_per_block', 'sfizz_set_num_voices'):
            getattr(d, f).argtypes = [ctypes.c_void_p, ctypes.c_int]
        d.sfizz_set_sample_quality.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        d.sfizz_get_sample_quality.argtypes = [ctypes.c_void_p, ctypes.c_int]
        d.sfizz_get_num_regions.argtypes = [ctypes.c_void_p]
        d.sfizz_enable_freewheeling.argtypes = [ctypes.c_void_p]
        for f in ('sfizz_send_note_on', 'sfizz_send_note_off'):
            getattr(d, f).argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        d.sfizz_render_block.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
                                         ctypes.c_int, ctypes.c_int]
        d.sfizz_free.argtypes = [ctypes.c_void_p]
        self.d = d

    def render(self, sfz_path: str, events: list, n_frames: int) -> tuple[np.ndarray, int]:
        """events: [(sample, 'on'|'off', pitch, vel)], sorted. Returns (float32 [n, 2], n_regions)."""
        d = self.d
        s = d.sfizz_create_synth()
        try:
            d.sfizz_set_samples_per_block(s, BLOCK)
            d.sfizz_set_sample_rate(s, float(SR))
            d.sfizz_set_sample_quality(s, SFIZZ_PROCESS_FREEWHEELING, QUALITY)
            d.sfizz_set_num_voices(s, POLYPHONY)
            d.sfizz_enable_freewheeling(s)
            if not d.sfizz_load_file(s, sfz_path.encode('utf-8')):
                raise RuntimeError(f'sfizz could not load {sfz_path}')
            assert d.sfizz_get_sample_quality(s, SFIZZ_PROCESS_FREEWHEELING) == QUALITY
            n_reg = d.sfizz_get_num_regions(s)
            nb = (n_frames + BLOCK - 1) // BLOCK
            out = np.zeros((2, nb * BLOCK), dtype=np.float32)
            L, R = np.zeros(BLOCK, np.float32), np.zeros(BLOCK, np.float32)
            chans = (ctypes.POINTER(ctypes.c_float) * 2)(L.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                                         R.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
            i = 0
            for b in range(nb):
                b0 = b * BLOCK
                while i < len(events) and events[i][0] < b0 + BLOCK:
                    t, k, p, v = events[i]
                    (d.sfizz_send_note_on if k == 'on' else d.sfizz_send_note_off)(s, int(t - b0), int(p), int(v))
                    i += 1
                d.sfizz_render_block(s, chans, 2, BLOCK)
                out[0, b0:b0 + BLOCK] = L
                out[1, b0:b0 + BLOCK] = R
            return out[:, :n_frames].T.copy(), n_reg
        finally:
            d.sfizz_free(s)


def midi_events(perf, t0, t1):
    """Same pedal logic as render_excerpt(): returns [(sample, 'on'|'off', pitch, vel)]."""
    from corpus.perf_common import pedal_down_at, PEDAL_THRESH
    from corpus.render_soundfont_performances import build_events
    held, sustained, out = {}, set(), []
    ped = pedal_down_at(perf, t0)
    N = int(round(EXCERPT_S * SR))
    for t, _pr, kind, p, v in build_events(perf, t0, t1):
        s = min(max(int(round((t - t0) * SR)), 0), N)
        if kind == 'on':
            out.append((s, 'on', p, v)); held[p] = held.get(p, 0) + 1; sustained.discard(p)
        elif kind == 'off':
            if held.get(p, 0) > 0:
                held[p] -= 1
            if held.get(p, 0) == 0:
                if ped:
                    sustained.add(p)
                else:
                    out.append((s, 'off', p, 0))
        else:
            down = v >= PEDAL_THRESH
            if ped and not down:
                for q in sorted(sustained):
                    if held.get(q, 0) == 0:
                        out.append((s, 'off', q, 0))
                sustained.clear()
            ped = down
    return out


def sort_events(ev):
    return sorted(ev, key=lambda e: (e[0], e[1] == 'on'))     # note-off before note-on at one sample


# ---------------------------------------------------------------- events (local, needs mido)
def cmd_events():
    from corpus.perf_common import ROOT, MAESTRO, load_midi, window_notes, read_plan
    out = os.path.join(ROOT, EVENTS_DIR)
    os.makedirs(out, exist_ok=True)
    for r in read_plan():
        perf = load_midi(os.path.join(MAESTRO, r['maestro_midi_filename'].replace('/', os.sep)))
        t0, t1 = float(r['excerpt_start_s']), float(r['excerpt_end_s'])
        ev = sort_events(midi_events(perf, t0, t1))
        with open(os.path.join(out, f"slot{int(r['slot']):03d}.json"), 'w', encoding='utf-8') as f:
            json.dump({'slot': int(r['slot']), 'n_notes': len(window_notes(perf, t0, t1)),
                       'events': ev}, f)
    print(len(read_plan()), 'event files ->', out)


# ---------------------------------------------------------------- render (any machine)
def cmd_render(root: str, shard: int, n_shards: int, libs: list):
    synth = Sfizz(root)
    slots = sorted(int(f[4:7]) for f in os.listdir(os.path.join(root, EVENTS_DIR)) if f.endswith('.json'))
    jobs = [(lib, s) for s in slots for lib in libs]
    jobs = [jobs[i] for i in np.random.default_rng(20261025).permutation(len(jobs))][shard::n_shards]  # mix libs per shard
    os.makedirs(os.path.join(root, LOG_DIR), exist_ok=True)
    log = os.path.join(root, LOG_DIR, f'shard{shard:02d}of{n_shards:02d}.csv')
    done = set()
    if os.path.exists(log):
        with open(log, encoding='utf-8') as f:
            done = {(r['lib'], int(r['slot'])) for r in csv.DictReader(f)}
    N = int(round(EXCERPT_S * SR))
    for lib, slot in jobs:
        if (lib, slot) in done:
            continue
        with open(os.path.join(root, EVENTS_DIR, f'slot{slot:03d}.json'), encoding='utf-8') as f:
            ev = [tuple(e) for e in json.load(f)['events']]
        tt = time.time()
        x, n_reg = synth.render(os.path.join(root, LIBS[lib]), ev, N)
        mono = x.mean(axis=1)
        pk = float(np.max(np.abs(mono)))
        dst = os.path.join(root, AUDIO_OUT, lib, f'{lib}_slot{slot:03d}.wav')
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        sf.write(dst, (mono / pk * PEAK).astype(np.float32), SR, subtype='FLOAT')
        row = dict(lib=lib, slot=slot, n_regions=n_reg, peak_before_norm=round(pk, 6),
                   raw_peak_stereo=round(float(np.max(np.abs(x))), 6), render_s=round(time.time() - tt, 1))
        new = not os.path.exists(log)
        with open(log, 'a', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if new:
                w.writeheader()
            w.writerow(row)
        print(f'{lib:10s} slot {slot:3d} regions {n_reg:4d} pk {pk:.4f} {row["render_s"]:6.1f}s', flush=True)


# ---------------------------------------------------------------- manifest (local)
def cmd_manifest():
    from corpus.perf_common import ROOT, DATA_V2, read_plan
    plan = {int(r['slot']): r for r in read_plan()}
    logs = {}
    ld = os.path.join(ROOT, LOG_DIR)
    for fn in sorted(os.listdir(ld)):
        with open(os.path.join(ld, fn), encoding='utf-8') as f:
            for r in csv.DictReader(f):
                logs[(r['lib'], int(r['slot']))] = r
    rows = []
    for (lib, slot), lg in sorted(logs.items(), key=lambda kv: (list(LIBS).index(kv[0][0]), kv[0][1])):
        p = plan[slot]
        rel = os.path.join(AUDIO_OUT, lib, f'{lib}_slot{slot:03d}.wav').replace(os.sep, '/')
        assert os.path.exists(os.path.join(ROOT, rel)), rel
        rows.append({
            'file': rel, 'synth_type': 'sampled_perf', 'instrument': lib, 'slot': slot,
            'plan_file': p['file'], 'plan_engine': p['engine'], 'sfz_file': LIBS[lib].replace(os.sep, '/'),
            'engine': 'sfizz 1.2.3 C API (ctypes), freewheeling', 'sample_quality': QUALITY,
            'block': BLOCK, 'polyphony': POLYPHONY, 'sr': SR, 'channels': 1, 'subtype': 'FLOAT',
            'normalisation': 'mono=mean(L,R); peak->0.9', 'peak_before_norm': lg['peak_before_norm'],
            'midi_source': 'MAESTRO v3.0.0 MIDI (CC BY-NC-SA 4.0)',
            'maestro_midi_filename': p['maestro_midi_filename'],
            'canonical_composer': p['canonical_composer'], 'canonical_title': p['canonical_title'],
            'year': p['year'], 'maestro_split': p['maestro_split'],
            'excerpt_start_s': p['excerpt_start_s'], 'excerpt_end_s': p['excerpt_end_s'],
            'n_notes': p['n_notes'] if 'n_notes' in p else '',
            'pedal_down_at_start': p['pedal_down_at_start'], 'frac_time_pedal_down': p['frac_time_pedal_down'],
            'perf_also_in_maestro_clips': p['perf_also_in_maestro_clips'],
            'note_truth': f'sfz_note_truth/{lib}_slot{slot:03d}.csv',
        })
    dst = os.path.join(results_dir('corpus'), 'sfz_perf_manifest.csv')
    with open(dst, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(len(rows), 'rows ->', dst)


if __name__ == '__main__':
    c = sys.argv[1]
    if c == 'events':
        cmd_events()
    elif c == 'render':
        cmd_render(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5:] or list(LIBS))
    elif c == 'manifest':
        cmd_manifest()
    else:
        raise SystemExit(__doc__)
