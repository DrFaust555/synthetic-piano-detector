"""Control C1b re-renders: SoundFont notes held 5.0 s and Otey notes rendered for 5.0 s.

SoundFont: exactly quality/render_soundfonts.py (TinySoundFont, 44.1 kHz, bank 0 / program 0, notes MIDI
21-96 step 3 x velocities 64/100/127 rendered sequentially with ONE synth per SoundFont, stereo -> mono
mean, silent outputs skipped, peak-normalised to 0.9, FLOAT) with note-on 5.0 s instead of 1.4 s (release
0.4 s).  The same code at 1.4 s is first checked to reproduce the stored renders.
Otey: Stattia/revision/data_v2/scripts/otey_v0_build/piano_v0.exe (bit-exact rebuild of the paper's
otey/out_full, see build_otey_v0.py) with the same knob set and velocity map, 5.0 s, no release / damper;
the same call at 3.0 s is first checked against otey/out_full.
Output: out_v3/cache/audio_c1b/{sf,otey}/... (FLOAT WAV).
"""
from __future__ import annotations

import os
import re
import sys
import glob
import subprocess
import tempfile
import numpy as np
import soundfile as sf

import v3_lib as V
from pv2_lib import ROOT

SRC = os.path.join(ROOT, 'audio', 'SynthPianoSamples')
SR = 44100
NOTES = list(range(21, 97, 3))
VELS = [64, 100, 127]
_PC = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
DST = os.path.join(V.CACHE3, 'audio_c1b')
OTEY_SCRIPTS = os.path.join(ROOT, 'Stattia', 'revision', 'data_v2', 'scripts')


def note_name(m):
    return f"{_PC[m % 12]}{m // 12 - 1}".replace('#', 's')


def sf2_list():
    s = sorted(glob.glob(os.path.join(SRC, '**', '*.sf2'), recursive=True))
    return [p for p in s if os.sep + 'rendered' + os.sep not in p]


def inst_name(path):
    return re.sub(r'[^A-Za-z0-9]+', '_', os.path.splitext(os.path.basename(path))[0]).strip('_')


def _render_sf2(path, dur, rel=0.4):
    import tinysoundfont
    synth = tinysoundfont.Synth(samplerate=SR)
    sfid = synth.sfload(path)
    synth.program_select(0, sfid, 0, 0)
    out = {}
    for m in NOTES:
        for v in VELS:
            try:
                synth.noteon(0, m, v)
                a1 = np.frombuffer(synth.generate(int(SR * dur)), dtype=np.float32)
                synth.noteoff(0, m)
                a2 = np.frombuffer(synth.generate(int(SR * rel)), dtype=np.float32)
                a = np.concatenate([a1, a2]).reshape(-1, 2).mean(axis=1)
            except Exception:
                continue
            pk = np.max(np.abs(a))
            if pk < 1e-3:
                continue
            out[f'{note_name(m)}_v{v}'] = (a / pk) * 0.9
    synth.sfunload(sfid)
    return out


def render_soundfonts(log=print):
    rep = {}
    for p in sf2_list():
        inst = inst_name(p)
        odir = os.path.join(DST, 'sf', inst)
        ref_dir = os.path.join(SRC, 'rendered', inst)
        done = os.path.isdir(odir) and len(glob.glob(os.path.join(odir, '*.wav'))) == len(glob.glob(os.path.join(ref_dir, '*.wav')))
        # reproduction check at the original 1.4-s hold (always run; cheap)
        chk = _render_sf2(p, 1.4)
        md = 0.0
        for k, a in chk.items():
            rf = os.path.join(ref_dir, k + '.wav')
            if os.path.exists(rf):
                b, _ = sf.read(rf, dtype='float32')
                md = max(md, float(np.max(np.abs(a.astype(np.float32) - b))) if len(a) == len(b) else np.inf)
        if not done:
            os.makedirs(odir, exist_ok=True)
            for k, a in _render_sf2(p, 5.0).items():
                sf.write(os.path.join(odir, k + '.wav'), a, SR, subtype='FLOAT')
        rep[inst] = {'repro_maxdiff_1p4s': md, 'n_check': len(chk), 'rendered_now': not done}
        log(f'  [C1b render] {inst}: reproduction max|d| at 1.4 s = {md:.2e}')
    return rep


def render_otey(log=print):
    sys.path.insert(0, OTEY_SCRIPTS)
    from build_otey_v0 import EXE, knob_args, vel_to_ms
    odir = os.path.join(DST, 'otey')
    os.makedirs(odir, exist_ok=True)
    kargs = knob_args()
    ref = sorted(glob.glob(os.path.join(ROOT, 'otey', 'out_full', '*.wav')))
    rep = {'n': len(ref)}

    def midi_of(name):
        m = re.match(r'([A-G])(s|#)?(\d)', name)
        pc = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}[m.group(1)] + (1 if m.group(2) else 0)
        return 12 * (int(m.group(3)) + 1) + pc

    def run(midi, vel, dur, out):
        with tempfile.TemporaryDirectory() as td:
            cmd = [EXE, f'{dur:.4f}', str(midi), f'{vel_to_ms(vel):.3f}', '--out', os.path.join(td, 'o.wav')] + kargs
            subprocess.run(cmd, cwd=td, capture_output=True, check=True)
            x, sr = sf.read(os.path.join(td, 'o.wav'), dtype='float32')
            sf.write(out, x, sr, subtype='FLOAT')
    # reproduction check (3.0 s) on two notes
    md = 0.0
    for r in ref[:2]:
        b = os.path.basename(r)
        name, v = re.match(r'(.+)_v(\d+)\.wav', b).groups()
        t = os.path.join(odir, '_check_' + b)
        run(midi_of(name), int(v), 3.0, t)
        x, _ = sf.read(t, dtype='float32')
        y, _ = sf.read(r, dtype='float32')
        md = max(md, float(np.max(np.abs(x - y))) if len(x) == len(y) else np.inf)
        os.remove(t)
    rep['repro_maxdiff_3s'] = md
    log(f'  [C1b render] Otey reproduction max|d| at 3.0 s = {md:.2e}')
    todo = [r for r in ref if not os.path.exists(os.path.join(odir, os.path.basename(r)))]
    for r in todo:
        b = os.path.basename(r)
        name, v = re.match(r'(.+)_v(\d+)\.wav', b).groups()
        run(midi_of(name), int(v), 5.0, os.path.join(odir, b))
    rep['rendered_now'] = len(todo)
    return rep
