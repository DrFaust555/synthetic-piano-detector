"""Build the blind real-vs-synthetic listening-test kit (paper §5.5).

    python -m quality.build_listening_kit

Picks isolated mid-register notes from each class, trims/normalizes each to a uniform
~3 s clip, randomizes the order, and writes neutrally-named clips + a hidden answer key
to Stattia/listening_kit/.  Classes: real (Iowa Steinway, K18 upright) vs synthetic
(soundfont, Otey physical model, Salamander = sample-based digital — the ceiling class).
"""

from __future__ import annotations
import os
import re
import glob
import random
import csv
import numpy as np
import librosa
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, 'Stattia', 'listening_kit')
SR = 44100
DUR = 3.0
SEED = 7
PREF_LO, PREF_HI = 48, 76          # prefer C3..E5 (natural register)

_PC = {'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5, 'F#': 6,
       'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11}


def midi_of(tok):
    m = re.match(r'^([A-G]#?)(-?\d)$', tok or '')
    return _PC[m.group(1)] + 12 * (int(m.group(2)) + 1) if m else None


def note_tok(name):
    m = re.search(r'([A-G]#?)(-?\d)', os.path.basename(name))
    return (m.group(1) + m.group(2)) if m else None


def _files(globs):
    fs = []
    for g in globs:
        fs += glob.glob(os.path.join(ROOT, g), recursive=True)
    return fs


# class_label, real/synth, source_name, globs, n_clips
CLASSES = [
    ('real',            'real',  'Iowa Steinway',       ['audio/ref_top_grand_iowa_steinway/**/*.aif*'], 7),
    ('real',            'real',  'K18 upright',         ['audio/ref_dull_upright/**/*.wav'], 7),
    ('soundfont',       'synth', 'soundfont',           ['audio/SynthPianoSamples/rendered/*/*.wav'], 5),
    ('otey',            'synth', 'Otey physical model', ['otey/out_full/*.wav'], 5),
    ('sampled_digital', 'synth', 'Salamander (sampled)', ['audio/ref_bright_yamaha_salamander/**/*.ogg'], 6),
]


def _velkey(f):
    s = os.path.basename(f).lower()
    for i, v in enumerate(['mf', '_v100', 'v100', '.f.', '_v127', 'v127', 'ff', 'v8', 'v9', 'v10']):
        if v in s:
            return i
    return 99


def pick(fs, n, rng):
    by = {}
    for f in fs:
        t = note_tok(f)
        if t:
            by.setdefault(t, []).append(f)
    cand = [t for t in by if midi_of(t) is not None and PREF_LO <= midi_of(t) <= PREF_HI]
    if len(cand) < n:
        cand = list(by.keys())
    rng.shuffle(cand)
    notes = sorted(cand[:n], key=lambda t: midi_of(t) or 0)
    return [sorted(by[t], key=_velkey)[0] for t in notes]


def process(path):
    y, _ = librosa.load(path, sr=SR, mono=True)
    pk = np.max(np.abs(y)) + 1e-9
    on = np.argmax(np.abs(y) > 0.02 * pk)        # trim leading silence
    on = max(0, on - int(0.01 * SR))
    seg = y[on:on + int(DUR * SR)]
    if len(seg) < int(DUR * SR):
        seg = np.pad(seg, (0, int(DUR * SR) - len(seg)))
    fade = int(0.01 * SR)
    seg[:fade] *= np.linspace(0, 1, fade)
    seg[-fade:] *= np.linspace(1, 0, fade)
    rms = np.sqrt(np.mean(seg ** 2)) + 1e-9       # loudness-normalise
    seg = seg * (0.1 / rms)
    p = np.max(np.abs(seg))
    if p > 0.97:
        seg *= 0.97 / p
    return seg.astype(np.float32)


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = random.Random(SEED)
    picked = []
    for label, rs, src, globs, n in CLASSES:
        fs = _files(globs)
        for f in pick(fs, n, rng):
            picked.append({'class': label, 'real_synth': rs, 'source': src,
                           'note': note_tok(f), 'src': os.path.relpath(f, ROOT).replace('\\', '/')})
    rng.shuffle(picked)
    key_rows = []
    for i, it in enumerate(picked, 1):
        cid = f'clip_{i:02d}'
        try:
            seg = process(os.path.join(ROOT, it['src']))
        except Exception as e:
            print(f"  ERR {it['src']}: {e}"); continue
        sf.write(os.path.join(OUT, cid + '.wav'), seg, SR, subtype='PCM_16')
        key_rows.append({'clip': cid, **it})
    with open(os.path.join(OUT, 'answer_key.csv'), 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=['clip', 'class', 'real_synth', 'source', 'note', 'src'])
        w.writeheader(); w.writerows(key_rows)
    n_real = sum(1 for r in key_rows if r['real_synth'] == 'real')
    print(f"Wrote {len(key_rows)} clips -> {OUT}  (real {n_real}, synth {len(key_rows)-n_real})")
    from collections import Counter
    print("by class:", dict(Counter(r['class'] for r in key_rows)))


if __name__ == '__main__':
    main()
