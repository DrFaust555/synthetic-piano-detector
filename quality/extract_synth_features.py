"""Extract the 47 analytic features for the SYNTHETIC negative class and label it.

    python -m quality.extract_synth_features

Sources (all labelled synth, with a tier):
  rendered  audio/SynthPianoSamples/rendered/<inst>/*.wav   (soundfont renders, clean)
  yt        audio/SynthPianoSamples/yt/*.wav                 (digital-piano demos, channel-matched, cleaned)
  otey      otey/out_full/*.wav                              (physical model, clean)
Uses the SAME register trends as the real corpus (register_trends_cal.json) so the
features are directly comparable.  Writes quality/cache/synth_features.csv.
"""

from __future__ import annotations
import os
import csv
import json
import glob
import numpy as np

from quality.pipeline import note_raw_descriptors, aggregate_file, FEATURE_COLS

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TRENDS = os.path.join(HERE, 'cache', 'register_trends_cal.json')
OUT = os.path.join(HERE, 'cache', 'synth_features.csv')
META = ['file', 'synth_type', 'instrument']


def _sources():
    items = []
    for d in sorted(glob.glob(os.path.join(ROOT, 'audio/SynthPianoSamples/rendered/*/'))):
        inst = os.path.basename(os.path.normpath(d))
        if inst == '_demos':
            continue
        for f in glob.glob(os.path.join(d, '*.wav')):
            items.append((f, 'rendered', inst))
    for f in glob.glob(os.path.join(ROOT, 'audio/SynthPianoSamples/yt/*.wav')):
        items.append((f, 'yt', os.path.basename(f).split('_')[0]))
    for f in glob.glob(os.path.join(ROOT, 'otey/out_full/*.wav')):
        items.append((f, 'otey', 'Otey'))
    return items


def main():
    trends = json.load(open(TRENDS, encoding='utf-8'))
    items = _sources()
    print(f"Synth files: {len(items)}", flush=True)
    cols = META + FEATURE_COLS
    n = 0
    with open(OUT, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for i, (path, styp, inst) in enumerate(items):
            try:
                meta, notes, tuning = note_raw_descriptors(path)
                feats = aggregate_file(notes, tuning, trends)
            except Exception as e:
                continue
            rel = os.path.relpath(path, ROOT).replace('\\', '/')
            rec = {'file': rel, 'synth_type': styp, 'instrument': inst, **feats}
            row = []
            for c in cols:
                v = rec.get(c, '')
                if isinstance(v, float):
                    v = '' if v != v else round(v, 6)
                row.append(v)
            w.writerow(row)
            n += 1
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(items)}", flush=True)
    print(f"\nWrote {n} synth rows -> {OUT}", flush=True)


if __name__ == '__main__':
    main()
