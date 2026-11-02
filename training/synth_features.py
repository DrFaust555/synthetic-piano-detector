"""Per-file features of the synthetic sources of the original corpus: data/corpus/synth_features.csv (needs the
audio).

    python training/synth_features.py

Sources (all labelled synthetic):
  rendered  audio/SynthPianoSamples/rendered/<inst>/*.wav   SoundFont single-note renders (one group per SoundFont)
  yt        audio/SynthPianoSamples/yt/*.wav                 digital-piano demos (cleaned)
  otey      otey/out_full/*.wav                              Otey physical-model notes
Same register trends as the real corpus (data/corpus/register_trends_cal.json).  Writes
results/training/synth_features.csv.  SYNTH_TYPES=yt (comma list) recomputes only these types and copies the other
rows from an existing results/training/synth_features.csv.
"""

from __future__ import annotations
import os
import sys
import csv
import json
import glob
import numpy as np

from detector.pipeline import note_raw_descriptors, aggregate_file, FEATURE_COLS

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..')))   # repository root
from detector.paths import DATA, RESULTS, AUDIO_ROOT, results_dir   # noqa: E402
OUT = results_dir('training')
ROOT = AUDIO_ROOT
TRENDS = os.path.join(DATA, 'corpus', 'register_trends_cal.json')
OUT = os.path.join(OUT, 'synth_features.csv')
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
    # SYNTH_TYPES=yt: recompute only these synth_types; the rows of the other types are copied unchanged
    # from the existing CSV (the protocol uses the isolated-note extraction for the single notes).
    only = [t for t in os.environ.get('SYNTH_TYPES', '').split(',') if t]
    keep_rows = []
    if only:
        items = [it for it in items if it[1] in only]
        with open(OUT, encoding='utf-8') as fh:
            rd = csv.reader(fh)
            hdr = next(rd)
            keep_rows = [r for r in rd if r[hdr.index('synth_type')] not in only]
    print(f"Synth files: {len(items)}", flush=True)
    cols = META + FEATURE_COLS
    n = 0
    with open(OUT, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in keep_rows:
            w.writerow(r)
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
