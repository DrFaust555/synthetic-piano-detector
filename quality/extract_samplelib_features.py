"""Extract 47-feature 'pseudo-recording' rows for the free sample libraries and score
them with the trained realness critic — tests whether the sample-based ceiling holds
across multiple independent, citable libraries (addresses weakness W1).

    python -m quality.extract_samplelib_features

Each library's isolated notes are deduped (one velocity per note) and partitioned into
several multi-note 'takes' (so the granularity matches the real multi-note corpus). Each
take -> one feature row. Rows are written to cache/samplelib_features.csv and scored with
cache/realness_xgb.json (aliveness = (1-P(synth))*100).
"""

from __future__ import annotations
import os
import re
import csv
import glob
import json
import random
import numpy as np

from quality.pipeline import note_raw_descriptors, aggregate_file, FEATURE_COLS
from quality.phasefeat import compute_phaseogram
from quality.build_wave_cache import safe_name
from detune.segment import load_mono, build_segments

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CACHE = os.path.join(HERE, 'cache')
TRENDS = json.load(open(os.path.join(CACHE, 'register_trends_cal.json'), encoding='utf-8'))
OUT = os.path.join(CACHE, 'samplelib_features.csv')
PHASE_DIR = os.path.join(CACHE, 'phase')
PHASE_MANIFEST = os.path.join(CACHE, 'phase_manifest.json')
SEED, TAKE_SIZE, N_TAKES = 0, 14, 8

LIBS = {
    'Salamander': ['audio/ref_bright_yamaha_salamander/**/*.ogg'],
    'K18':        ['audio/ref_dull_upright/**/*.wav'],
    'FreePats':   ['audio/ref_bad_old_piano/**/*.wav'],
    'Iowa':       ['audio/ref_top_grand_iowa_steinway/**/*.aif*'],
}


def _note(name):
    m = re.search(r'([A-G]#?b?)(-?\d)', os.path.basename(name))
    return (m.group(1) + m.group(2)) if m else None


def _dedup(files):
    by = {}
    for f in files:
        t = _note(f)
        if t and t not in by:           # first velocity per note
            by[t] = f
    return by


def main():
    os.makedirs(PHASE_DIR, exist_ok=True)
    rng = random.Random(SEED)
    cols = ['file', 'instrument'] + FEATURE_COLS
    rows, man = [], []
    for lib, globs in LIBS.items():
        files = []
        for g in globs:
            files += glob.glob(os.path.join(ROOT, g), recursive=True)
        notes_by = _dedup(sorted(files))
        # process each unique note-file once
        per_file = {}
        for tok, path in notes_by.items():
            try:
                _, notes, tuning = note_raw_descriptors(path)
                if not notes:
                    continue
                x, sr = load_mono(path)
                segs = [s for s in build_segments(x, sr)[0] if 24.0 < s.f0 < 4500.0]
                if not segs:
                    continue
                s = max(segs, key=lambda s: s.energy)
                per_file[tok] = (notes, tuning, compute_phaseogram(s.x, sr, s.f0))
            except Exception:
                continue
        toks = list(per_file)
        print(f"{lib}: {len(toks)} usable notes", flush=True)
        for k in range(N_TAKES):
            sample = rng.sample(toks, min(TAKE_SIZE, len(toks)))
            merged, tunings, phs = [], [], []
            for t in sample:
                merged += per_file[t][0]; tunings.append(per_file[t][1]); phs.append(per_file[t][2])
            keys = set().union(*[set(t) for t in tunings])
            mtune = {kk: float(np.nanmedian([t.get(kk, np.nan) for t in tunings])) for kk in keys}
            feats = aggregate_file(merged, mtune, TRENDS)
            fid = f'samplelib/{lib}_take{k}'
            rows.append({'file': fid, 'instrument': lib, **feats})
            np.save(os.path.join(PHASE_DIR, safe_name(fid) + '.npy'),
                    np.stack(phs).astype(np.float32))
            man.append({'file': fid, 'label': 1, 'stype': 'sampled_digital',
                        'group': f'syn:samplelib:{lib}', 'n_notes': len(phs)})

    with open(OUT, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in rows:
            w.writerow([r.get(c, '') if not isinstance(r.get(c), float)
                        else ('' if r[c] != r[c] else round(r[c], 6)) for c in cols])
    print(f"\nWrote {len(rows)} pseudo-recording rows -> {OUT}", flush=True)

    # append phaseogram-cache takes to the phase manifest (replacing any prior samplelib)
    existing = []
    if os.path.exists(PHASE_MANIFEST):
        existing = [m for m in json.load(open(PHASE_MANIFEST, encoding='utf-8'))
                    if not str(m.get('file', '')).startswith('samplelib/')]
    json.dump(existing + man, open(PHASE_MANIFEST, 'w', encoding='utf-8'))
    print(f"Appended {len(man)} samplelib takes to phase_manifest "
          f"({len(existing) + len(man)} total)", flush=True)

    # NOTE: do NOT score the libraries with cache/realness_xgb.json here — once train_realness
    # includes the libraries, that model has seen them (in-sample → leakage). The valid held-out
    # evaluation is the GroupKFold (by instrument) in train_realness.py / realness_net.py, which
    # reports the sampled_digital aliveness out-of-fold.
    print("Run `python -m quality.train_realness` (or realness_net) for the held-out evaluation.")


if __name__ == '__main__':
    main()
