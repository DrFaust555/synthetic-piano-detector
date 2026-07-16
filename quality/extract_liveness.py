"""Axis L — extract file-level liveness (modulation) features for every labelled
file in features_cal.csv, into cache/liveness_cal.csv.  Resumable (checkpoint +
skip already-done).  Merged into the timbre-quality model (axis B+L).

    python -m quality.extract_liveness            # extract (resume)

Liveness = the low-frequency "life" of a note: incoherent unison/decay beating
(beat_flatness, fm_wander, beat_to_slow) that real instruments have and a dead
physical model lacks (DISCRIMINATOR.md §11b).  Complementary to timbre/tuning:
on polyphonic audio everyone looks alive, so this mainly separates DEAD isolated
synth (Otey) — m1 (chorus) is caught by the timbre/tuning axes instead.
"""
from __future__ import annotations
import os
import csv

from quality.modulation import modulation_features
from detune.segment import load_mono

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
AUDIO_DIR = os.path.join(ROOT, 'audio')   # features_cal 'file' col = basename under audio/
FEATURES_CAL = os.path.join(HERE, 'cache', 'features_cal.csv')
OUT = os.path.join(HERE, 'cache', 'liveness_cal.csv')

LIVENESS_COLS = ['beat_mod', 'beat_flatness', 'fm_wander', 'slow_coherent', 'beat_to_slow']


def _done() -> set[str]:
    if not os.path.exists(OUT):
        return set()
    return {r['file'] for r in csv.DictReader(open(OUT, encoding='utf-8'))}


def main():
    rows = list(csv.DictReader(open(FEATURES_CAL, encoding='utf-8')))
    done = _done()
    new = os.path.getsize(OUT) == 0 if os.path.exists(OUT) else True
    fh = open(OUT, 'a', newline='', encoding='utf-8')
    wr = csv.writer(fh)
    if not os.path.exists(OUT) or os.path.getsize(OUT) == 0:
        wr.writerow(['file'] + LIVENESS_COLS)
    n_ok = n_skip = n_err = 0
    for i, r in enumerate(rows):
        f = r['file']
        if f in done:
            n_skip += 1
            continue
        path = f if os.path.isabs(f) else os.path.join(AUDIO_DIR, f)
        try:
            x, sr = load_mono(path)
            m = modulation_features(x, sr)
            wr.writerow([f] + [m.get(c, 0.0) for c in LIVENESS_COLS])
            n_ok += 1
        except Exception:  # noqa: BLE001 — missing/unreadable file → NaN row
            wr.writerow([f] + ['nan'] * len(LIVENESS_COLS))
            n_err += 1
        if (i + 1) % 200 == 0:
            fh.flush()
            print(f"  {i+1}/{len(rows)}  ok={n_ok} skip={n_skip} err={n_err}", flush=True)
    fh.close()
    print(f"DONE: ok={n_ok} skip={n_skip} err={n_err} → {OUT}")


if __name__ == '__main__':
    main()
