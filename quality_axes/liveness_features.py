"""Liveness descriptors of every rated file: the 5 file-level modulation descriptors of modulation.py
(beat_mod, beat_flatness, fm_wander, slow_coherent, beat_to_slow) for every file of data/corpus/features_cal.csv.
The table used by the models is data/quality/liveness_cal.csv; re-extraction (needs the audio) writes
results/quality_models/liveness_cal.csv (resumable).

    SPD_AUDIO=<audio root> python -m quality_axes.liveness_features

Liveness = the low-frequency life of a note: incoherent unison and decay beating (beat_flatness, fm_wander,
beat_to_slow) that real instruments have and a plain physical model lacks.  On polyphonic audio nearly everything
looks alive, so these descriptors mainly separate dead isolated synthetic notes.
"""
from __future__ import annotations
import os
import csv

from quality_axes.modulation import modulation_features
from detector.segment import load_mono

from detector.paths import AUDIO_ROOT   # noqa: E402
from quality_axes.models import FEATURES_CAL, LIVENESS_CAL, save_path   # noqa: E402
AUDIO_DIR = os.path.join(AUDIO_ROOT, 'audio')   # features_cal 'file' col = name under audio/
OUT = LIVENESS_CAL                               # the shipped table (read by timbre.py)

LIVENESS_COLS = ['beat_mod', 'beat_flatness', 'fm_wander', 'slow_coherent', 'beat_to_slow']


EXTRACT_OUT = save_path('liveness_cal.csv')


def _done() -> set[str]:
    if not os.path.exists(EXTRACT_OUT):
        return set()
    return {r['file'] for r in csv.DictReader(open(EXTRACT_OUT, encoding='utf-8'))}


def main():
    rows = list(csv.DictReader(open(FEATURES_CAL, encoding='utf-8')))
    done = _done()
    new = os.path.getsize(EXTRACT_OUT) == 0 if os.path.exists(EXTRACT_OUT) else True
    fh = open(EXTRACT_OUT, 'a', newline='', encoding='utf-8')
    wr = csv.writer(fh)
    if not os.path.exists(EXTRACT_OUT) or os.path.getsize(EXTRACT_OUT) == 0:
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
    print(f"DONE: ok={n_ok} skip={n_skip} err={n_err} → {EXTRACT_OUT}")


if __name__ == '__main__':
    main()
