"""Extract intrinsic per-file features for every rated recording.

    python -m quality.extract_features            # all rated files
    python -m quality.extract_features --limit 5  # quick smoke test

Two passes: (1) collect raw per-note descriptors for all files and fit the
register (MIDI) trends; (2) register-normalise + aggregate each file.  Writes
``quality/cache/features.csv`` and ``quality/cache/register_trends.json`` (the
latter so the predictor applies the identical normalisation to new files).
"""

from __future__ import annotations
import os
import csv
import json
import argparse
import time

from quality.pipeline import (
    note_raw_descriptors, fit_trends, aggregate_file,
    FEATURE_COLS, META_COLS, DESCRIPTORS,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIO_DIR = os.path.join(ROOT, 'audio')
MANIFEST = os.path.join(AUDIO_DIR, 'manifest.csv')
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')
FEATURES_CSV = os.path.join(CACHE, 'features.csv')
TRENDS_JSON = os.path.join(CACHE, 'register_trends.json')


def _read_manifest() -> list[dict]:
    rows = []
    with open(MANIFEST, encoding='utf-8') as f:
        for r in csv.DictReader(f):
            rating = (r.get('rating') or '').strip()
            if not rating:
                continue
            try:
                r['rating'] = int(rating)
            except ValueError:
                continue
            rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser(description="Витяг інтринсивних ознак на файл.")
    ap.add_argument('--limit', type=int, default=0, help="обробити лише N файлів (тест)")
    args = ap.parse_args()

    os.makedirs(CACHE, exist_ok=True)
    rows = _read_manifest()
    if args.limit:
        rows = rows[:args.limit]
    print(f"Файлів з рейтингом: {len(rows)}")

    # --- pass 1: raw per-note descriptors + tuning ---
    per_file = []          # (manifest_row, meta, notes, tuning)
    all_notes = []
    t0 = time.time()
    for i, r in enumerate(rows, 1):
        path = os.path.join(AUDIO_DIR, r['file'])
        if not os.path.exists(path):
            print(f"  [пропуск] немає файлу: {r['file']}")
            continue
        try:
            meta, notes, tuning = note_raw_descriptors(path)
        except Exception as e:
            print(f"  [помилка] {r['file']}: {e}")
            continue
        per_file.append((r, meta, notes, tuning))
        all_notes.extend(notes)
        print(f"  [{i}/{len(rows)}] {r['file']:<42} нот={meta['n_notes']:3d} "
              f"detune={tuning['detune_final']}  ({time.time()-t0:.0f}s)")

    if not per_file:
        print("Немає оброблених файлів — нічого писати.")
        return

    # --- fit register trends on the pooled note population ---
    trends = fit_trends(all_notes)
    with open(TRENDS_JSON, 'w', encoding='utf-8') as f:
        json.dump(trends, f, ensure_ascii=False, indent=1)
    print(f"Trend-нормалізація збережена: {TRENDS_JSON}  "
          f"(нот усього: {len(all_notes)})")

    # --- pass 2: register-normalise + aggregate, write CSV ---
    cols = META_COLS + FEATURE_COLS
    n_written = 0
    with open(FEATURES_CSV, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r, meta, notes, tuning in per_file:
            feats = aggregate_file(notes, tuning, trends)
            rec = {
                'file': r['file'], 'rating': r['rating'],
                'type': r.get('type', ''), 'source': r.get('source', ''),
                'category': r.get('category', ''), 'mic_moving': r.get('mic_moving', ''),
                'n_notes': meta['n_notes'], 'median_midi': meta['median_midi'],
                'duration_s': round(meta['duration_s'], 2), 'mode': meta['mode'],
            }
            rec.update(feats)
            row = []
            for c in cols:
                v = rec.get(c, '')
                if isinstance(v, float):
                    v = '' if v != v else round(v, 6)   # NaN -> empty cell
                row.append(v)
            w.writerow(row)
            n_written += 1

    print(f"Записано {n_written} рядків -> {FEATURES_CSV}")
    print(f"Ознак на файл: {len(FEATURE_COLS)}  (descriptors {len(DESCRIPTORS)}x2 + tuning)")


if __name__ == '__main__':
    main()
