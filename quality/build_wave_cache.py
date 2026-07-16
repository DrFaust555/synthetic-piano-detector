"""One-time per-note phaseogram cache for the time-domain realness_net.

    python -m quality.build_wave_cache [--force] [--max-real N]

For real (ratings_calibrated, minus m1) and synthetic (soundfont renders + cleaned
YouTube digitals + Otey + m1) files: segment → diversity-subsample notes → one (C, P)
phaseogram per note → save (n_notes, C, P) per file under quality/cache/phase/.
Manifest carries the realness label (synth=1), synth type and the anti-leakage group.
"""

from __future__ import annotations
import os
import csv
import json
import glob
import argparse
import random
import numpy as np

from detune.segment import load_mono, build_segments
from quality.pipeline import select_diverse_notes
from quality.phasefeat import compute_phaseogram, C, P
from quality.grouping import group_key

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
AUDIO = os.path.join(ROOT, 'audio')
CACHE = os.path.join(HERE, 'cache', 'phase')
MANIFEST = os.path.join(HERE, 'cache', 'phase_manifest.json')
RATINGS = os.path.join(AUDIO, 'ratings_calibrated.csv')


def safe_name(fn: str) -> str:
    return fn.replace('/', '__').replace('\\', '__')


def _items(max_real: int):
    """List of (abs_path, file_id, label, stype, group)."""
    items = []
    # ── real (+ m1 relabelled synth) from ratings_calibrated ──
    reals = []
    with open(RATINGS, encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if (r.get('rating_cal') or '').strip() in ('', 'nan'):
                continue
            fn = r['file']
            path = os.path.join(AUDIO, fn)
            if not os.path.exists(path):
                continue
            if 'm1_patches' in fn:
                items.append((path, fn, 1, 'm1', 'syn:m1'))
            else:
                reals.append((path, fn, 0, 'real', group_key(fn, r.get('brand', ''))))
    random.Random(0).shuffle(reals)
    # stratified-ish: keep one per group first, then fill up to max_real
    seen, primary, extra = set(), [], []
    for it in reals:
        (primary if it[4] not in seen else extra).append(it)
        seen.add(it[4])
    items += (primary + extra)[:max_real]
    # ── synth: renders / yt / otey ──
    for d in sorted(glob.glob(os.path.join(AUDIO, 'SynthPianoSamples/rendered/*/'))):
        inst = os.path.basename(os.path.normpath(d))
        if inst == '_demos':
            continue
        for p in glob.glob(os.path.join(d, '*.wav')):
            rel = os.path.relpath(p, ROOT).replace('\\', '/')
            items.append((p, rel, 1, 'rendered', f'syn:rendered:{inst}'))
    for p in glob.glob(os.path.join(AUDIO, 'SynthPianoSamples/yt/*.wav')):
        rel = os.path.relpath(p, ROOT).replace('\\', '/')
        inst = os.path.basename(p).split('_')[0]
        items.append((p, rel, 1, 'yt', f'syn:yt:{inst}'))
    for p in glob.glob(os.path.join(ROOT, 'otey/out_full/*.wav')):
        rel = os.path.relpath(p, ROOT).replace('\\', '/')
        items.append((p, rel, 1, 'otey', 'syn:otey:Otey'))
    return items


def _v2_extra_items():
    """Розширення набору (Фаза 2): MAESTRO-позитиви + банк негативів.
    Групи: maestro = РІК (не per-clip — той самий Disklavier/канал не має стояти по обидва
    боки фолда); всі Otey-похідні сім'ї (otey_*, oteySamples, samplelib=modelSamples-банки)
    = ОДНА група 'syn:otey:Otey', як у базі; m1 пропускаємо — база вже має ці файли як
    syn:m1 (інакше ті самі записи двічі під різними групами → GroupKFold-витік)."""
    items = []
    mm = os.path.join(ROOT, 'audio', 'maestro_clips_manifest.csv')
    if os.path.exists(mm):
        for r in csv.DictReader(open(mm, encoding='utf-8')):
            p = os.path.join(ROOT, 'audio', r['clip'])
            if os.path.exists(p):
                year = (r.get('year') or
                        os.path.basename(r['clip']).split('_')[0]).strip()
                items.append((p, r['clip'], 0, 'maestro', 'maestro:' + year))
    nm = os.path.join(HERE, 'cache', 'negatives_manifest.csv')
    if os.path.exists(nm):
        for r in csv.DictReader(open(nm, encoding='utf-8')):
            fam = r['family']
            if fam == 'm1':
                continue
            p = os.path.join(ROOT, r['path'])
            if os.path.exists(p):
                grp = ('syn:otey:Otey'
                       if fam.startswith('otey') or fam in ('oteySamples', 'samplelib')
                       else 'neg:' + fam)
                items.append((p, r['path'], 1, fam, grp))
    return items


def _phaseograms(path):
    x, sr = load_mono(path)
    segs, _ = build_segments(x, sr)
    idx = select_diverse_notes(segs)
    imgs = [compute_phaseogram(segs[j].x, sr, segs[j].f0)
            for j in idx if 24.0 < segs[j].f0 < 4500.0]
    return (np.stack(imgs).astype(np.float32) if imgs else None), (x, sr)


def _phaseograms_aug(x, sr, rng):
    from quality.channel_aug import augment
    y = augment(x, sr, rng)
    segs, _ = build_segments(y, sr)
    idx = select_diverse_notes(segs)
    imgs = [compute_phaseogram(segs[j].x, sr, segs[j].f0)
            for j in idx if 24.0 < segs[j].f0 < 4500.0]
    return np.stack(imgs).astype(np.float32) if imgs else None


def build(force=False, max_real=800, v2=False, aug_frac=0.5, seed=0, manifest_out=None):
    os.makedirs(CACHE, exist_ok=True)
    items = _items(max_real)
    if v2:
        items = items + _v2_extra_items()
        seen_paths, dd = set(), []                       # той самий файл лише раз
        for it in items:
            key = os.path.normcase(os.path.abspath(it[0]))
            if key not in seen_paths:
                seen_paths.add(key)
                dd.append(it)
        items = dd
    manifest_out = manifest_out or (os.path.join(HERE, 'cache', 'phase_manifest_v2.json')
                                    if v2 else MANIFEST)
    rng = np.random.default_rng(seed)
    print(f"Files: {len(items)}  (real {sum(1 for i in items if i[2]==0)}, "
          f"synth {sum(1 for i in items if i[2]==1)})  v2={v2} aug_frac={aug_frac if v2 else 0}",
          flush=True)
    manifest, total = [], 0
    for k, (path, fid, label, stype, grp) in enumerate(items):
        npy = os.path.join(CACHE, safe_name(fid) + '.npy')
        loaded_audio = None
        if os.path.exists(npy) and not force:
            arr = np.load(npy, mmap_mode='r')
        else:
            try:
                arr, loaded_audio = _phaseograms(path)
                if arr is None:
                    continue
                np.save(npy, arr)
            except Exception as e:
                print(f"  ERR {fid[:50]}: {e}", flush=True)
                continue
        total += int(arr.shape[0])
        manifest.append({'file': fid, 'label': label, 'stype': stype,
                         'group': grp, 'n_notes': int(arr.shape[0])})
        # v2 waveform-augmentation: додаткова копія ОБОХ класів (channel robustness) —
        # застосовуємо до НОТИ перед побудовою фазограми (не у фазограм-домені).
        # Аугментувати лише позитиви НЕ МОЖНА: мережа вивчає «бруд каналу = реальне»
        # (виявлено channel_robustness-тестом: v2-only-real dAUC +0.113 проти v1 +0.089).
        if v2 and rng.random() < aug_frac:
            aug_fid = fid + '#waug'                          # load_dataset шукає safe_name(file)+.npy
            aug_npy = os.path.join(CACHE, safe_name(aug_fid) + '.npy')
            try:
                if os.path.exists(aug_npy) and not force:
                    arr_a = np.load(aug_npy, mmap_mode='r')
                else:
                    if loaded_audio is None:
                        loaded_audio = load_mono(path)
                    arr_a = _phaseograms_aug(loaded_audio[0], loaded_audio[1], rng)
                    if arr_a is None:
                        raise ValueError('no aug notes')
                    np.save(aug_npy, arr_a)
                total += int(arr_a.shape[0])
                manifest.append({'file': aug_fid, 'label': label, 'stype': stype,
                                 'group': grp, 'n_notes': int(arr_a.shape[0]), 'aug': 1})
            except Exception as e:
                print(f"  (aug skip {fid[:40]}: {e})", flush=True)
        if (k + 1) % 100 == 0:
            print(f"  {k+1}/{len(items)}  notes={total}", flush=True)
    json.dump(manifest, open(manifest_out, 'w', encoding='utf-8'))
    print(f"\nDone: {len(manifest)} entries, {total} notes (C={C},P={P}) -> "
          f"{os.path.relpath(manifest_out, ROOT)}", flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--max-real', type=int, default=800)
    ap.add_argument('--v2', action='store_true',
                    help='розширений набір (MAESTRO+негативи) + waveform-aug -> phase_manifest_v2.json')
    ap.add_argument('--aug-frac', type=float, default=0.5)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    build(force=args.force, max_real=args.max_real, v2=args.v2,
          aug_frac=args.aug_frac, seed=args.seed)
