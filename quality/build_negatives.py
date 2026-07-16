"""Банк негативів (Фаза 2) — маніфест `cache/negatives_manifest.csv` (path, family) для
перетренування критиків. Джерела негативів:

  otey_rand     рендери фізмоделі з ВИПАДКОВИМИ knobs у межах LO/HI (otey/tune_critic)
  otey_tuned    конфіги з історії otey/tune_runs.jsonl (best_knobs кожного запуску)
  otey_refined  частина рендерів, прогнана крізь нейро-refiner (nn/refiner_oggcal_ft.pt)
  samplelib/oteySamples/m1/pianoteq   наявні банки (лише посилання, без рендеру)

Рендери пишуться в audio/neg_bank/ (гітигнор *.wav). Маніфест — вхід для train_realness/
build_wave_cache (розширений набір негативів).

    python -m quality.build_negatives --n-rand 200 --refine 40   # повний банк
    python -m quality.build_negatives --smoke                      # 4 рендери, швидко
"""
from __future__ import annotations
import os
import csv
import json
import argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CACHE = os.path.join(HERE, 'cache')
NEG_DIR = os.path.join(ROOT, 'audio', 'neg_bank')
MANIFEST = os.path.join(CACHE, 'negatives_manifest.csv')
RUNS_LOG = os.path.join(ROOT, 'otey', 'tune_runs.jsonl')
REFINER_CKPT = os.path.join(ROOT, 'nn', 'refiner_oggcal_ft.pt')
SR = 44100


def _rel(p):
    return os.path.relpath(p, ROOT).replace('\\', '/')


def _knob_vec(knobs: dict, KNOBS, DEF):
    return np.array([float(knobs.get(k, DEF[i])) for i, k in enumerate(KNOBS)], dtype=float)


def render_random(n, seed=0, vel=75):
    from otey.tune_critic import LO, HI, render_clip
    rng = np.random.default_rng(seed)
    os.makedirs(NEG_DIR, exist_ok=True)
    rows = []
    for i in range(n):
        x = LO + rng.random(len(LO)) * (HI - LO)
        p = os.path.join(NEG_DIR, f'rand_{i:04d}.wav')
        try:
            render_clip(x, p, vel=vel)
            rows.append((_rel(p), 'otey_rand'))
        except Exception as e:
            print(f"  ERR rand_{i}: {e}")
        if (i + 1) % 25 == 0:
            print(f"  rand {i+1}/{n}", flush=True)
    return rows


def render_from_runs():
    from otey.tune_critic import KNOBS, DEF, render_clip
    if not os.path.exists(RUNS_LOG):
        return []
    os.makedirs(NEG_DIR, exist_ok=True)
    rows = []
    for k, line in enumerate(open(RUNS_LOG, encoding='utf-8')):
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        kn = rec.get('best_knobs') or {}
        x = _knob_vec(kn, KNOBS, DEF)
        p = os.path.join(NEG_DIR, f"run_{rec.get('runid', k)}.wav")
        try:
            render_clip(x, p, vel=int(rec.get('vel', 75)))
            rows.append((_rel(p), 'otey_tuned'))
        except Exception as e:
            print(f"  ERR run {k}: {e}")
    return rows


def refine_some(k, seed=0):
    """Per-note рендер випадкових knobs + нейро-refiner → сім'я otey_refined."""
    if not os.path.exists(REFINER_CKPT):
        print(f"  [refine] нема {os.path.relpath(REFINER_CKPT, ROOT)} — пропуск")
        return []
    try:
        import torch
        import soundfile as sf
        from nn.refiner import Refiner
        from inference.physical_model import PhysicalModel
        from nn.build_oggcal_pairs import cal_for_midi
        from otey.tune_critic import KNOBS, LO, HI, NOTES
    except Exception as e:
        print(f"  [refine] недоступно: {e}"); return []
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    G = Refiner().to(dev)
    ck = torch.load(REFINER_CKPT, weights_only=False, map_location=dev)
    G.load_state_dict(ck['state_dict'] if 'state_dict' in ck else ck)
    G.eval()
    rng = np.random.default_rng(seed)
    os.makedirs(NEG_DIR, exist_ok=True)
    rows = []
    for i in range(k):
        over = LO + rng.random(len(LO)) * (HI - LO)
        nm, m = NOTES[i % len(NOTES)]
        kn = dict(cal_for_midi(m))
        kn.update(mix=0.15, voicing_rand=0.0)
        kn.update({KNOBS[j]: float(over[j]) for j in range(len(KNOBS))})
        try:
            x = PhysicalModel(knobs=kn, use_cache=False).render(m, 75, 2.0)
            t = torch.from_numpy(x[None].astype(np.float32)).to(dev)
            mm = torch.tensor([m], dtype=torch.long, device=dev)
            vv = torch.tensor([75.0], device=dev)
            pp = torch.tensor([0.0], device=dev)
            with torch.no_grad():
                y = G(t, mm, vv, pp).squeeze(0).cpu().numpy()
            y = y / (np.max(np.abs(y)) + 1e-9) * 0.9
            p = os.path.join(NEG_DIR, f'refined_{i:04d}.wav')
            sf.write(p, y.astype(np.float32), SR)
            rows.append((_rel(p), 'otey_refined'))
        except Exception as e:
            print(f"  ERR refine {i}: {e}")
    return rows


def reference_banks():
    import glob
    fam = {
        'samplelib': ['modelSamples/Belarus_upright/*.wav', 'modelSamples/FreePats_old/*.wav',
                      'modelSamples/Iowa_Steinway/*.wav', 'modelSamples/K18_upright/*.wav',
                      'modelSamples/Salamander_Yam/*.wav'],
        'oteySamples': ['oteySamples/*.wav'],
        'm1': ['audio/m1_patches/*.wav'],
        'pianoteq': ['audio/pianoteq/*.wav'],
    }
    rows = []
    for family, pats in fam.items():
        for pat in pats:
            for p in sorted(glob.glob(os.path.join(ROOT, pat))):
                rows.append((_rel(p), family))
    return rows


def build(n_rand=200, refine=40, seed=0, include_banks=True):
    rows = []
    print("[negatives] random-knob renders …", flush=True)
    rows += render_random(n_rand, seed=seed)
    print("[negatives] tune_runs history …", flush=True)
    rows += render_from_runs()
    print("[negatives] refiner pass …", flush=True)
    rows += refine_some(refine, seed=seed)
    if include_banks:
        print("[negatives] reference banks …", flush=True)
        rows += reference_banks()
    os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)
    with open(MANIFEST, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(['path', 'family'])
        w.writerows(rows)
    from collections import Counter
    by = Counter(r[1] for r in rows)
    print(f"[negatives] {len(rows)} -> {os.path.relpath(MANIFEST, ROOT)}   {dict(by)}")
    return rows


FEATURES_NEG = os.path.join(CACHE, 'features_negatives.csv')


def extract_features():
    """FEATURE_COLS для НОВОРЕНДЕРЕНИХ негативів (otey_rand/otey_tuned/otey_refined) ->
    features_negatives.csv. Банк-сім'ї (samplelib/m1/pianoteq/oteySamples) уже покриті
    наявними таблицями синтетики, тож їх тут не переекстрактовуємо."""
    from quality.pipeline import note_raw_descriptors, aggregate_file, FEATURE_COLS
    if not os.path.exists(MANIFEST):
        print("Спершу build (рендер + маніфест)."); return
    import json as _json
    trends = _json.load(open(os.path.join(CACHE, 'register_trends_cal.json'), encoding='utf-8'))
    man = list(csv.DictReader(open(MANIFEST, encoding='utf-8')))
    new = [r for r in man if r['family'].startswith('otey_')]
    rows = []
    for i, r in enumerate(new):
        p = os.path.join(ROOT, r['path'])
        try:
            _m, notes, tuning = note_raw_descriptors(p)
            feats = aggregate_file(notes, tuning, trends)
            rows.append({'file': r['path'], 'family': r['family'],
                         **{c: feats.get(c, '') for c in FEATURE_COLS}})
        except Exception as e:
            print(f"  ERR {r['path']}: {e}")
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(new)} features", flush=True)
    with open(FEATURES_NEG, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['file', 'family'] + FEATURE_COLS)
        w.writeheader(); w.writerows(rows)
    print(f"[negatives] {len(rows)} рядків -> {os.path.relpath(FEATURES_NEG, ROOT)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--n-rand', type=int, default=200)
    ap.add_argument('--refine', type=int, default=40)
    ap.add_argument('--no-banks', action='store_true')
    ap.add_argument('--features', action='store_true',
                    help='після рендеру порахувати FEATURE_COLS для otey_* -> features_negatives.csv')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    if args.smoke:
        build(n_rand=4, refine=2, seed=args.seed)
        extract_features(); return
    build(n_rand=args.n_rand, refine=args.refine, seed=args.seed,
          include_banks=not args.no_banks)
    if args.features:
        extract_features()


if __name__ == '__main__':
    main()
