"""Матриця сліпоти (Фаза 1) — AUC кожного критика окремо ПО СІМ'ЯХ негативів проти
real-good. Показує, де саме кожен критик сліпий (низький AUC = сім'я його обдурює).

Сім'ї негативів:
  otey_raw     oteySamples/ + рендери фізмоделі (otey/out_full/)
  otey_refined out/stereo/natur/*_corrected + modelSamples/oggSamples/ (нейро-виправлені)
  samplelib    modelSamples/{Belarus_upright,FreePats_old,Iowa_Steinway,K18_upright,Salamander_Yam}/
  m1           audio/m1_patches/
  pianoteq     audio/pianoteq/  (комерційний фізмодель — near-real, найважчий)

    python -m quality.critic_eval                      # матриця + cache/critic_eval.csv
    python -m quality.critic_eval --ckpt v2             # net з cache/versions/realness_net_v2.pt
    python -m quality.critic_eval --real 40 --limit 30  # менша вибірка (швидше)
"""
from __future__ import annotations
import os
import csv
import glob
import argparse
import numpy as np

from sklearn.metrics import roc_auc_score

from quality.critic_zoo import default_critics, REAL_MIN

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CACHE = os.path.join(HERE, 'cache')
AUDIO = os.path.join(ROOT, 'audio')

FAMILIES = {
    'otey_raw':     ['oteySamples/*.wav', 'otey/out_full/*.wav'],
    'otey_refined': ['out/stereo/natur/*_corrected.wav', 'modelSamples/oggSamples/*.wav'],
    'samplelib':    ['modelSamples/Belarus_upright/*.wav', 'modelSamples/FreePats_old/*.wav',
                     'modelSamples/Iowa_Steinway/*.wav', 'modelSamples/K18_upright/*.wav',
                     'modelSamples/Salamander_Yam/*.wav'],
    'm1':           ['audio/m1_patches/*.wav'],
    'pianoteq':     ['audio/pianoteq/*.wav'],
}


def _glob_family(patterns, cap, rng):
    files = []
    for pat in patterns:
        files += sorted(glob.glob(os.path.join(ROOT, pat)))
    if cap and len(files) > cap:
        files = [files[i] for i in sorted(rng.choice(len(files), cap, replace=False))]
    return files


def _real_good(cap, rng):
    files = []
    with open(os.path.join(CACHE, 'features_cal.csv'), encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if int(r.get('is_anchor', 0) or 0):
                continue
            try:
                if float(r['rating_cal']) < REAL_MIN:
                    continue
            except (ValueError, KeyError):
                continue
            if 'm1_patches' in r['file']:
                continue
            p = os.path.join(AUDIO, r['file'])
            if os.path.exists(p):
                files.append(p)
    if cap and len(files) > cap:
        files = [files[i] for i in sorted(rng.choice(len(files), cap, replace=False))]
    return files


def _p_synth_over(critic, files):
    out = []
    for f in files:
        try:
            out.append(critic.score(f)['p_synth'])
        except Exception:
            out.append(np.nan)
    return np.array(out, dtype=float)


def run(real_cap=50, fam_cap=40, ckpt='v1', seed=0):
    if ckpt and ckpt != 'v1':
        import quality.realness_net as rn
        rn.NET_CKPT = os.path.join(CACHE, 'versions', f'realness_net_{ckpt}.pt')
        rn._NET = None
        print(f"[net ckpt] {os.path.relpath(rn.NET_CKPT, ROOT)}")
    rng = np.random.default_rng(seed)
    critics = default_critics()
    real = _real_good(real_cap, rng)
    fams = {name: _glob_family(pats, fam_cap, rng) for name, pats in FAMILIES.items()}
    print(f"real-good {len(real)} | " +
          " ".join(f"{k}={len(v)}" for k, v in fams.items()), flush=True)

    rows = []
    fam_names = [k for k in FAMILIES if fams[k]]
    hdr = f"{'critic':10s}" + ''.join(f'{k[:11]:>13s}' for k in fam_names)
    print("\nAUC real-good vs family (1.0 = catches it; low = BLIND):")
    print(hdr); print('-' * len(hdr))
    for c in critics:
        p_real = _p_synth_over(c, real)
        line = f"{(c.name + ('*' if c.advisory else '')):10s}"
        rec = {'critic': c.name, 'advisory': int(c.advisory)}
        for k in fam_names:
            p_fam = _p_synth_over(c, fams[k])
            y = np.r_[np.zeros(len(p_real)), np.ones(len(p_fam))]
            p = np.r_[p_real, p_fam]
            ok = np.isfinite(p)
            auc = (float(roc_auc_score(y[ok], p[ok]))
                   if len(set(y[ok])) == 2 else float('nan'))
            rec[k] = round(auc, 4)
            line += f"{auc:13.3f}"
        rows.append(rec)
        print(line, flush=True)
    print("\n* = advisory (у звіт; НЕ у worst-case ансамблю)")

    out = os.path.join(CACHE, 'critic_eval.csv')
    with open(out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['critic', 'advisory'] + fam_names)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"-> {os.path.relpath(out, ROOT)}")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--real', type=int, default=50)
    ap.add_argument('--limit', type=int, default=40, help='cap per negative family')
    ap.add_argument('--ckpt', default='v1', help="net checkpoint: v1 (default) | v2 | …")
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    run(real_cap=args.real, fam_cap=args.limit, ckpt=args.ckpt, seed=args.seed)


if __name__ == '__main__':
    main()
