"""Канальна робастність фазограмної мережі (go/no-go для v2, Фаза 2).

Кожен файл оцінюється net-критиком двічі: чистим і після waveform-домен
channel_aug (кімната/EQ/компресія/шум/OGG — «інший запис того самого звуку»).
Робастний критик має давати ту саму відповідь: separability (AUC real-vs-synth)
не повинна падати від зміни каналу.

Критерій активації v2:  ΔAUC = AUC_clean − AUC_aug  ≤ 0.03,
і v2 має тримати канал не гірше за v1.

Примітка: real-good файли були у тренуванні обох версій, тож абсолютний AUC тут
оптимістичний; чесна величина — саме ΔAUC (деградація від каналу на ТИХ САМИХ
файлах), її in-sample не роздуває.

    python -m quality.channel_robustness                    # v1 і v2, порівняння
    python -m quality.channel_robustness --ckpt v2          # лише один чекпойнт
    python -m quality.channel_robustness --real 20 --limit 10   # швидше
"""
from __future__ import annotations
import os
import csv
import argparse
import tempfile
import numpy as np

from sklearn.metrics import roc_auc_score

from quality.critic_eval import FAMILIES, _glob_family, _real_good

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CACHE = os.path.join(HERE, 'cache')


def _aug_seed(path: str) -> int:
    """Детермінований seed на файл: та сама аугментація для v1 і v2."""
    import zlib
    return zlib.crc32(os.path.basename(path).encode('utf-8')) & 0x7fffffff


def _score_pair(path: str, scorer=None):
    """(p_synth_clean, p_synth_aug) для одного файлу; nan якщо не вдалось.
    scorer(path)->p_synth; default = net_p_synth."""
    from quality.channel_aug import augment
    from detune.segment import load_mono
    import soundfile as sf
    if scorer is None:
        from quality.realness_net import net_p_synth
        scorer = net_p_synth
    try:
        p_clean = scorer(path)
    except Exception:
        p_clean = float('nan')
    p_aug = float('nan')
    try:
        x, sr = load_mono(path)
        y = augment(x, sr, np.random.default_rng(_aug_seed(path)))
        fd, tmp = tempfile.mkstemp(suffix='.wav')
        os.close(fd)
        try:
            sf.write(tmp, y, sr, subtype='FLOAT')
            p_aug = scorer(tmp)
        finally:
            os.remove(tmp)
    except Exception:
        pass
    return p_clean, p_aug


def summarize(labels, p_clean, p_aug):
    """AUC clean/aug + Δ + зсув скорів реальних. Чиста функція (тестована)."""
    labels = np.asarray(labels, dtype=float)
    p_clean = np.asarray(p_clean, dtype=float)
    p_aug = np.asarray(p_aug, dtype=float)

    def _auc(p):
        ok = np.isfinite(p)
        if len(set(labels[ok])) < 2:
            return float('nan')
        return float(roc_auc_score(labels[ok], p[ok]))

    auc_c, auc_a = _auc(p_clean), _auc(p_aug)
    real = labels == 0
    okr = real & np.isfinite(p_clean) & np.isfinite(p_aug)
    real_shift = float(np.mean(p_aug[okr] - p_clean[okr])) if okr.any() else float('nan')
    return {'auc_clean': auc_c, 'auc_aug': auc_a,
            'delta': (auc_c - auc_a) if np.isfinite(auc_c) and np.isfinite(auc_a) else float('nan'),
            'real_p_shift': real_shift}


def run_ckpt(ckpt, files, labels):
    scorer = None
    if ckpt == 'xgb':                      # канало-робастний feature-критик як референс
        from quality.critic_zoo import XgbCritic
        xc = XgbCritic()
        scorer = lambda p: float(xc.score(p)['p_synth'])
        print("[xgb] channel-robust feature critic (референс досяжного dAUC)", flush=True)
    else:
        import quality.realness_net as rn
        if ckpt and ckpt != 'v1':
            rn.NET_CKPT = os.path.join(CACHE, 'versions', f'realness_net_{ckpt}.pt')
        else:
            rn.NET_CKPT = os.path.join(CACHE, 'realness_net.pt')
        rn._NET = None
        print(f"[{ckpt}] net ckpt: {os.path.relpath(rn.NET_CKPT, ROOT)}", flush=True)
    pc, pa = [], []
    for i, f in enumerate(files):
        c, a = _score_pair(f, scorer)
        pc.append(c); pa.append(a)
        if (i + 1) % 20 == 0:
            print(f"  [{ckpt}] {i+1}/{len(files)}", flush=True)
    s = summarize(labels, pc, pa)
    print(f"  [{ckpt}] AUC clean={s['auc_clean']:.3f}  aug={s['auc_aug']:.3f}  "
          f"dAUC={s['delta']:+.3f}  real p_synth shift={s['real_p_shift']:+.3f}", flush=True)
    return s, pc, pa


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--real', type=int, default=30)
    ap.add_argument('--limit', type=int, default=12, help='cap на сім\'ю негативів')
    ap.add_argument('--ckpt', default=None,
                    help="чекпойнти через кому (напр. 'v1,v2b'); default = v1,v2")
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    real = _real_good(args.real, rng)
    files, labels, fam_of = list(real), [0] * len(real), ['real'] * len(real)
    for name, pats in FAMILIES.items():
        fs = _glob_family(pats, args.limit, rng)
        files += fs; labels += [1] * len(fs); fam_of += [name] * len(fs)
    print(f"files: {len(files)} (real {labels.count(0)}, synth {labels.count(1)})", flush=True)

    ckpts = args.ckpt.split(',') if args.ckpt else ['v1', 'v2']
    results, scores = {}, {}
    for ck in ckpts:
        results[ck], pc, pa = run_ckpt(ck, files, labels)
        scores[ck] = (pc, pa)

    out = os.path.join(CACHE, 'channel_robustness.csv')
    with open(out, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['file', 'label', 'family'] +
                   [f'{ck}_{c}' for ck in ckpts for c in ('clean', 'aug')])
        for i, fp in enumerate(files):
            w.writerow([os.path.relpath(fp, ROOT), labels[i], fam_of[i]] +
                       [round(scores[ck][j][i], 4) for ck in ckpts for j in (0, 1)])
    print(f"-> {os.path.relpath(out, ROOT)}", flush=True)

    print(f"\nVERDICT (критерій dAUC <= 0.03):")
    for ck, s in results.items():
        ok = np.isfinite(s['delta']) and s['delta'] <= 0.03
        print(f"  {ck}: dAUC={s['delta']:+.3f}  {'PASS' if ok else 'FAIL'}")
    if len(ckpts) == 2:
        a, b = results[ckpts[0]], results[ckpts[1]]
        if np.isfinite(a['delta']) and np.isfinite(b['delta']):
            better = ckpts[1] if b['delta'] < a['delta'] else ckpts[0]
            print(f"  канал краще тримає: {better} "
                  f"({ckpts[0]} {a['delta']:+.3f} vs {ckpts[1]} {b['delta']:+.3f})")


if __name__ == '__main__':
    main()
