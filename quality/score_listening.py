"""Score the blind real-vs-synthetic listening test (paper §5.5).

    python -m quality.score_listening [responses.csv]

Joins listener responses with the hidden answer key and reports human accuracy overall
and PER CLASS, with a binomial test of "caught rate > chance" per synthetic sub-class.
The hypothesis under test: humans catch soundfont/Otey well but FAIL on sample-based
digitals (caught ≈ chance) — i.e. humans hit the same ceiling as the per-note critic.
Drop the example row from responses_template.csv before collecting real responses.
"""

from __future__ import annotations
import os
import re
import sys
import csv
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
KIT = os.path.join(ROOT, 'Stattia', 'listening_kit')

# critic's per-class aliveness (P(real)*100) for side-by-side context
CRITIC_ALIVE = {'real': 97, 'soundfont': 11, 'otey': 2, 'sampled_digital': 90}


def _binom_gt_half(k, n):
    """P(X>=k | p=0.5) — one-sided; small p => caught above chance."""
    if n == 0:
        return float('nan')
    try:
        from scipy.stats import binomtest
        return binomtest(k, n, 0.5, alternative='greater').pvalue
    except Exception:
        from math import erfc, sqrt
        z = (k - n * 0.5) / (0.5 * sqrt(n))
        return 0.5 * erfc(z / sqrt(2))


def main():
    import glob
    arg = sys.argv[1] if len(sys.argv) > 1 else KIT
    if os.path.isdir(arg):                       # every returned file (.csv from download, .txt from copy/share)
        files = sorted(glob.glob(os.path.join(arg, '*.csv')) + glob.glob(os.path.join(arg, '*.txt')))
    else:
        files = [arg]
    key = {}
    with open(os.path.join(KIT, 'answer_key.csv'), encoding='utf-8') as f:
        for r in csv.DictReader(f):
            key[r['clip']] = (r['real_synth'], r['class'])

    rows, src_files = [], []
    for fp in files:
        for r in csv.DictReader(open(fp, encoding='utf-8')):
            rows.append(r); src_files.append(os.path.basename(fp))
    keep = [(r, f) for r, f in zip(rows, src_files)
            if (r.get('listener_id') or '').strip().lower()
            not in ('example_nick', 'listener_id', '')]   # drop sample + pasted-in headers
    rows = [r for r, _ in keep]; src_files = [f for _, f in keep]
    if not rows:
        print(f"No responses found in {arg}. Drop the returned responses_*.csv there."); return
    _EXP_FIX = {'amatour': 'amateur', 'amayeur': 'amateur'}  # typos seen in returned forms
    for r in rows:
        e = (r.get('experience') or '').strip().lower()
        r['experience'] = _EXP_FIX.get(e, e)
    clip_cols = sorted([c for c in rows[0].keys() if re.fullmatch(r'clip_\d+', c or '')]) if rows else []
    print(f"Listeners: {len(rows)}   clips scored: {len(clip_cols)}")

    # per (truth_class) accumulate judgments: 'caught' = judged synthetic
    classes = ['real', 'soundfont', 'otey', 'sampled_digital']
    acc = {c: {'n': 0, 'correct': 0, 'caught': 0, 'conf': []} for c in classes}
    overall_n = overall_correct = 0
    for r in rows:
        for c in clip_cols:
            v = (r.get(c) or '').strip()
            if v not in ('1', '2', '3', '4') or c not in key:
                continue
            truth_rs, truth_cls = key[c]
            pred_synth = int(v) >= 3
            correct = (pred_synth and truth_rs == 'synth') or (not pred_synth and truth_rs == 'real')
            a = acc[truth_cls]
            a['n'] += 1
            a['correct'] += int(correct)
            a['caught'] += int(pred_synth)
            a['conf'].append(1 if v in ('1', '4') else 0)
            overall_n += 1
            overall_correct += int(correct)

    print(f"\nOverall human accuracy: {100*overall_correct/max(1,overall_n):.0f}%  (n={overall_n})\n")
    print(f"  {'class':16s} {'n':>4s} {'acc/caught':>11s} {'critic_alive':>12s} {'p(>chance)':>11s}")
    for c in classes:
        a = acc[c]
        if a['n'] == 0:
            continue
        if c == 'real':
            rate = a['correct'] / a['n']
            metric = f"acc {100*rate:3.0f}%"
            p = ''
        else:                                   # synth: caught rate + binom test vs chance
            rate = a['caught'] / a['n']
            metric = f"caught {100*rate:3.0f}%"
            p = f"{_binom_gt_half(a['caught'], a['n']):.3f}"
        print(f"  {c:16s} {a['n']:>4d} {metric:>11s} {CRITIC_ALIVE.get(c,''):>12} {p:>11}")

    # ceiling by listener expertise (does the 'even experts fail' claim hold?)
    print("\nSample-based caught% by listener experience:")
    for gname, exps in [('expert (musician/technician)', {'musician', 'technician'}),
                        ('non-expert (lay/amateur)', {'lay', 'amateur'})]:
        n = k = 0
        for r in rows:
            if (r.get('experience') or '').strip() not in exps:
                continue
            for c in clip_cols:
                v = (r.get(c) or '').strip()
                if v in ('1', '2', '3', '4') and c in key and key[c][1] == 'sampled_digital':
                    n += 1; k += int(int(v) >= 3)
        if n:
            print(f"  {gname:28s} n={n:3d}  caught={100*k/n:4.0f}%  p={_binom_gt_half(k, n):.3f}")

    sd = acc['sampled_digital']
    if sd['n']:
        rate = 100 * sd['caught'] / sd['n']
        p = _binom_gt_half(sd['caught'], sd['n'])
        verdict = ("CEILING CONFIRMED: humans cannot catch sample-based digitals above chance"
                   if p > 0.05 else
                   "humans DO catch sample-based digitals above chance — ceiling weaker than hypothesized")
        print(f"\n  Sample-based digitals: caught {rate:.0f}%, p={p:.3f} -> {verdict}")

    # ---- listener-level (cluster-robust) inference: judgments within a listener are NOT
    # independent, so the pooled binomial above is optimistic for the significant classes.
    # Per-listener rate per class -> one-sided Wilcoxon signed-rank vs chance 0.5.
    per_listener = []                                    # (uid, exp, {cls: rate}, acc_overall)
    for i, r in enumerate(rows):
        uid = f"{(r.get('listener_id') or '?').strip()}@{src_files[i]}"
        cnt = {c: [0, 0] for c in classes}
        n_ok = n_corr = 0
        for c in clip_cols:
            v = (r.get(c) or '').strip()
            if v not in ('1', '2', '3', '4') or c not in key:
                continue
            truth_rs, truth_cls = key[c]
            pred_synth = int(v) >= 3
            cnt[truth_cls][0] += 1
            cnt[truth_cls][1] += int(pred_synth) if truth_rs == 'synth' else int(not pred_synth)
            n_ok += 1
            n_corr += int((pred_synth and truth_rs == 'synth') or (not pred_synth and truth_rs == 'real'))
        rates = {c: (cnt[c][1] / cnt[c][0] if cnt[c][0] else float('nan')) for c in classes}
        per_listener.append((uid, r['experience'], rates, n_corr / max(1, n_ok)))

    print("\nListener-level tests (per-listener rate vs chance 0.5, one-sided Wilcoxon):")
    try:
        from scipy.stats import wilcoxon
        for c in classes[1:]:
            vals = np.array([pl[2][c] for pl in per_listener if np.isfinite(pl[2][c])])
            stat = wilcoxon(vals - 0.5, alternative='greater', zero_method='zsplit')
            print(f"  {c:16s} mean caught {100*vals.mean():3.0f}%  sd {100*vals.std():3.0f}  "
                  f"listeners>chance {np.sum(vals > 0.5)}/{len(vals)}  p={stat.pvalue:.4f}")
    except ImportError:
        print("  (scipy missing — skipped)")

    # ---- confidence: 'sure' = 1 (definitely real) / 4 (definitely synthetic)
    print("\nHigh-confidence judgments only (v=1 'sure real' / v=4 'sure synth'):")
    for c in classes:
        n = k = sure = tot = 0
        for r in rows:
            for cc in clip_cols:
                v = (r.get(cc) or '').strip()
                if v not in ('1', '2', '3', '4') or cc not in key or key[cc][1] != c:
                    continue
                tot += 1
                if v in ('1', '4'):
                    sure += 1; n += 1; k += int(v == '4')
        if n:
            lbl = 'sure-synth' if key and c != 'real' else 'sure-synth'
            print(f"  {c:16s} sure {100*sure/max(1,tot):3.0f}% of judgments; "
                  f"of sure: called synth {100*k/n:3.0f}%")

    # ---- inter-rater agreement on the real/synth binarization
    mat = []
    for r in rows:
        mat.append([1 if (r.get(c) or '').strip() in ('3', '4') else
                    0 if (r.get(c) or '').strip() in ('1', '2') else -1 for c in clip_cols])
    mat = np.array(mat)
    pairs = agree = 0
    for j in range(mat.shape[1]):
        col = mat[:, j][mat[:, j] >= 0]
        n1 = int(col.sum()); n0 = len(col) - n1
        pairs += len(col) * (len(col) - 1) / 2
        agree += n1 * (n1 - 1) / 2 + n0 * (n0 - 1) / 2
    print(f"\nInter-rater agreement (pairwise, real/synth): {100*agree/max(1,pairs):.0f}%")

    # ---- top listeners (QC + who beats the panel)
    print("\nTop-5 listeners by overall accuracy:")
    for uid, exp, rates, a in sorted(per_listener, key=lambda t: -t[3])[:5]:
        print(f"  {a*100:3.0f}%  {exp:10s} sampled-caught {100*rates['sampled_digital']:3.0f}%  {uid}")

    # ---- anonymized export for the paper's data release (no names)
    if '--anon' in sys.argv:
        out = os.path.join(KIT, 'results_anon.csv')
        order = np.argsort([-pl[3] for pl in per_listener])  # stable, non-identifying order
        with open(out, 'w', newline='', encoding='utf-8') as fh:
            w = csv.writer(fh)
            w.writerow(['listener', 'experience'] + clip_cols)
            for k_, i in enumerate(order):
                w.writerow([f'L{k_+1:02d}', rows[i]['experience']]
                           + [(rows[i].get(c) or '').strip() for c in clip_cols])
        print(f"\nanonymized responses -> {os.path.relpath(out, ROOT)}")


if __name__ == '__main__':
    main()
