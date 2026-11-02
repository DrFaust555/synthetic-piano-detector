"""Sec. 4.2: a well-tuned acoustic piano draws more false alarms.

Paper numbers: among the 20 % best-tuned real performances, 11.7 % exceed the 5 %-false-alarm threshold of the
fusion detector, against 0.8 % among the 20 % worst-tuned (5 % overall).

Tuning = detune_final, the integral detuning of the performance (one of the 47 features; smaller = better tuned).
The real performances are split into fifths by its rank.  The fusion detector is the performance fusion of the
phaseogram CNN and the 47 features with the digital pianos in training (data/predictions/perf/fusion_perf_f*.npz, each
file scored by the model of its own fold); its threshold is the 95th percentile of the scores of the real
performances.  The same table split by the unison index is given as a check.

    python experiments/02_sampled_vs_algorithmic/tuned_false_alarms.py
        -> results/02_sampled_vs_algorithmic/tuned_false_alarms.md
"""
import os
import sys
import numpy as np
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))   # repository root
from detector.paths import DATA, results_dir   # noqa: E402
import protocol.corpus as V                    # noqa: E402

OUT = results_dir('02_sampled_vs_algorithmic')


def main():
    mp = pd.read_csv(os.path.join(DATA, 'predictions', 'perf_meta.csv'))
    ms = pd.read_csv(os.path.join(DATA, 'predictions', 'perf_meta_sfz.csv'))
    meta = pd.concat([mp, ms], ignore_index=True)
    fold = meta.fold.values.astype(int)
    df, _ = V.build_corpus_v3(log=lambda *a: None)
    d = df.set_index('file').loc[mp.file].reset_index()
    p = np.full(len(meta), np.nan)
    for k in range(5):
        p[fold == k] = np.load(os.path.join(DATA, 'predictions', 'perf', f'fusion_perf_f{k}.npz'))['p'][fold == k]
    p = p[:len(d)]
    r = d[(d.kind == 'real_performance').values].copy()
    r['p'] = p[(d.kind == 'real_performance').values]
    t5 = np.quantile(r.p, 0.95)
    md = ['# False alarms of the fusion detector by tuning of the real performance\n\n',
          f'{len(r)} real performances; threshold = 95th percentile of their scores (P = {t5:.3f}); overall above it: '
          f'{np.mean(r.p > t5):.1%}.\n\n']
    for col, lab in (('detune_final', 'integral detuning (detune_final)'), ('unison_index', 'unison index')):
        q = pd.qcut(r[col].rank(method='first'), 5, labels=False)
        md.append(f'**Fifths by {lab}** (1 = best tuned)\n\n| fifth | files | median {col} | above threshold |\n'
                  '|---|---|---|---|\n')
        for i, g in r.groupby(q):
            md.append(f'| {int(i) + 1} | {len(g)} | {g[col].median():.2f} | {np.mean(g.p > t5):.1%} |\n')
        if r[col].isna().any():
            md.append(f'\n{int(r[col].isna().sum())} file(s) without a value are left out.\n')
        md.append('\n')
    open(os.path.join(OUT, 'tuned_false_alarms.md'), 'w', encoding='utf-8').write(''.join(md))
    print(''.join(md))


if __name__ == '__main__':
    main()
