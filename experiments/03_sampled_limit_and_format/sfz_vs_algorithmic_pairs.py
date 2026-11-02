"""Sec. 4.3: the SFZ sampled play against the algorithmic render of the same MAESTRO excerpt.

Paper numbers: the fusion detector scores the library render as less synthetic than the algorithmic render of the
same excerpt in 89 % of the 480 pairs (77 % Salamander, 88 % K18, 95 % FreePats, 97 % Iowa); median P(synthetic)
0.004 for the library renders and 0.977 for the algorithmic renders.

Each of the 120 excerpts (slot of data/performances/perf_excerpt_plan.csv) was rendered once algorithmically
(SoundFont or Otey) and once by each of the four SFZ libraries, so every library render forms one pair.  Scores:
data/predictions/perf/{fusion_perf,cnn_perf}_f*.npz (digital pianos in training, SFZ play never trained; each file scored
by the model of its own fold).  The network-alone rows are given for comparison.

    python experiments/03_sampled_limit_and_format/sfz_vs_algorithmic_pairs.py
        -> results/03_sampled_limit_and_format/sfz_vs_algorithmic_pairs.md
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))   # repository root
from detector.paths import DATA, results_dir   # noqa: E402

OUT = results_dir('03_sampled_limit_and_format')
LIBS = ['Salamander', 'K18', 'Iowa', 'FreePats']


def main():
    meta = pd.concat([pd.read_csv(os.path.join(DATA, 'predictions', 'perf_meta.csv')),
                      pd.read_csv(os.path.join(DATA, 'predictions', 'perf_meta_sfz.csv'))], ignore_index=True)
    fold = meta.fold.values.astype(int)
    plan = pd.read_csv(os.path.join(DATA, 'performances', 'perf_excerpt_plan.csv'))
    slot_of = dict(zip(plan['file'], plan['slot']))
    algo = (meta.kind == 'algo_performance').values
    keys = meta.loc[algo, 'file'].map(lambda f: slot_of.get(f, slot_of.get(f.replace('audio/', ''), np.nan)))
    assert keys.notna().all(), 'every algorithmic render must map to an excerpt slot'
    sfz = (meta.kind == 'sfz_performance').values
    slot = meta.file.str.extract(r'_slot(\d+)\.wav$')[0]
    md = ['# SFZ sampled play vs the algorithmic render of the same excerpt\n\n',
          'Pair = one library render and the algorithmic render of the same excerpt. "library lower" = share of pairs '
          'in which the library render has the lower P(synthetic); Wilcoxon signed-rank test on the differences.\n\n',
          '| detector | library | pairs | library lower | median P, library | median P, algorithmic | Wilcoxon p |\n'
          '|---|---|---|---|---|---|---|\n']
    for m, lab in (('fusion_perf', 'fusion'), ('cnn_perf', 'network alone')):
        p = np.full(len(meta), np.nan)
        for k in range(5):
            p[fold == k] = np.load(os.path.join(DATA, 'predictions', 'perf', f'{m}_f{k}.npz'))['p'][fold == k]
        a = pd.Series(p[algo], index=keys.values.astype(int))
        for lib in LIBS + [None]:
            msk = sfz & (meta.subkind == f'sfz:{lib}').values if lib else sfz
            s = pd.Series(p[msk], index=slot[msk].astype(int).values)
            rows = [(sv, av) for sl, sv in s.items() if sl in a.index for av in np.atleast_1d(a.loc[sl])]
            dd = np.array([x - y for x, y in rows])
            md.append(f'| {lab} | {lib or "all four"} | {len(dd)} | {np.mean(dd < 0):.0%} | '
                      f'{np.median([x for x, _ in rows]):.3f} | {np.median([y for _, y in rows]):.3f} | '
                      f'{wilcoxon(dd).pvalue:.1g} |\n')
    open(os.path.join(OUT, 'sfz_vs_algorithmic_pairs.md'), 'w', encoding='utf-8').write(''.join(md))
    print(''.join(md))


if __name__ == '__main__':
    main()
