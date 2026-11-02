"""Real performances by piano brand (brand parsed from the source folder / file name; brands with >= 20 performances):
each brand scored against all other real performances.  Median over brands of the AUC and of the share above the
5 % false-alarm threshold, for the feature detector (setting B, own fold, as q2_final), the network and the fusion
(cnn_iso/jobs_perf_cnn_dig_sfz).
    python experiments/01_detection/real_brands.py   -> results/01_detection/real_perf_brands.md
"""
import os, sys
import numpy as np
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))   # repository root
from detector.paths import DATA, RESULTS, AUDIO_ROOT, results_dir   # noqa: E402
OUT = results_dir('01_detection')
import protocol.common as L
import protocol.corpus as V
from protocol.corpus import PERF_KINDS


def brand(f):
    if '/' in f:
        d, b = f.split('/', 1)
        return b.split('_')[0] if d == 'brands_extra' else d.replace('_extra', '')
    t = f.split('_')
    return t[1] if len(t) > 2 else '?'


df, _ = V.build_corpus_v3(log=lambda *a: None)
d = df[df.kind.isin(PERF_KINDS) & ~df.kind.isin(['sound_module', 'digital_other'])].reset_index(drop=True)
P = {'Features': pd.Series(V.run_cv3(d.assign(trainable=True), list(L.FEATURE_COLS), 'T1')['own'], index=d.file)}
meta = pd.read_csv(os.path.join(DATA, 'predictions', 'perf_meta.csv'))
for name, m in (('Network', 'cnn_perf'), ('Fusion', 'fusion_perf')):
    p = np.full(len(meta), np.nan)
    for k in range(5):
        p[meta.fold.values == k] = np.load(os.path.join(DATA, 'predictions', 'perf', f'{m}_f{k}.npz'))['p'][:len(meta)][meta.fold.values == k]
    P[name] = pd.Series(p, index=meta.file)
real = d.loc[d.kind == 'real_performance', 'file']
br = real.map(brand).values
vc = pd.Series(br).value_counts()
big = vc[vc >= 20].index
md = [f'# Real performances by brand ({len(big)} brands with >= 20 performances, {int(vc[big].sum())} of {len(real)} files)\n\n'
      '| detector | median AUC (min-max) | median share above threshold (min-max) | highest |\n|---|---|---|---|\n']
for name, s in P.items():
    pr = s.loc[real].values
    t = np.quantile(pr, 0.95)
    a, q = [], []
    for b in big:
        g = br == b
        a.append(L.auc_fast(np.r_[np.zeros((~g).sum()), np.ones(g.sum())], np.r_[pr[~g], pr[g]]))
        q.append(np.mean(pr[g] > t))
    a, q = np.array(a), np.array(q)
    md.append(f'| {name} | {np.median(a):.3f} ({a.min():.2f}-{a.max():.2f}) | {np.median(q):.1%} ({q.min():.0%}-{q.max():.0%}) | '
              f'{big[np.argmax(q)]} {q.max():.0%} |\n')
open(os.path.join(OUT, 'real_perf_brands.md'), 'w', encoding='utf-8').write(''.join(md))
print(''.join(md))
