"""Question 2, final setting B (digital instruments in training) WITHOUT the Roland SC-55 sound module and the
Roland GP-3 (3 files; not digital pianos, not discussed in the paper).  XGBoost, 47 features, own-fold by group;
SFZ sampled play never trained.  AUC [95% CI] / detection at 5 % false alarms; also setting A for reference.
    python experiments/01_detection/performances_features.py      -> results/01_detection/q2_final.md
"""
import os, sys
import numpy as np
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))   # repository root
from detector.paths import DATA, RESULTS, AUDIO_ROOT, results_dir   # noqa: E402
OUT = results_dir('01_detection')
ROOT = AUDIO_ROOT
import protocol.common as L
import protocol.corpus as V
import protocol.corpus_base as P
from protocol.corpus import PERF_KINDS

df, _ = V.build_corpus_v3(log=lambda *a: None)
d = df[df.kind.isin(PERF_KINDS) & ~df.kind.isin(['sound_module', 'digital_other'])].reset_index(drop=True)
sfz, _ = P._filter_valid(pd.read_csv(os.path.join(DATA, 'performances', 'sfz_perf_features.csv')))
sfz = sfz.reset_index(drop=True)
sfz['grp'] = 'syn:sfz:' + sfz['instrument']
fmap = L.assign_folds(df.loc[df.base, 'grp'], pd.concat([df.loc[~df.base, 'grp'], sfz['grp']]))
sf = sfz.grp.map(fmap).astype(int).values
k, sk, y, grp = d.kind.values, d.subkind.values, d.y.values, d.grp.values
rp = k == 'real_performance'
dig = np.isin(k, ['sampled_digital_performance', 'modelled_digital', 'm1'])
rows = [('SoundFont renders', sk == 'sf2_perf'), ('Otey renders', sk == 'otey_perf'), ('Korg M1', k == 'm1'),
        ('Digital pianos, modeled', k == 'modelled_digital'), ('Digital pianos, sampled', k == 'sampled_digital_performance'),
        ('all non-acoustic (trained kinds)', y == 1)]
md = [f'# Question 2 without SC-55 / GP-3 ({len(d)} performance files, {int(rp.sum())} real)\n\n'
      '| setting | ' + ' | '.join(r[0] for r in rows) + ' | SFZ sampled play* | real FA at 0.5 |\n' + '|---' * (len(rows) + 3) + '|\n']
for name, tr, cond in (('A: digital never trained', ~dig, 'T2'), ('B: digital in training (final)', np.ones(len(d), bool), 'T1')):
    r = V.run_cv3(d.assign(trainable=tr), list(L.FEATURE_COLS), cond, ext={'sfz': sfz}, ext_fold={'sfz': sf})
    p, ps = r['own'], r['ext_own']['sfz']
    t5 = np.quantile(p[rp], 0.95)
    cells = [f'{V.fmt3(V.cmp3(y, p, grp, rp, m), det=False)} / {np.mean(p[m] > t5):.0%}' for _, m in rows]
    yy = np.r_[np.zeros(rp.sum()), np.ones(len(ps))]
    cs = V.cmp3(yy, np.r_[p[rp], ps], np.r_[grp[rp], sfz.grp.values], yy == 0, yy == 1)
    md.append(f'| {name} | ' + ' | '.join(cells) + f' | {V.fmt3(cs, det=False)} / {np.mean(ps > t5):.0%} | {np.mean(p[rp] > 0.5):.1%} |\n')
open(os.path.join(OUT, 'q2_final.md'), 'w', encoding='utf-8').write(''.join(md))
print(''.join(md))
