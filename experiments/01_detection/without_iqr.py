"""Question 2, final setting B: does the performance detector rest on cross-note spread?  XGBoost with all 47
features vs without the 16 inter-quartile ranges (features 17-32), vs the 16 IQRs alone; same corpus, folds and
scoring as q2_final.py (SFZ sampled play never trained).  AUC [95% CI] / detection at 5 % false alarms.
    python experiments/01_detection/without_iqr.py      -> results/01_detection/q2_no_iqr.md
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
rows = [('SoundFont renders', sk == 'sf2_perf'), ('Otey renders', sk == 'otey_perf'), ('Korg M1', k == 'm1'),
        ('Digital pianos, modeled', k == 'modelled_digital'), ('Digital pianos, sampled', k == 'sampled_digital_performance'),
        ('all non-acoustic', y == 1)]
ALL = list(L.FEATURE_COLS)
IQR = [c for c in ALL if c.endswith('_iqr')]
assert len(IQR) == 16, IQR
sets = {'all 47': ALL, 'without the 16 IQRs (31)': [c for c in ALL if c not in IQR], 'the 16 IQRs only': IQR}
md = [f'# Question 2, setting B: with and without the cross-note spread ({len(d)} files, {int(rp.sum())} real)\n\n'
      '| features | ' + ' | '.join(r[0] for r in rows) + ' | SFZ sampled play* |\n' + '|---' * (len(rows) + 2) + '|\n']
for name, cols in sets.items():
    r = V.run_cv3(d.assign(trainable=True), cols, 'T1', ext={'sfz': sfz}, ext_fold={'sfz': sf})
    p, ps = r['own'], r['ext_own']['sfz']
    t5 = np.quantile(p[rp], 0.95)
    cells = [f'{V.fmt3(V.cmp3(y, p, grp, rp, m), det=False)} / {np.mean(p[m] > t5):.0%}' for _, m in rows]
    yy = np.r_[np.zeros(rp.sum()), np.ones(len(ps))]
    cs = V.cmp3(yy, np.r_[p[rp], ps], np.r_[grp[rp], sfz.grp.values], yy == 0, yy == 1)
    md.append(f'| {name} | ' + ' | '.join(cells) + f' | {V.fmt3(cs, det=False)} / {np.mean(ps > t5):.0%} |\n')
    print(md[-1], flush=True)
open(os.path.join(OUT, 'q2_no_iqr.md'), 'w', encoding='utf-8').write(''.join(md))
print(''.join(md))
