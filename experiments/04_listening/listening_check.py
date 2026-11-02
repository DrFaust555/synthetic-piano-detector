"""Independent recomputation of the listening-test statistics from the answers alone (no protocol imports):
judged-synthetic rate per class with pooled binomial and per-listener Wilcoxon tests, per-listener AUC (cross-checked
with Mann-Whitney U), panel AUC and pooled AUC; compared with data/listening/listening_per_class.csv.

    python experiments/04_listening/listening_check.py
        -> results/04_listening/listening_listener_auc.csv, listening_per_class_auditor.csv
"""
from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, binomtest, mannwhitneyu
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))   # repository root
from detector.paths import DATA, RESULTS, AUDIO_ROOT, results_dir   # noqa: E402
OUT = results_dir('04_listening')
ROOT = AUDIO_ROOT
KIT = os.path.join(DATA, 'listening')
OUTD = OUT
os.makedirs(OUTD, exist_ok=True)

key = pd.read_csv(os.path.join(KIT, 'answer_key.csv'))
res = pd.read_csv(os.path.join(KIT, 'results_anon.csv'))
clips = key['clip'].tolist()
R = res[clips].values.astype(int)
assert R.min() >= 1 and R.max() <= 4 and R.shape == (32, 30)
src = key['source'].values
cls = np.select([src == 'Iowa Steinway', src == 'K18 upright', src == 'Salamander (sampled)', src == 'soundfont',
                 src == 'Otey physical model'], ['Iowa', 'K18', 'Salamander', 'SoundFont', 'Otey'], 'UNKNOWN')
assert (cls != 'UNKNOWN').all()
J = (R >= 3).astype(float)
groups = {'Iowa': ['Iowa'], 'K18': ['K18'], 'Salamander': ['Salamander'], 'SoundFont': ['SoundFont'], 'Otey': ['Otey'],
          'Iowa+K18': ['Iowa', 'K18'], 'RECORDED': ['Iowa', 'K18', 'Salamander'], 'ALGORITHMIC': ['SoundFont', 'Otey']}
rows = []
for g, mem in groups.items():
    ci = np.isin(cls, mem)
    per = J[:, ci].mean(axis=1)
    d = per - 0.5
    r = {'class': g, 'clips': int(ci.sum()), 'rate': float(J[:, ci].mean()),
         'binom_p_greater': binomtest(int(J[:, ci].sum()), int(J[:, ci].size), 0.5, alternative='greater').pvalue}
    for zm in ('zsplit', 'wilcox', 'pratt'):
        try:
            r[f'wilcoxon_greater_{zm}'] = float(wilcoxon(d, alternative='greater', zero_method=zm).pvalue)
        except ValueError:
            r[f'wilcoxon_greater_{zm}'] = np.nan
    r['n_listeners_gt/eq/lt'] = f'{int((per > .5).sum())}/{int((per == .5).sum())}/{int((per < .5).sum())}'
    rows.append(r)
pc = pd.DataFrame(rows)
pv2 = pd.read_csv(os.path.join(DATA, 'listening', 'listening_per_class.csv'))
pd.set_option('display.width', 250)
print(pc.to_string())
print(pv2[['class', 'rate', 'binom_p_greater', 'wilcoxon_p_greater']].to_string())

rec = np.isin(cls, ['Iowa', 'K18', 'Salamander'])
alg = np.isin(cls, ['SoundFont', 'Otey'])
la = []
for l in range(R.shape[0]):
    yy = np.r_[np.zeros(rec.sum()), np.ones(alg.sum())]
    la.append(roc_auc_score(yy, np.r_[R[l, rec], R[l, alg]]))
la = np.array(la)
# cross-check with Mann-Whitney U / (n1 n2)
la2 = np.array([mannwhitneyu(R[l, alg], R[l, rec]).statistic / (alg.sum() * rec.sum()) for l in range(32)])
assert np.allclose(la, la2)
mr = R.mean(axis=0)
panel = roc_auc_score(alg[rec | alg].astype(int), mr[rec | alg])
pooled = roc_auc_score(np.r_[np.zeros(R[:, rec].size), np.ones(R[:, alg].size)], np.r_[R[:, rec].ravel(), R[:, alg].ravel()])
print(f'mean listener AUC recorded vs algorithmic = {la.mean():.4f} (median {np.median(la):.4f}; >0.5: {(la > .5).sum()}, '
      f'=0.5: {(la == .5).sum()}, <0.5: {(la < .5).sum()}); Wilcoxon(AUC-0.5>0, zsplit) p = '
      f'{wilcoxon(la - .5, alternative="greater", zero_method="zsplit").pvalue:.3g}')
print(f'panel AUC (per-clip mean rating) = {panel:.4f};  pooled AUC (all judgments) = {pooled:.4f}')
# panel ranking details
order = np.argsort(mr)
print('clips sorted by mean rating (class, mean):', [(cls[i], round(mr[i], 2)) for i in order])
pd.DataFrame({'listener': res['listener'], 'auc_rec_vs_alg': la}).to_csv(os.path.join(OUTD, 'listening_listener_auc.csv'), index=False)
pc.to_csv(os.path.join(OUTD, 'listening_per_class_auditor.csv'), index=False)
