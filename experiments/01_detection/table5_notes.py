"""Table 5, single-note block: AUC [95% grouped-bootstrap CI] / share detected at 5 % false alarms, for the
feature detector, the phaseogram CNN and the fusion detector.

    python experiments/01_detection/table5_notes.py      -> results/01_detection/table5_notes.md

Inputs (out-of-fold scores, data/predictions/):
  notes_features_oof.csv  feature detector (XGBoost, 27 features): column of the feature detector for the sources
  notes_oof.csv           phaseogram CNN (cnn_p) and fusion (fus27_p), with the feature detector of the same run
                          (xgb27_p), which gives the "real pianos" row of the feature column
The real class is the 668 CCMusic notes; the threshold P_th of each detector is the 95th percentile of their scores.
"Real pianos": each of the six brand groups is scored against the other five real brands; the row gives the median
over the six groups of the AUC and of the share above P_th.
"""
import os
import sys
import numpy as np
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))   # repository root
from detector.paths import DATA, results_dir   # noqa: E402
import protocol.common as L                    # noqa: E402
from protocol.cnn import universe              # noqa: E402

OUT = results_dir('01_detection')


def cell(p, real, syn, g_real, g_syn, t5):
    y = np.r_[np.zeros(real.sum()), np.ones(syn.sum())]
    pp = np.r_[p[real], p[syn]]
    a = L.auc_fast(y, pp)
    if len(set(g_syn)) == 1:
        return f'{a:.3f} / {np.mean(p[syn] > t5):.0%}'
    lo, hi = L.grouped_boot_ci(y, pp, np.r_[g_real, g_syn])
    return f'{a:.3f} [{lo:.2f}, {hi:.2f}] / {np.mean(p[syn] > t5):.0%}'


def brands(p, g):
    t5 = np.quantile(p, 0.95)
    aucs, det = [], []
    for b in sorted(set(g)):
        m = g == b
        aucs.append(L.auc_fast(m.astype(int), p))
        det.append(np.mean(p[m] > t5))
    return f'{np.median(aucs):.3f} / {np.median(det):.0%}'


def main():
    mf = dict(zip(*[universe()[c] for c in ('file', 'mf_layer')]))
    # feature detector
    f = pd.read_csv(os.path.join(DATA, 'predictions', 'notes_features_oof.csv'))
    f['mf'] = f.file.map(mf).fillna(False).astype(bool)
    # CNN and fusion
    o = pd.read_csv(os.path.join(DATA, 'predictions', 'notes_oof.csv'))
    o['mf'] = o.file.map(mf).fillna(False).astype(bool)
    o = o[o.corpus | ((o.source == 'library') & o.usable)].reset_index(drop=True)

    dets = []
    p = f.P.values
    real = (f.src == 'CCMusic').values
    sel = {'SoundFont synthesizers (10)': (f.src == 'SoundFont').values, 'Otey physical model': (f.src == 'Otey').values,
           'Both algorithmic sources (SoundFont + Otey)': f.src.isin(['SoundFont', 'Otey']).values,
           'Sample libraries (4), mf layer': ((f.src == 'library') & f.mf).values}
    dets.append(('Feature detector', p, real, sel, f.grp.values,
                 o.loc[o.source == 'ccmusic', 'xgb27_p'].values, o.loc[o.source == 'ccmusic', 'sub'].values))
    real_o = (o.source == 'ccmusic').values
    sel_o = {'SoundFont synthesizers (10)': (o.source == 'soundfont').values, 'Otey physical model': (o.source == 'otey').values,
             'Both algorithmic sources (SoundFont + Otey)': o.source.isin(['soundfont', 'otey']).values,
             'Sample libraries (4), mf layer': ((o.source == 'library') & o.mf).values}
    for name, c in (('Phaseogram CNN', 'cnn_p'), ('Fusion detector', 'fus27_p')):
        dets.append((name, o[c].values, real_o, sel_o, o['sub'].values,
                     o.loc[real_o, c].values, o.loc[real_o, 'sub'].values))

    rows = {k: [k] for k in list(sel) + ['Real pianos (8 in 6 brand groups)']}
    head = ['Single notes']
    for name, p, real, sel_, grp, p_brand, g_brand in dets:
        t5 = np.quantile(p[real], 0.95)
        head.append(f'{name}, P_th = {t5:.3f}')
        for k, m in sel_.items():
            rows[k].append(cell(p, real, m, grp[real], grp[m], t5))
        rows['Real pianos (8 in 6 brand groups)'].append(brands(p_brand, g_brand))
    md = ['# Table 5, single notes: AUC [95% CI] / share detected at 5 % false alarms\n\n',
          '| ' + ' | '.join(head) + ' |\n', '|---' * len(head) + '|\n']
    md += ['| ' + ' | '.join(r) + ' |\n' for r in rows.values()]
    md.append('\nSources without an interval form a single instrument group. "Real pianos": median over the six brand '
              'groups, each scored against the other five.\n')
    open(os.path.join(OUT, 'table5_notes.md'), 'w', encoding='utf-8').write(''.join(md))
    print(''.join(md))


if __name__ == '__main__':
    main()
