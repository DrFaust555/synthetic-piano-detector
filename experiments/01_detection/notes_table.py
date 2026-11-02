"""Table 2, notes block: AUC [grouped-bootstrap CI] and share detected at 5 % false alarms on CCMusic for the
feature detector, the phaseogram network alone and their fusion, from the protocol v5b out-of-fold scores
(recompute_v5b/v5b/oof_v5b.csv: same notes and folds for the three detectors; clean notes).
    python experiments/01_detection/notes_table.py   -> results/01_detection/notes_cnn_fusion_table.md
"""
import os, sys
import numpy as np
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))   # repository root
from detector.paths import DATA, RESULTS, AUDIO_ROOT, results_dir   # noqa: E402
OUT = results_dir('01_detection')
import protocol.common as L
from protocol.cnn import universe

o = pd.read_csv(os.path.join(DATA, 'predictions', 'notes_oof.csv'))
mf = dict(zip(*[universe()[c] for c in ('file', 'mf_layer')]))
o['mf'] = o.file.map(mf).fillna(False).astype(bool)
cc = (o.source == 'ccmusic') & o.corpus
rows = [('SoundFont', (o.source == 'soundfont') & o.corpus), ('Otey', (o.source == 'otey') & o.corpus),
        ('both algorithmic', o.source.isin(['soundfont', 'otey']) & o.corpus),
        ('libraries, mf', (o.source == 'library') & o.usable & o.mf), ('libraries, all layers', (o.source == 'library') & o.usable)]
dets = [('features (XGBoost, 27)', 'xgb27_p'), ('network alone', 'cnn_p'), ('fusion', 'fus27_p')]
md = ['# Notes: features / network / fusion (v5b out-of-fold, clean notes)\n\nAUC vs CCMusic [95% CI] / detected at 5 % FA\n\n',
      '| source | ' + ' | '.join(d for d, _ in dets) + ' |\n|---|---|---|---|\n']
for name, m in rows:
    cells = []
    for _, c in dets:
        p = o[c].values
        t5 = np.quantile(p[cc], 0.95)
        y = np.r_[np.zeros(cc.sum()), np.ones(m.sum())]
        pp = np.r_[p[cc], p[m]]
        g = np.r_[o.loc[cc, 'sub'].values, o.loc[m, 'sub'].values]
        a = L.auc_fast(y, pp)
        star = o.loc[m, 'sub'].nunique() == 1
        lo, hi = L.grouped_boot_ci(y, pp, g)
        cells.append(f'{a:.3f}{"*" if star else f" [{lo:.2f}, {hi:.2f}]"} / {np.mean(p[m] > t5):.0%}')
    md.append(f'| {name} (n={m.sum()}) | ' + ' | '.join(cells) + ' |\n')
open(os.path.join(OUT, 'notes_cnn_fusion_table.md'), 'w', encoding='utf-8').write(''.join(md))
print(''.join(md))
