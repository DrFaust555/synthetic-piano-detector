"""Sec. 4.2: the sample libraries fall inside the range of the real pianos (fusion detector, single notes).

Paper numbers:
  - each library (mf layer) scored against all real (CCMusic) notes: AUC 0.48 Iowa, 0.58 Salamander, 0.58 K18,
    0.59 FreePats;
  - each of the six real brand groups scored against the other five real brands: 0.26 Steinway, 0.31 Hsinghai,
    0.45 Young Chang, 0.57 Yamaha, 0.68 Kawai, 0.71 Pearl River.

Input: data/predictions/notes_oof.csv, column fus27_p (fusion of the phaseogram CNN and the 27-feature detector,
out-of-fold); the mf flag of a library note comes from protocol.cnn.universe().  The real class is the 668 CCMusic
notes of the corpus; brand group = column `sub` of the CCMusic rows.

    python experiments/02_sampled_vs_algorithmic/libraries_and_brands.py
        -> results/02_sampled_vs_algorithmic/libraries_and_brands.md
"""
import os
import sys
import numpy as np
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))   # repository root
from detector.paths import DATA, results_dir   # noqa: E402
import protocol.common as L                    # noqa: E402
import protocol.corpus as V                    # noqa: E402
from protocol.cnn import universe              # noqa: E402

OUT = results_dir('02_sampled_vs_algorithmic')
COL = 'fus27_p'


def main():
    mf = dict(zip(*[universe()[c] for c in ('file', 'mf_layer')]))
    o = pd.read_csv(os.path.join(DATA, 'predictions', 'notes_oof.csv'))
    o['mf'] = o.file.map(mf).fillna(False).astype(bool)
    p = o[COL].values
    cc = ((o.source == 'ccmusic') & o.corpus).values
    md = ['# Sample libraries and real brand groups (fusion detector, single notes)\n\n',
          f'Real class: {cc.sum()} CCMusic notes. Libraries: mf layer, scored against all real notes '
          '(one synthetic group each, so the interval resamples the real side only).\n\n',
          '| library (mf layer) | notes | AUC vs all real notes |\n|---|---|---|\n']
    for lib in ('Iowa', 'Salamander', 'K18', 'FreePats'):
        m = ((o.source == 'library') & o.usable & o.mf & (o['sub'] == lib)).values
        y = np.r_[np.zeros(cc.sum()), np.ones(m.sum())]
        r = V.cmp3(y, np.r_[p[cc], p[m]], np.r_[o['sub'].values[cc], o['sub'].values[m]], y == 0, y == 1)
        md.append(f'| {lib} | {m.sum()} | {V.fmt3(r, det=False)} |\n')
    md.append('\n| real brand group | notes | AUC vs the other five brands |\n|---|---|---|\n')
    r = o[cc]
    rows = []
    for b in sorted(r['sub'].unique()):
        m = (r['sub'] == b).values
        rows.append((L.auc_fast(m.astype(int), r[COL].values), b, m.sum()))
    for a, b, n in sorted(rows):
        md.append(f'| {b.replace("cc:", "")} | {n} | {a:.3f} |\n')
    open(os.path.join(OUT, 'libraries_and_brands.md'), 'w', encoding='utf-8').write(''.join(md))
    print(''.join(md))


if __name__ == '__main__':
    main()
