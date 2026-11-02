"""Sec. 4.2: SoundFont notes whose analysis window plays a stored sample unaltered (no loop reached, played at its
own pitch) are separated as well as all SoundFont notes.

Paper numbers: feature detector AUC 0.961 (unaltered sample) against 0.955 (all SoundFont notes), vs CCMusic notes.

Inputs:
  data/predictions/notes_features_oof.csv     out-of-fold scores P of the feature detector (XGBoost, 27 features,
                                              single-model-of-own-fold scoring) for every isolated note
  data/corpus/soundfont_note_categories.csv   per rendered SoundFont note, from the SF2 data: sample root key,
                                              transposition, loop mode and loop start; cat_orig = category of the
                                              analysis window (unaltered sample / transposed only / looped only /
                                              transposed + looped)

    python experiments/02_sampled_vs_algorithmic/soundfont_unaltered_samples.py
        -> results/02_sampled_vs_algorithmic/soundfont_unaltered_samples.md
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

OUT = results_dir('02_sampled_vs_algorithmic')


def main():
    s = pd.read_csv(os.path.join(DATA, 'predictions', 'notes_features_oof.csv'))
    cats = pd.read_csv(os.path.join(DATA, 'corpus', 'soundfont_note_categories.csv'))
    cat = s['file'].map(cats.set_index('file')['cat_orig']).fillna('').values
    y = (s.src != 'CCMusic').values.astype(int)
    p, grp = s['P'].values, s['grp'].values
    rm = (s.src == 'CCMusic').values
    sf = (s.src == 'SoundFont').values
    md = ['# SoundFont notes by synthesis method (feature detector, AUC vs CCMusic notes)\n\n',
          f'{rm.sum()} CCMusic notes, {sf.sum()} SoundFont notes. Cell: AUC [95% CI]; det = share with P > 0.5 '
          '(n notes, number of SoundFonts).\n\n| SoundFont notes | AUC |\n|---|---|\n']
    rows = [('all', sf)] + [(c, sf & (cat == c)) for c in ('unaltered sample', 'transposed only', 'looped only',
                                                            'transposed + looped')]
    for name, sm in rows:
        r = V.cmp3(y, p, grp, rm, sm)
        md.append(f"| {name} | {V.fmt3(r)} (n={int(sm.sum())}, {r['g_syn']} SF) |\n")
    md.append(f'\nCCMusic false alarms (P > 0.5): {L.fpct(np.mean(p[rm] > 0.5))}.\n')
    open(os.path.join(OUT, 'soundfont_unaltered_samples.md'), 'w', encoding='utf-8').write(''.join(md))
    print(''.join(md))


if __name__ == '__main__':
    main()
