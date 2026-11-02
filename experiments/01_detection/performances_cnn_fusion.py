"""Table 2, performance block: AUC [grouped-bootstrap CI] and share detected at 5 % false alarms on the real
performances for the phaseogram network alone and its fusion with the 47 features, setting B (digital pianos in
training; SFZ sampled play never trained).  Fold predictions of cnn_iso/jobs_perf_cnn_dig_sfz (trained on a
corpus that still held the SC-55 and GP-3 files; those 3 files are left out of every row here, as in q2_final).
    python experiments/01_detection/performances_cnn_fusion.py   -> results/01_detection/perf_cnn_fusion_table.md
"""
import os, sys
import re
import numpy as np
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))   # repository root
from detector.paths import DATA, RESULTS, AUDIO_ROOT, results_dir   # noqa: E402
OUT = results_dir('01_detection')
import protocol.corpus as V

mp = pd.read_csv(os.path.join(DATA, 'predictions', 'perf_meta.csv'))
ms = pd.read_csv(os.path.join(DATA, 'predictions', 'perf_meta_sfz.csv'))
meta = pd.concat([mp, ms], ignore_index=True)
fold, y = meta.fold.values.astype(int), meta.y.values.astype(int)
df, _ = V.build_corpus_v3(log=lambda *a: None)
d = df.set_index('file').loc[mp.file].reset_index()
grp = np.r_[d.grp.values, ms.grp.values]
kind = np.r_[d.kind.values, ms.kind.values]
sub = np.r_[d.subkind.values, ms.subkind.values]
keep = ~np.isin(kind, ['sound_module', 'digital_other'])
lg = lambda p: np.log(np.clip(p, 1e-7, 1 - 1e-7) / (1 - np.clip(p, 1e-7, 1 - 1e-7)))
S = {}
for m in ('cnn_perf', 'fusion_perf'):
    p = np.full(len(meta), np.nan)
    for k in range(5):
        p[fold == k] = np.load(os.path.join(DATA, 'predictions', 'perf', f'{m}_f{k}.npz'))['p'][fold == k]
    S[m] = lg(p)
rp = keep & (kind == 'real_performance')
trained = keep & np.isin(kind, ['algo_performance', 'sampled_digital_performance', 'modelled_digital', 'm1'])
toy = np.r_[mp.file.str.replace('m1_patches/m1_', '', regex=False).str.contains(re.compile(r'toy|vox|talk|phase|magic|res|stick', re.I)).values,
            np.zeros(len(ms), bool)]
rows = [('SoundFont renders', sub == 'sf2_perf'), ('Otey renders', sub == 'otey_perf'), ('Korg M1', kind == 'm1'),
        ('Korg M1, piano presets', (kind == 'm1') & ~toy), ('Korg M1, not piano imitations', (kind == 'm1') & toy),
        ('Digital pianos, modeled', kind == 'modelled_digital'), ('Digital pianos, sampled', kind == 'sampled_digital_performance'),
        ('All non-acoustic', trained & ~rp), ('SFZ sampled play*', kind == 'sfz_performance')]
md = ['# Performances, setting B: network alone / fusion (AUC [95% CI] / detected at 5 % FA)\n\n',
      f'Real performances: {rp.sum()}\n\n| source | n | network alone | fusion (network + 47 features) |\n|---|---|---|---|\n']
for lab, msk in rows:
    msk = msk & keep
    cells = []
    for m in ('cnn_perf', 'fusion_perf'):
        sc = S[m]
        t5 = np.quantile(sc[rp], 0.95)
        cells.append(f'{V.fmt3(V.cmp3(y, sc, grp, rp, msk), det=False)} / {np.mean(sc[msk] > t5):.0%}')
    md.append(f'| {lab} | {msk.sum()} | ' + ' | '.join(cells) + ' |\n')
open(os.path.join(OUT, 'perf_cnn_fusion_table.md'), 'w', encoding='utf-8').write(''.join(md))
print(''.join(md))
