"""Question 2, setting B, without the Korg M1 patches that by NAME are not piano imitations (toy piano, voice,
talk, phaser, 'magic', resonant, stick).  The rule uses the patch names only, never the detector's scores.  The
patches are removed from the corpus (training and scoring); everything else as q2_final.py.
    python experiments/01_detection/m1_piano_presets.py      -> results/01_detection/q2_m1_pianos_only.md
"""
import os, sys, re
import numpy as np
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))   # repository root
from detector.paths import DATA, RESULTS, AUDIO_ROOT, results_dir   # noqa: E402
OUT = results_dir('01_detection')
ROOT = AUDIO_ROOT
import protocol.common as L
import protocol.corpus as V
from protocol.corpus import PERF_KINDS

NON_PIANO = re.compile(r'toy|vox|talk|phase|magic|res|stick', re.I)
df, _ = V.build_corpus_v3(log=lambda *a: None)
d0 = df[df.kind.isin(PERF_KINDS) & ~df.kind.isin(['sound_module', 'digital_other'])].reset_index(drop=True)
drop = (d0.kind == 'm1') & d0.file.str.replace('m1_patches/m1_', '', regex=False).str.contains(NON_PIANO)
md = [f'# Question 2, setting B, Korg M1 without non-piano patches\n\nDropped by name ({int(drop.sum())}): '
      + ', '.join(d0.file[drop].str.replace('m1_patches/m1_', '', regex=False)) + '\n\n'
      '| corpus | SoundFont | Otey | Korg M1 | Roland | sampled digital | all non-acoustic |\n|---|---|---|---|---|---|---|\n']
for name, d in (('all 30 M1 patches', d0), (f'M1 piano patches only ({int((d0.kind == "m1").sum() - drop.sum())})', d0[~drop].reset_index(drop=True))):
    k, sk, y, grp = d.kind.values, d.subkind.values, d.y.values, d.grp.values
    rp = k == 'real_performance'
    p = V.run_cv3(d.assign(trainable=True), list(L.FEATURE_COLS), 'T1')['own']
    t5 = np.quantile(p[rp], 0.95)
    rows = [sk == 'sf2_perf', sk == 'otey_perf', k == 'm1', k == 'modelled_digital', k == 'sampled_digital_performance', y == 1]
    md.append(f'| {name} | ' + ' | '.join(f'{V.fmt3(V.cmp3(y, p, grp, rp, m), det=False)} / {np.mean(p[m] > t5):.0%}' for m in rows) + ' |\n')
open(os.path.join(OUT, 'q2_m1_pianos_only.md'), 'w', encoding='utf-8').write(''.join(md))
print(''.join(md))
