"""Table 6, column P(synthetic): the performance fusion detector (phaseogram CNN + 47 features, digital pianos in
training), each performance scored by the model of its own fold (data/predictions/perf/fusion_perf_f*.npz); Pianoteq,
never in training, by the model of its fold (data/predictions/pianoteq_cnn_fusion.csv).  Threshold = 5 % false alarms
on the real performances.  The median of a source is its Table 6 entry.

Paper numbers (Table 6): Boesendorfer 280VC 0.03, Pianoteq 0.94, Salamander SFZ play 0.41, Otey 0.99,
Korg M1 piano presets 0.97.

    python experiments/05_provenance_and_quality/table6_provenance.py
        -> results/05_provenance_and_quality/table6_provenance.md
"""
import os, sys
import numpy as np
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))   # repository root
from detector.paths import DATA, RESULTS, AUDIO_ROOT, results_dir   # noqa: E402
OUT = results_dir('05_provenance_and_quality')
import protocol.common as L

meta = pd.concat([pd.read_csv(os.path.join(DATA, 'predictions', 'perf_meta.csv')),
                  pd.read_csv(os.path.join(DATA, 'predictions', 'perf_meta_sfz.csv'))], ignore_index=True)
fold = meta.fold.values.astype(int)
p = np.full(len(meta), np.nan)
for k in range(5):
    p[fold == k] = np.load(os.path.join(DATA, 'predictions', 'perf', f'fusion_perf_f{k}.npz'))['p'][fold == k]
kind, sub, f = meta.kind.values, meta.subkind.values, meta.file
rp = kind == 'real_performance'
t5 = float(np.quantile(p[rp], 0.95))
toy = f.str.replace('m1_patches/m1_', '', regex=False).str.contains(r'(?i)toy|vox|talk|phase|magic|res|stick').values
pt = pd.read_csv(os.path.join(DATA, 'predictions', 'pianoteq_cnn_fusion.csv'))
rows = [('Real acoustic (all real performances)', p[rp], None),
        ('Real acoustic (Boesendorfer 280VC, 1 file)', p[(f == '42_bosendorfer_280vc_97.wav').values], None),
        ('Pianoteq', pt.p_fusion_perf.values, np.array(['syn:pianoteq'] * len(pt))),
        ('SFZ sampled play: Salamander', p[sub == 'sfz:Salamander'], meta.grp.values[sub == 'sfz:Salamander']),
        ('SFZ sampled play: all four', p[kind == 'sfz_performance'], meta.grp.values[kind == 'sfz_performance']),
        ('Otey', p[sub == 'otey_perf'], meta.grp.values[sub == 'otey_perf']),
        ('Korg M1, piano presets (22)', p[(kind == 'm1') & ~toy], meta.grp.values[(kind == 'm1') & ~toy])]
md = [f'# Table 6 provenance, fusion detector\n\nThreshold at 5 % false alarms on real performances: P > {t5:.3f}\n\n'
      '| Source | n | median P(synth) | IQR | detected | AUC vs real [95% CI] |\n|---|---|---|---|---|---|\n']
for name, v, g in rows:
    auc = ''
    if g is not None:
        y = np.r_[np.zeros(rp.sum()), np.ones(len(v))]
        pp = np.r_[p[rp], v]
        lo, hi = L.grouped_boot_ci(y, pp, np.r_[meta.grp.values[rp], g])
        auc = f'{L.auc_fast(y, pp):.3f}{"*" if len(set(g)) == 1 else ""} [{lo:.3f}, {hi:.3f}]'
    md.append(f'| {name} | {len(v)} | {np.median(v):.3f} | {np.quantile(v, .25):.3f}-{np.quantile(v, .75):.3f} | '
              f'{np.mean(v > t5):.0%} | {auc} |\n')
open(os.path.join(OUT, 'table6_provenance.md'), 'w', encoding='utf-8').write(''.join(md))
print(''.join(md))
