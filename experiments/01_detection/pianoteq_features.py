"""Table 3 (Sec. 4.5), provenance column on PERFORMANCES only: median P(synth) per source with the question-2
detector (setting B, 47 features, XGBoost, own fold).  Sources: real performances; Pianoteq (13 files, not in the
corpus: model of the fold assigned to its new group); Salamander = SFZ sampled play of MAESTRO excerpts (never
trained); Otey performance renders; Korg M1 piano patches (by name, as q2_m1_pianos_only.py) and all 30 patches.
Pianoteq features: extracted with PIANO_TRANSCRIBER=flux (cache of table3_provenance.py).
    python experiments/01_detection/pianoteq_features.py      -> results/01_detection/table3_provenance_perf.md
"""
import os, sys, re, glob
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

NON_PIANO = re.compile(r'toy|vox|talk|phase|magic|res|stick', re.I)
df, _ = V.build_corpus_v3(log=lambda *a: None)
d = df[df.kind.isin(PERF_KINDS) & ~df.kind.isin(['sound_module', 'digital_other'])].reset_index(drop=True)
cols = list(L.FEATURE_COLS)
rp = (d.kind == 'real_performance').values
pt = L.extract_cached([os.path.join(ROOT, 'audio', 'pianoteq', f'pianoteq_modartt9_{k:02d}.wav') for k in range(1, 14)],
                      'table3_perf_flux', workers=1)
pt['grp'] = 'syn:pianoteq'
sfz, _ = P._filter_valid(pd.read_csv(os.path.join(DATA, 'performances', 'sfz_perf_features.csv')))
sfz = sfz.reset_index(drop=True)
sfz['grp'] = 'syn:sfz:' + sfz['instrument']
allg = pd.concat([df.loc[~df.base, 'grp'], sfz['grp']])          # as q2_final: SFZ folds unchanged
fmap = L.assign_folds(df.loc[df.base, 'grp'], allg)
load = pd.concat([df.loc[df.base, 'grp'], allg]).map(fmap).value_counts().reindex(range(L.N_FOLDS), fill_value=0)
fmap['syn:pianoteq'] = int(load.idxmin())                         # Pianoteq: lightest fold, nothing else moves
r = V.run_cv3(d.assign(trainable=True), cols, 'T1', ext={'pt': pt, 'sfz': sfz},
              ext_fold={'pt': pt.grp.map(fmap).astype(int).values, 'sfz': sfz.grp.map(fmap).astype(int).values})
p, ppt, psfz = r['own'], r['ext_own']['pt'], r['ext_own']['sfz']
t5 = float(np.quantile(p[rp], 0.95))
m1 = (d.kind == 'm1').values
m1p = m1 & ~d.file.str.replace('m1_patches/m1_', '', regex=False).str.contains(NON_PIANO).values
rows = [('Real acoustic (all real performances)', p[rp], None),
        ('Real acoustic (Boesendorfer 280VC, 1 file)', p[(d.file == '42_bosendorfer_280vc_97.wav').values], None),
        ('Pianoteq', ppt, pt.grp.values)]
for inst in sorted(sfz.instrument.unique()):
    k = (sfz.instrument == inst).values
    rows.append((f'SFZ sampled play: {inst}', psfz[k], sfz.grp.values[k]))
otey = (d.subkind == 'otey_perf').values
rows += [('SFZ sampled play: all four', psfz, sfz.grp.values), ('Otey', p[otey], d.grp.values[otey]),
         ('Korg M1, piano patches (22)', p[m1p], d.grp.values[m1p]),
         ('Korg M1, not piano imitations (8)', p[m1 & ~m1p], d.grp.values[m1 & ~m1p]), ('Korg M1, all 30 patches', p[m1], d.grp.values[m1])]


def auc_ci(v, g):
    """AUC against all real performances; grouped stratified bootstrap (* = one synthetic group)."""
    from sklearn.metrics import roc_auc_score
    if g is None:
        return ''
    y, pp = np.r_[np.zeros(rp.sum()), np.ones(len(v))], np.r_[p[rp], v]
    lo, hi = L.grouped_boot_ci(y, pp, np.r_[d.grp.values[rp], g])
    return f'{roc_auc_score(y, pp):.3f}{"*" if len(set(g)) == 1 else ""} [{lo:.3f}, {hi:.3f}]'


md = [f'# Table 3 provenance on performances (question 2, setting B)\n\nThreshold at 5 % false alarms on real performances: P > {t5:.3f}\n'
      'AUC against all real performances; * = one synthetic group (real-side-only interval)\n\n'
      '| Source | n | median P(synth) | IQR | detected | AUC vs real [95% CI] |\n|---|---|---|---|---|---|\n']
for name, v, g in rows:
    md.append(f'| {name} | {len(v)} | {np.median(v):.3f} | {np.quantile(v, .25):.3f}-{np.quantile(v, .75):.3f} | '
              f'{np.mean(v > t5):.0%} | {auc_ci(v, g)} |\n')
open(os.path.join(OUT, 'table3_provenance_perf.md'), 'w', encoding='utf-8').write(''.join(md))
print(''.join(md))
