"""Feature detector (XGBoost) over ten random group-to-fold assignments (seeds 0-9, the protocol's greedy rule
pv2_lib.assign_folds), final settings: (a) isolated notes, 27 features, T2, library notes never trained;
(b) performances, 47 features, setting B (digital instruments in training; SC-55 and GP-3 removed), SFZ sampled
play never trained.  Cells: median [min-max] over the seeds of AUC / detection at 5 % false alarms; seed 0 is the
assignment used for the single-assignment numbers.
    python experiments/01_detection/fold_assignments.py     -> results/01_detection/xgb_seed_sweep.{md,csv}
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
iso = df[df.kind.isin(['real_isolated', 'algo_single']) & ~df.sf_dup].reset_index(drop=True)
def _library_notes():
    libs = P.library_note_files()
    lf = L.extract_cached(list(libs['path']), 'library_notes', isolated=True)
    libs = pd.concat([libs.reset_index(drop=True), lf], axis=1)
    libs['grp'] = 'syn:lib:' + libs['library']
    libs['valid'] = (libs[list(L.FEATURE_COLS)].notna().sum(axis=1) >= 30) & (libs['error'].fillna('') == '')
    return libs[libs['valid']].reset_index(drop=True)


lib = _library_notes()
perf = df[df.kind.isin(PERF_KINDS) & ~df.kind.isin(['sound_module', 'digital_other'])].reset_index(drop=True)
sfz, _ = P._filter_valid(pd.read_csv(os.path.join(DATA, 'performances', 'sfz_perf_features.csv')))
sfz = sfz.reset_index(drop=True)
sfz['grp'] = 'syn:sfz:' + sfz['instrument']
mf = lib.mf_layer.astype(bool).values
cc = (iso.kind == 'real_isolated').values
rp = (perf.kind == 'real_performance').values
rows_a = [('SoundFont', (iso.subkind == 'soundfont_single').values), ('Otey', (iso.subkind == 'otey_single').values),
          ('algorithmic', (iso.kind == 'algo_single').values)]
rows_b = [('SoundFont', (perf.subkind == 'sf2_perf').values), ('Otey', (perf.subkind == 'otey_perf').values),
          ('Korg M1', (perf.kind == 'm1').values), ('Roland', (perf.kind == 'modelled_digital').values),
          ('sampled digital', (perf.kind == 'sampled_digital_performance').values), ('all non-acoustic', perf.y.values == 1)]
rec = []
for seed in range(10):
    fmap = L.assign_folds(df.loc[df.base, 'grp'], df.loc[~df.base, 'grp'], seed=seed)      # corpus (seed 0 = v3 folds)
    fsfz = L.assign_folds(df.loc[df.base, 'grp'], pd.concat([df.loc[~df.base, 'grp'], sfz['grp']]), seed=seed)  # SFZ only
    a = iso.assign(fold=iso.grp.map(fmap).astype(int))
    lf = lib.grp.map(fmap).astype(int).values
    r = V.run_cv3(a, V.FS3_ISO['all47'], 'T2', ext={'lib': lib}, ext_fold={'lib': lf})
    p, pl = r['own'], r['ext_own']['lib']
    t5 = np.quantile(p[cc], 0.95)
    for lab, m in rows_a:
        rec.append({'seed': seed, 'panel': 'isolated', 'source': lab, 'auc': V.cmp3(a.y.values, p, a.grp.values, cc, m)['auc'],
                    'det': np.mean(p[m] > t5)})
    for lab, sel in (('libraries, all layers*', np.ones(len(lib), bool)), ('libraries, mf*', mf)):
        yy = np.r_[np.zeros(cc.sum()), np.ones(sel.sum())]
        auc = V.cmp3(yy, np.r_[p[cc], pl[sel]], np.r_[a.grp.values[cc], lib.grp.values[sel]], yy == 0, yy == 1)['auc']
        rec.append({'seed': seed, 'panel': 'isolated', 'source': lab, 'auc': auc, 'det': np.mean(pl[sel] > t5)})
    b = perf.assign(fold=perf.grp.map(fmap).astype(int), trainable=True)
    r = V.run_cv3(b, list(L.FEATURE_COLS), 'T1', ext={'sfz': sfz}, ext_fold={'sfz': sfz.grp.map(fsfz).astype(int).values})
    p, ps = r['own'], r['ext_own']['sfz']
    t5 = np.quantile(p[rp], 0.95)
    for lab, m in rows_b:
        rec.append({'seed': seed, 'panel': 'performances', 'source': lab, 'auc': V.cmp3(b.y.values, p, b.grp.values, rp, m)['auc'],
                    'det': np.mean(p[m] > t5)})
    yy = np.r_[np.zeros(rp.sum()), np.ones(len(ps))]
    auc = V.cmp3(yy, np.r_[p[rp], ps], np.r_[b.grp.values[rp], sfz.grp.values], yy == 0, yy == 1)['auc']
    rec.append({'seed': seed, 'panel': 'performances', 'source': 'SFZ sampled play*', 'auc': auc, 'det': np.mean(ps > t5)})
    print('seed', seed, 'done', flush=True)
d = pd.DataFrame(rec)
d.to_csv(os.path.join(OUT, 'xgb_seed_sweep.csv'), index=False)
md = ['# Feature detector over ten group-to-fold assignments (median [min-max]; seed 0)\n\n',
      '| panel | source | AUC median [min-max] | AUC seed 0 | detected median [min-max] | detected seed 0 |\n|---|---|---|---|---|---|\n']
for (pn, src), g in d.groupby(['panel', 'source'], sort=False):
    s0 = g[g.seed == 0].iloc[0]
    md.append(f'| {pn} | {src} | {g.auc.median():.3f} [{g.auc.min():.3f}-{g.auc.max():.3f}] | {s0.auc:.3f} | '
              f'{g.det.median():.0%} [{g.det.min():.0%}-{g.det.max():.0%}] | {s0.det:.0%} |\n')
open(os.path.join(OUT, 'xgb_seed_sweep.md'), 'w', encoding='utf-8').write(''.join(md))
print(''.join(md))
