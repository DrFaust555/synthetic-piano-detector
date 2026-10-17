"""Protocol v4, isolated regime -- ONE command for every number in v4/RESULTS_v4.md.

    cd <project root>
    python Stattia/revision/analysis/v4/run_v4.py

(xgboost 3.2.0 must be importable; MAPS notes are fetched once by v4/get_maps_isol.py, which this script
calls if audio/_maps_isol/ is missing.)

Recorded-note class (label 0): CCMusic 8 pianos + Salamander, Iowa, K18, FreePats + MAPS ENSTDkCl/ENSTDkAm
ISOL/NO S0 notes (one real Yamaha Disklavier, two rooms -> ONE group).  Synthetic (label 1): 10 SoundFont
instruments (velocity copies removed) + Otey.  One group per piano / library / Disklavier / SoundFont / Otey.
Grouped CV: v2 greedy GroupKFold rule (5 folds, largest group first, ties in a seed-permuted order), seeds
0-9; plus leave-one-group-out (LOGO).  XGBoost with the protocol hyper-parameters (pv2_lib.make_model),
scale_pos_weight = n_recorded / n_synthetic of the whole training population, own-fold scoring.
Room augmentation (training only): each training note, with probability 0.5, is replaced by one of two
pre-extracted channel copies (random MIT-survey IR + white noise, SNR ~ U(30, 50) dB).  Tests: unaltered
notes ("clean") and every test note through a random channel copy ("channel").
Writes only into Stattia/revision/analysis/v4/.
"""
from __future__ import annotations

import os
import sys
import json
import time
import platform
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import v4_lib as W                         # noqa: E402
import pv2_lib as L                        # noqa: E402
import v3_lib as V                         # noqa: E402
from pv2_lib import FEATURE_COLS           # noqa: E402

N_SEEDS = 10
VARIANTS = ('all', 'mf')                   # recorded libraries / MAPS: all layers, or mf (MAPS 'M') only
FSETS = ('all47', 'med16')
AUGS = ('none', 'aug')
REC_ORDER = ['Hsinghai', 'Kawai', 'Kawai-G', 'PearlRiver', 'Steinway', 'Steinway-T', 'Yamaha', 'YoungChang',
             'Salamander', 'Iowa', 'K18', 'FreePats', 'MAPS ENSTDkCl', 'MAPS ENSTDkAm']
COLL = {'CCMusic': 'CCMusic (8 pianos)', 'Salamander': 'Salamander', 'Iowa': 'Iowa', 'K18': 'K18',
        'FreePats': 'FreePats', 'MAPS': 'MAPS Disklavier (both rooms)'}


# ----------------------------------------------------------------------------- worker
def cv_job(job):
    """one grouped CV; returns own-fold P for clean and channel test features."""
    import warnings
    warnings.simplefilter('ignore')
    Xo, Xa, y, fold = job['Xo'], job['Xa'], job['y'], job['fold']
    n = len(y)
    use_aug = job['aug'] == 'aug'
    Xtr_all = Xo.copy()
    if use_aug:
        m, c = job['mask'], job['copy']
        for k in range(len(Xa)):
            sel = m & (c == k) & job['aug_ok'][k]
            Xtr_all[sel] = Xa[k][sel]
    Xch = Xo.copy()
    for k in range(len(Xa)):
        sel = (job['tcopy'] == k) & job['aug_ok'][k]
        Xch[sel] = Xa[k][sel]
    spw = float(np.sum(y == 0)) / max(1, int(np.sum(y == 1)))
    pc, pch = np.full(n, np.nan), np.full(n, np.nan)
    for f in np.unique(fold):
        tr, te = fold != f, fold == f
        _, (a, b) = L.fit_predict('xgb', Xtr_all[tr], y[tr], [Xo[te], Xch[te]], spw=spw)
        pc[te], pch[te] = a, b
    return {k: job[k] for k in ('variant', 'fs', 'aug', 'seed', 'scheme')} | {'P_clean': pc, 'P_chan': pch}


def coll_job(job):
    """collection signature: target collection (1) vs CCMusic (0), nobody labelled synthetic."""
    import warnings
    from sklearn.model_selection import GroupKFold
    warnings.simplefilter('ignore')
    X, y, g = job['X'], job['y'], job['g']
    spw = float(np.sum(y == 0)) / max(1, int(np.sum(y == 1)))
    p = np.full(len(y), np.nan)
    for tr, te in GroupKFold(n_splits=5).split(X, y, g):
        _, (a,) = L.fit_predict('xgb', X[tr], y[tr], [X[te]], spw=spw)
        p[te] = a
    return {k: job[k] for k in ('target', 'fs', 'feat')} | {'auc': L.auc_fast(y, p), 'n1': int(y.sum()),
                                                            'n0': int((y == 0).sum())}


# ----------------------------------------------------------------------------- metrics
def metrics(sub, P):
    y = sub['y'].values
    src, inst = sub['src'].values, sub['inst'].values
    rec = y == 0
    out = {}
    for nm, sm in (('SoundFont', src == 'SoundFont'), ('Otey', src == 'Otey'), ('algorithmic', y == 1)):
        yy = np.r_[np.zeros(rec.sum()), np.ones(sm.sum())]
        out[f'auc:rec:{nm}'] = L.auc_fast(yy, np.r_[P[rec], P[sm]])
        cc = src == 'CCMusic'
        out[f'auc:cc:{nm}'] = L.auc_fast(np.r_[np.zeros(cc.sum()), np.ones(sm.sum())], np.r_[P[cc], P[sm]])
        out[f'det:{nm}'] = float(np.mean(P[sm] > 0.5))
    out['fa:rec'] = float(np.mean(P[rec] > 0.5))
    alg = y == 1
    for i in REC_ORDER:
        m = inst == i
        if m.any():
            out[f'fa:{i}'] = float(np.mean(P[m] > 0.5))
            out[f'meanP:{i}'] = float(np.mean(P[m]))
            out[f'aucalg:{i}'] = L.auc_fast(np.r_[np.zeros(m.sum()), np.ones(alg.sum())], np.r_[P[m], P[alg]])
    coll = np.where(src == 'MAPS', 'MAPS', np.where(src == 'library', inst, src))
    for c in COLL:
        m = coll == c
        if m.any():
            out[f'fa:coll:{c}'] = float(np.mean(P[m] > 0.5))
    cc = src == 'CCMusic'
    for c in ('Salamander', 'Iowa', 'K18', 'FreePats', 'MAPS'):
        m = coll == c
        if m.any():   # P(collection scored more synthetic than a CCMusic note); 0.5 = indistinguishable
            out[f'collscore:{c}'] = L.auc_fast(np.r_[np.zeros(cc.sum()), np.ones(m.sum())], np.r_[P[cc], P[m]])
    return out


def ci_rows(sub, P):
    """seed-0 grouped-bootstrap CIs (recorded groups and synthetic groups resampled separately)."""
    y, src, grp = sub['y'].values, sub['src'].values, sub['grp'].values
    res = {}
    for nm, sm in (('SoundFont', src == 'SoundFont'), ('Otey', src == 'Otey'), ('algorithmic', y == 1)):
        res[nm] = V.cmp3(y, P, grp, y == 0, sm)
    return res


# ----------------------------------------------------------------------------- formatting
def mm(vals, pct=False):
    v = np.asarray(vals, float)
    v = v[np.isfinite(v)]
    if not len(v):
        return 'n/a'
    if pct:
        return f'{100 * np.median(v):.1f}% [{100 * v.min():.1f}-{100 * v.max():.1f}]'
    return f'{np.median(v):.3f} [{v.min():.3f}-{v.max():.3f}]'


def main():
    t0 = time.time()
    xv = L.check_xgboost()
    if not os.path.isdir(W.MAPS_DIR):
        import get_maps_isol
        get_maps_isol.main()
    print('v4: corpus + features ...', flush=True)
    B = W.load_base().reset_index(drop=True)
    A = W.aug_features(B)
    Xo_all = {fs: B[W.FS[fs]].values.astype(float) for fs in FSETS}
    Xa_all = {fs: [A[k][W.FS[fs]].values.astype(float) for k in range(W.N_AUG_COPIES)] for fs in FSETS}
    aug_ok = [((A[k][FEATURE_COLS].notna().sum(axis=1) >= 30) & (A[k]['error'].fillna('').astype(str) == '')).values
              for k in range(W.N_AUG_COPIES)]
    n_aug_bad = [int((~a).sum()) for a in aug_ok]
    snr = pd.concat([A[k]['snr_db'] for k in range(W.N_AUG_COPIES)])
    n_ir = pd.concat([A[k]['ir'] for k in range(W.N_AUG_COPIES)]).nunique()

    variants = {'all': np.ones(len(B), bool),
                'mf': (B['y'] == 1).values | (B['src'] == 'CCMusic').values | B['layer_mf'].values.astype(bool)}
    # augmentation draws on the full table, so a note gets the same draw in every variant / feature set
    draws = {}
    for s in range(N_SEEDS):
        r = np.random.default_rng(1000 + s)
        draws[s] = (r.random(len(B)) < 0.5, r.integers(0, W.N_AUG_COPIES, len(B)), r.integers(0, W.N_AUG_COPIES, len(B)))
    jobs, subs, foldmaps = [], {}, {}
    for v in VARIANTS:
        sub = B[variants[v]].reset_index(drop=True)
        idx = np.where(variants[v])[0]
        subs[v] = sub
        for scheme, seeds in (('kfold', range(N_SEEDS)), ('logo', [0])):
            for s in seeds:
                if scheme == 'kfold':
                    fmap = L.assign_folds(sub['grp'], seed=s)
                    fold = sub['grp'].map(fmap).astype(int).values
                else:
                    fold = pd.factorize(sub['grp'], sort=True)[0]
                foldmaps[(v, scheme, s)] = fold
                for fs in FSETS:
                    for ag in AUGS:
                        m, c, tc = draws[s]
                        jobs.append({'variant': v, 'fs': fs, 'aug': ag, 'seed': s, 'scheme': scheme,
                                     'Xo': Xo_all[fs][idx], 'Xa': [x[idx] for x in Xa_all[fs]],
                                     'aug_ok': [a[idx] for a in aug_ok], 'y': sub['y'].values, 'fold': fold,
                                     'mask': m[idx], 'copy': c[idx], 'tcopy': tc[idx]})
    # collection-signature jobs (variant 'all'; nobody labelled synthetic)
    sub = subs['all']
    cc = (sub.src == 'CCMusic').values
    midi = sub['midi'].fillna(-1).astype(int).astype(str).values
    targets = [(c, (sub.inst == c).values) for c in ('Salamander', 'Iowa', 'K18', 'FreePats')] + \
              [('MAPS (both rooms)', (sub.src == 'MAPS').values), ('MAPS ENSTDkCl', (sub.inst == 'MAPS ENSTDkCl').values),
               ('MAPS ENSTDkAm', (sub.inst == 'MAPS ENSTDkAm').values)] + \
              [(f'CCMusic {p} (vs the other 7)', (sub.inst == p).values) for p in REC_ORDER[:8]] + \
              [(f'SoundFont {i}', (sub.inst == i).values) for i in sorted(sub.loc[sub.src == 'SoundFont', 'inst'].unique())] + \
              [('Otey', (sub.src == 'Otey').values)]
    cjobs = []
    sidx = np.where(variants['all'])[0]
    for tname, tm in targets:
        neg = cc & ~tm
        sel = tm | neg
        g = np.where(tm, 'm:' + midi, sub['inst'].values)[sel]
        for fs in FSETS:
            for feat in ('clean', 'channel'):
                X = Xo_all[fs][sidx].copy()
                if feat == 'channel':          # every note (both classes) through channel copy 0
                    ok = aug_ok[0][sidx]
                    X[ok] = Xa_all[fs][0][sidx][ok]
                cjobs.append({'target': tname, 'fs': fs, 'feat': feat, 'X': X[sel], 'y': tm[sel].astype(int), 'g': g})
    # MAPS room vs room (same Disklavier): label = room, grouped by pitch
    mp = (sub.src == 'MAPS').values
    for fs in FSETS:
        for feat in ('clean', 'channel'):
            X = Xo_all[fs][sidx].copy()
            if feat == 'channel':
                ok = aug_ok[0][sidx]
                X[ok] = Xa_all[fs][0][sidx][ok]
            cjobs.append({'target': 'MAPS ENSTDkAm vs ENSTDkCl (same piano, two rooms)', 'fs': fs, 'feat': feat,
                          'X': X[mp], 'y': (sub.inst.values[mp] == 'MAPS ENSTDkAm').astype(int), 'g': 'm:' + midi[mp]})
    print(f'v4: {len(jobs)} CV runs + {len(cjobs)} collection runs ...', flush=True)
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=14) as ex:
        res = list(ex.map(cv_job, jobs, chunksize=1))
        cres = list(ex.map(coll_job, cjobs, chunksize=2))
    print(f'v4: models done ({time.time() - t0:.0f} s)', flush=True)

    # ---------------------------------------------------------------- collect
    rows = []
    seed0 = {}
    for r in res:
        sub = subs[r['variant']]
        for test in ('clean', 'chan'):
            P = r[f'P_{test}']
            rows.append({k: r[k] for k in ('variant', 'fs', 'aug', 'seed', 'scheme')} | {'test': test, **metrics(sub, P)})
            if r['seed'] == 0:
                seed0[(r['variant'], r['fs'], r['aug'], r['scheme'], test)] = P
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(W.OUT4, 'v4_runs.csv'), index=False, float_format='%.6f')
    C = pd.DataFrame(cres)
    C.to_csv(os.path.join(W.OUT4, 'v4_collection_signature.csv'), index=False, float_format='%.6f')
    sc = B[['file', 'src', 'inst', 'grp', 'y', 'layer_mf']].copy()
    for (v, fs, ag, scheme, test), P in seed0.items():
        if v == 'all':
            sc[f'P_{fs}_{ag}_{scheme}_{test}'] = P
    sc.to_csv(os.path.join(W.OUT4, 'v4_scores_seed0_all.csv'), index=False, float_format='%.6f')
    cis = {}
    for v in VARIANTS:
        for fs in FSETS:
            for ag in AUGS:
                for test in ('clean', 'chan'):
                    cis[(v, fs, ag, test)] = ci_rows(subs[v], seed0[(v, fs, ag, 'kfold', test)])

    def cell(v, fs, ag, test, key, scheme='kfold', pct=False):
        q = T[(T.variant == v) & (T.fs == fs) & (T.aug == ag) & (T.test == test) & (T.scheme == scheme)]
        if key not in q:
            return 'n/a'
        if scheme == 'logo':
            x = q[key].values[0]
            return f'{100 * x:.1f}%' if pct else f'{x:.3f}'
        return mm(q[key].values, pct)

    # v3 reference (out_v3/fold_seed_sweep.csv: CCMusic real class, libraries never trained)
    sw = pd.read_csv(os.path.join(W.ANA, 'out_v3', 'fold_seed_sweep.csv'))
    v3k = {'SoundFont': 'SoundFont single notes', 'Otey': 'Otey single notes', 'algorithmic': 'algorithmic (both)'}

    md = ['# RESULTS v4 -- isolated regime with an enlarged recorded-note class\n\n',
          f'Generated {time.strftime("%Y-%m-%d %H:%M")} by one command, from the project root:\n\n',
          '```\npython Stattia/revision/analysis/v4/run_v4.py\n```\n\n',
          f'Environment: Python {platform.python_version()}, numpy {np.__version__}, pandas {pd.__version__}, xgboost {xv} '
          '(n_jobs=1 per model, random_state=0; protocol hyper-parameters of pv2_lib.make_model). '
          f'Runtime {{RUNTIME}} s (feature extraction cached in `v4/cache/`). Scripts: `v4/run_v4.py`, `v4/v4_lib.py`, '
          '`v4/get_maps_isol.py`. Existing protocol files and outputs are read only.\n\n']
    n = B.groupby(['src', 'inst']).size()
    md.append('## 1. Data\n\n')
    md.append('**Recorded-note class (label 0)**, one CV group per instrument:\n\n')
    rr = []
    for i in REC_ORDER:
        m = B.inst == i
        rr.append([i, {'CCMusic': 'CCMusic', 'library': 'free library', 'MAPS': 'MAPS'}[B.loc[m, 'src'].iloc[0]],
                   int(m.sum()), int((m & B.layer_mf).sum()),
                   B.loc[m, 'grp'].iloc[0]])
    md.append(L.md_table(['instrument', 'collection', 'notes (all layers)', 'notes (mf variant)', 'CV group'], rr))
    md.append(f'\n**Synthetic (label 1)**: {int((B.src == "SoundFont").sum())} SoundFont notes of '
              f'{B.loc[B.src == "SoundFont", "inst"].nunique()} instruments (velocity copies removed, as v3; one group '
              f'each) + {int((B.src == "Otey").sum())} Otey notes (one group). Features of CCMusic / SoundFont / Otey = the '
              'stored v3 features (quality/cache), libraries = the v2 extraction cache (out/cache/library_notes.csv, read '
              'only), MAPS = the same 47-feature pipeline (v4/cache/maps_orig.csv). Check: the v3 seed-0 CCMusic-vs-SoundFont '
              'AUC (0.937) is reproduced exactly from these inputs with the v3 folds.\n\n')
    md.append('**MAPS** (Emiya, Bertin & Badeau, MAPS -- MIDI Aligned Piano Sounds v0.4, Zenodo '
              'doi:10.5281/zenodo.18160555; licence Creative Commons BY-NC-SA 2.0 FR, non-commercial research use with '
              'attribution): only the two REAL Disklavier subsets ENSTDkCl (close mics) and ENSTDkAm (ambient mics), '
              'subfolder ISOL/NO (normal isolated notes), sustain pedal off (S0; the readme reports a pedal re-press bug '
              'in S1 isolated notes), all dynamics P/M/F ("mf variant" = M). The software-piano subsets (AkPn*, Sptk*, '
              'Stbg*) were not downloaded. Fetched by HTTP range requests from the two zip archives (about 90 MB per room '
              'instead of 2.6-2.7 GB) into `audio/_maps_isol/` with licence.txt and readme.txt. The two rooms are the same '
              'physical piano, so they form ONE CV group; false alarms are reported per room. Two variants of the '
              'recorded class: **all** = all velocity layers of the libraries and all MAPS dynamics; **mf** = mf layer of '
              'each library (Salamander v8, FreePats single layer), MAPS M only; CCMusic unchanged in both.\n\n')
    md.append(f'**Room augmentation**: two channel copies per note (all {len(B)} notes, both classes), each = the note '
              f'convolved with a random measured room IR of the MIT IR Survey (270 IRs, {n_ir} distinct drawn) + white '
              f'noise at SNR ~ U(30, 50) dB re the RMS of the first second after onset (drawn SNR median '
              f'{np.nanmedian(snr):.1f} dB); deterministic per note (RNG = crc32 of "v4aug<k>:" + path). Features '
              f're-extracted for every copy (cached, `v4/cache/iso_aug.csv`; extraction failures / <30 features: '
              f'{n_aug_bad} -- such rows fall back to the unaltered note). No subsampling was needed. Training: each '
              'training note is, with probability 0.5, replaced by one of its two copies (Bernoulli and copy drawn per '
              'note and seed, RNG 1000+seed). Tests: **clean** = unaltered test notes; **channel** = every test note '
              'through one random copy of the same channel model. The no-augmentation model is also scored on the '
              'channel test (mismatch control).\n\n')
    md.append(f'**Evaluation**: grouped CV over the {B.grp.nunique()} instrument groups (8 CCMusic pianos, 4 libraries, the Disklavier, '
              '10 SoundFonts, Otey) with the v2 greedy GroupKFold rule (5 folds; seeds 0-9 permute the tie order), every '
              'note scored by the model of its own fold (all notes are trainable, so this is the plain out-of-fold '
              'score); plus leave-one-group-out (LOGO, one fold per group). AUC = recorded notes vs the synthetic source; cells = '
              'median [min-max] over the 10 fold assignments; FA = share of recorded notes with P(synthetic) > 0.5. v3 '
              'reference = FINAL_NUMBERS.md (real class = CCMusic only, 6 brand groups, libraries never trained).\n\n')

    # ---------------------------------------------------------------- Table 1 compact
    for fs in FSETS:
        md.append(f'## 2{"a" if fs == "all47" else "b"}. Compact comparison, {fs}\n\n')
        rows = []
        for nm in ('SoundFont', 'Otey', 'algorithmic'):
            v3 = mm(sw[f'T7iso:{fs}:T2:{v3k[nm]}'].values)
            rows.append([nm, v3, cell('all', fs, 'none', 'clean', f'auc:rec:{nm}'), cell('mf', fs, 'none', 'clean', f'auc:rec:{nm}'),
                         cell('all', fs, 'aug', 'clean', f'auc:rec:{nm}'), cell('all', fs, 'aug', 'chan', f'auc:rec:{nm}'),
                         cell('all', fs, 'none', 'chan', f'auc:rec:{nm}'),
                         cell('all', fs, 'none', 'clean', f'auc:rec:{nm}', 'logo') + ' / ' +
                         cell('all', fs, 'aug', 'clean', f'auc:rec:{nm}', 'logo')])
        fa_v3 = mm(sw[f'T7iso:fa:{fs} T2'].values, pct=True)
        rows.append(['FA, all recorded notes', fa_v3 + ' (CCMusic)', cell('all', fs, 'none', 'clean', 'fa:rec', pct=True),
                     cell('mf', fs, 'none', 'clean', 'fa:rec', pct=True), cell('all', fs, 'aug', 'clean', 'fa:rec', pct=True),
                     cell('all', fs, 'aug', 'chan', 'fa:rec', pct=True), cell('all', fs, 'none', 'chan', 'fa:rec', pct=True),
                     cell('all', fs, 'none', 'clean', 'fa:rec', 'logo', True) + ' / ' +
                     cell('all', fs, 'aug', 'clean', 'fa:rec', 'logo', True)])
        md.append(L.md_table(['recorded vs', 'v3 (CCMusic only)', 'v4 all layers', 'v4 mf', 'v4+aug, clean test',
                              'v4+aug, channel test', 'v4 no aug, channel test (mismatch)', 'LOGO: v4 / v4+aug (clean)'], rows))
        rows = []
        for nm in ('SoundFont', 'Otey', 'algorithmic'):
            rows.append([nm, mm(sw[f'T7iso:{fs}:T2:{v3k[nm]}'].values), cell('all', fs, 'none', 'clean', f'auc:cc:{nm}'),
                         cell('mf', fs, 'none', 'clean', f'auc:cc:{nm}'), cell('all', fs, 'aug', 'clean', f'auc:cc:{nm}'),
                         cell('all', fs, 'aug', 'chan', f'auc:cc:{nm}')])
        md.append('\nSame models, real side restricted to the CCMusic notes (directly comparable with v3):\n\n')
        md.append(L.md_table(['CCMusic vs', 'v3', 'v4 all layers', 'v4 mf', 'v4+aug, clean', 'v4+aug, channel'], rows))
        rows = []
        for nm in ('SoundFont', 'Otey', 'algorithmic'):
            cells = [nm]
            for v, ag, test in (('all', 'none', 'clean'), ('mf', 'none', 'clean'), ('all', 'aug', 'clean'), ('all', 'aug', 'chan')):
                r = cis[(v, fs, ag, test)][nm]
                cells.append(V.fmt3(r))
            rows.append(cells)
        md.append('\nSeed 0 with grouped-bootstrap 95% CI (recorded groups and synthetic groups resampled separately; '
                  '"real-side only" = one synthetic group); det = share of the synthetic notes with P > 0.5:\n\n')
        md.append(L.md_table(['recorded vs', 'v4 all layers', 'v4 mf', 'v4+aug, clean', 'v4+aug, channel'], rows))
        md.append('\n')

    # ---------------------------------------------------------------- per-instrument false alarms
    md.append('## 3. Per recorded instrument: false-alarm rate (P > 0.5), median [min-max] over seeds 0-9\n\n')
    rows = []
    for i in REC_ORDER:
        rows.append([i] + [cell(v, fs, ag, test, f'fa:{i}', pct=True) for v, fs, ag, test in
                           (('all', 'all47', 'none', 'clean'), ('mf', 'all47', 'none', 'clean'), ('all', 'med16', 'none', 'clean'),
                            ('all', 'all47', 'aug', 'clean'), ('all', 'all47', 'aug', 'chan'), ('all', 'all47', 'none', 'chan'))]
                    + [cell('all', 'all47', 'none', 'clean', f'fa:{i}', 'logo', True)])
    md.append(L.md_table(['recorded instrument', 'all47', 'all47, mf variant', 'med16', 'all47 +aug, clean',
                          'all47 +aug, channel', 'all47 no aug, channel', 'all47 LOGO'], rows))
    md.append('\nPer recorded instrument: mean P(synthetic) and AUC of that instrument alone vs all algorithmic notes '
              '(all47, all layers, no augmentation / with augmentation, clean test):\n\n')
    rows = []
    for i in REC_ORDER:
        rows.append([i, cell('all', 'all47', 'none', 'clean', f'meanP:{i}'), cell('all', 'all47', 'none', 'clean', f'aucalg:{i}'),
                     cell('all', 'all47', 'aug', 'clean', f'meanP:{i}'), cell('all', 'all47', 'aug', 'clean', f'aucalg:{i}')])
    md.append(L.md_table(['recorded instrument', 'mean P', 'AUC vs algorithmic', 'mean P, +aug', 'AUC vs algorithmic, +aug'], rows))

    # ---------------------------------------------------------------- collection signatures
    md.append('\n## 4. Are recorded collections distinguishable among themselves?\n\n')
    md.append('**(a) Through the detector**: AUC that the detector scores a collection\'s notes as MORE synthetic than '
              'CCMusic notes (both labelled recorded; 0.5 = the detector treats them alike). Median [min-max] over seeds:\n\n')
    rows = []
    for c in ('Salamander', 'Iowa', 'K18', 'FreePats', 'MAPS'):
        rows.append([COLL[c]] + [cell(v, fs, ag, test, f'collscore:{c}') for v, fs, ag, test in
                                 (('all', 'all47', 'none', 'clean'), ('all', 'med16', 'none', 'clean'),
                                  ('all', 'all47', 'aug', 'clean'), ('all', 'all47', 'aug', 'chan'))])
    md.append(L.md_table(['collection vs CCMusic', 'all47', 'med16', 'all47 +aug, clean', 'all47 +aug, channel'], rows))
    md.append('\n**(b) Collection-signature classifier** (nobody labelled synthetic): XGBoost, same hyper-parameters, '
              'trained to tell a target collection (1) from the CCMusic notes (0); 5-fold GroupKFold with CCMusic grouped '
              'by piano (unseen pianos in each test fold) and the target grouped by nominal MIDI pitch (unseen pitches); '
              'pooled out-of-fold AUC. Reference rows: each CCMusic piano against the other seven (same collection and '
              'channel, different piano), each SoundFont and Otey against CCMusic (a synthetic source through the same '
              'test), and the two MAPS rooms against each other (same piano, different room). "channel" = every note of '
              'both classes replaced by its channel copy 0 (random IR + noise).\n\n')
    rows = []
    for t in C['target'].unique():
        q = C[C.target == t]
        g = lambda fs, f: q[(q.fs == fs) & (q.feat == f)]   # noqa: E731
        rows.append([t, f"{int(g('all47', 'clean').n1.iloc[0])} / {int(g('all47', 'clean').n0.iloc[0])}"] +
                    [f"{g(fs, f).auc.iloc[0]:.3f}" for fs in FSETS for f in ('clean', 'channel')])
    md.append(L.md_table(['target (1) vs CCMusic (0)', 'n target / n reference', 'all47 clean', 'all47 channel',
                          'med16 clean', 'med16 channel'], rows))
    sumc = {}
    for lab, pref in (('free libraries', ('Salamander', 'Iowa', 'K18', 'FreePats')), ('MAPS (both rooms)', ('MAPS (both rooms)',)),
                      ('CCMusic piano vs other 7', ('CCMusic ',)), ('SoundFonts', ('SoundFont ',)), ('Otey', ('Otey',))):
        q = C[C.target.apply(lambda s: any(s == p or (p.endswith(' ') and s.startswith(p)) for p in pref))]
        sumc[lab] = {f'{fs}:{f}': float(q[(q.fs == fs) & (q.feat == f)].auc.median()) for fs in FSETS for f in ('clean', 'channel')}
    md.append('\nMedians by kind: ' + '; '.join(f"{k}: all47 {v['all47:clean']:.3f} (channel {v['all47:channel']:.3f}), "
                                               f"med16 {v['med16:clean']:.3f}" for k, v in sumc.items()) + '.\n')
    md.append('\n**Reading notes.** (i) A recorded group that fills a fold alone is trained on exactly the same data in every '
              'fold assignment, so its row can show zero spread over seeds (Salamander, 480 notes, all-layers variant). '
              '(ii) Section 4a uses the detector trained with every collection labelled recorded, so values far from 0.5 '
              'mean the detector still ranks one recorded collection as more (or less) synthetic-looking than CCMusic. '
              '(iii) Section 4b is a different question -- can ANY classifier tell a collection apart from CCMusic? -- '
              'and its scale is set by the reference rows: a different CCMusic piano through the same channel, and the '
              'same Disklavier in a different room, are themselves separable well above 0.5, so a library\'s AUC must be '
              'read against those rows, not against 0.5.\n')
    md.append('\n## 5. Files\n\n`v4_runs.csv` (every CV run x test: all metrics), `v4_collection_signature.csv`, '
              '`v4_scores_seed0_all.csv` (per-note P, seed 0, all-layers variant), `key_numbers_v4.json`, `cache/` '
              '(maps_orig.csv, iso_aug.csv).\n')
    key = {'compact': {f'{v}|{fs}|{ag}|{test}|{scheme}|{k}': cell(v, fs, ag, test, k, scheme)
                       for v in VARIANTS for fs in FSETS for ag in AUGS for test in ('clean', 'chan')
                       for scheme in ('kfold', 'logo') for k in ('auc:rec:SoundFont', 'auc:rec:Otey', 'auc:rec:algorithmic', 'fa:rec')},
           'collection_medians': sumc, 'n_aug_fallback': n_aug_bad}
    rt = time.time() - t0
    key['runtime_s'] = rt
    json.dump(key, open(os.path.join(W.OUT4, 'key_numbers_v4.json'), 'w'), indent=1)
    text = ''.join(md).replace('{RUNTIME}', f'{rt:.0f}')
    open(os.path.join(W.OUT4, 'RESULTS_v4.md'), 'w', encoding='utf-8').write(text)
    print(f'wrote {os.path.join(W.OUT4, "RESULTS_v4.md")} ({rt:.0f} s)')


if __name__ == '__main__':
    main()
