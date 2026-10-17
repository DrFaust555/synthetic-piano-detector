"""Protocol v2 -- format-specific detectors (section 9 of RESULTS_v2.md).

Called by run_all.py.  Same corpus, fixed folds, XGBoost (train_realness.py parameters, n_jobs=1),
all47 and med16, stratified grouped bootstrap (B = 2000, seed 0) as protocol_v2.py.

  9.1 PERFORMANCE-ONLY detector: trained on real performances (y=0) vs algorithmic performance renders
      (SoundFont + Otey, y=1) only (T2).  Sampled digital pianos, modelled digital pianos, Korg M1, SC-55 and
      GP-3 are scored zero-shot (mean of the 5 fold models).  T1 (secondary): those performance-format
      sample-based sources are added to training, labelled synthetic (out-of-fold by brand group).
  9.2 Feature-group ablation for real performances vs sampled digital pianos under the performance-only model.
  9.3 ISOLATED-ONLY detector (I-only): CCMusic notes vs SoundFont + Otey single notes (T2); library notes
      scored zero-shot.  T1: all 999 library note files added to isolated training, labelled synthetic
      (out-of-fold by library).
  9.4 Corpus-level detector with vs without the dry performance renders in training (effect on the 128
      sampled digital pianos), T2.
  9.5 Instrument-level tests: one value per instrument (median P of its notes; phaseogram irregularity
      from figures/drift_notes_all.csv), Mann-Whitney.
  9.6 Exact counts of zero IQR values and undefined Group B / C values in performance files.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

import pv2_lib as L
import protocol_v2 as P
from pv2_lib import FEATURE_COLS, MED16, FEATSETS, OUT, ROOT
from quality.pipeline import BASS_FEATURE_COLS, TUNING

IQR16 = [c for c in FEATURE_COLS if c.endswith('_iqr')]
GROUPS = {'only_A_med': MED16, 'only_A_iqr': IQR16, 'only_B': list(BASS_FEATURE_COLS), 'only_C': list(TUNING)}
GLABEL = {'all47': 'all 47', 'med16': '16 A medians', 'only_A_med': 'only A medians (16)', 'only_A_iqr': 'only A IQRs (16)',
          'only_B': 'only B (8)', 'only_C': 'only C (7)'}
PERF_KINDS = ['real_performance', 'algo_performance', 'sampled_digital_performance', 'modelled_digital', 'm1',
              'sound_module', 'digital_other']


def _cell(r, show_fa=False):
    if not np.isfinite(r.get('auc', np.nan)):
        return 'n/a'
    s = f"{L.fci(r)}; det {L.fpct(r['det_syn'])}; P {r['meanP_syn']:.2f}"
    return s + (f"; FA {L.fpct(r['fa_real'])}" if show_fa else '')


def perf_only(df, key, csv):
    d = df[df['kind'].isin(PERF_KINDS)].reset_index(drop=True)
    comps = {c[1]: c for c in P._comparisons(d)}
    ids = [i for i in ('P1', 'P1a', 'P1b', 'P2', 'P2a', 'P2d', 'P2d-line', 'P2d-mic', 'P2d-unk', 'P2d-lt', 'P2d-nolt',
                       'P3', 'P3d', 'P4', 'P5', 'P6') if i in comps and comps[i][4].any()]
    y, grp = d['y'].values, d['grp'].values
    runs = {}
    for fs in ('all47', 'med16'):
        for cond in ('T2', 'T1'):
            runs[(fs, cond)] = P.run_cv(d, 'xgb', fs, cond)[0]
    md = ['### 9.1 Performance-only detector\n\n']
    md.append(f"Training population (T2): {int((d.kind == 'real_performance').sum())} real performances vs "
              f"{int((d.kind == 'algo_performance').sum())} algorithmic performance renders "
              f"({d.loc[d.kind == 'algo_performance', 'grp'].nunique()} engines: 10 SoundFonts + Otey; each render shares "
              'the CV group of its engine). Every other performance-format source is scored zero-shot (mean of the 5 '
              'fold models): ' + ', '.join(f'{k} {int((d.kind == k).sum())}' for k in PERF_KINDS[2:]) +
              '. T1: those sources are also trained on, labelled synthetic (out-of-fold, brand-grouped folds). '
              'Cells: AUC [95% CI, digital pianos grouped by brand]; det = share of the synthetic files with P > 0.5; '
              'P = mean P(synthetic) of the synthetic files; FA = real performances with P > 0.5.\n\n')
    rows = []
    for cid in ids:
        _, _, label, rm, sm = comps[cid]
        cells = [cid, label.replace('real_performance vs ', ''), f'{int(sm.sum())}/{len(set(grp[sm]))}']
        for fs, cond in (('all47', 'T2'), ('med16', 'T2'), ('all47', 'T1'), ('med16', 'T1')):
            r = L.compare(y, runs[(fs, cond)], grp, rm, sm)
            cells.append(_cell(r))
            key[f'perfonly:{fs}:{cond}:{cid}'] = r
            csv.append({'model': 'performance_only', 'featset': fs, 'cond': cond, 'id': cid, 'comparison': label, **r})
        rows.append(cells)
    md.append(L.md_table(['id', 'real performances vs', 'files/groups', 'all47 T2', 'med16 T2', 'all47 T1', 'med16 T1'], rows))
    fa = {f'{fs} {cond}': float(np.mean(runs[(fs, cond)][(d.kind == 'real_performance').values] > 0.5))
          for fs in ('all47', 'med16') for cond in ('T2', 'T1')}
    key['perfonly:fa_real'] = fa
    md.append('\nReal-performance false alarms (P > 0.5): ' + ', '.join(f'{k} {L.fpct(v)}' for k, v in fa.items()) + '.\n\n')

    # ---- 9.2 ablation for P2 under the performance-only model
    md.append('### 9.2 Feature-group ablation, performance-only detector\n\n')
    rows = []
    for fs in ('all47', 'only_A_med', 'only_A_iqr', 'only_B', 'only_C'):
        if fs not in FEATSETS:
            FEATSETS[fs] = GROUPS[fs]
        cells = [GLABEL[fs]]
        for cond in ('T2', 'T1'):
            sc = runs[('all47', cond)] if fs == 'all47' else P.run_cv(d, 'xgb', fs, cond)[0]
            for cid in ('P2', 'P1', 'P3'):
                if cid not in comps:
                    continue
                _, _, label, rm, sm = comps[cid]
                r = L.compare(y, sc, grp, rm, sm)
                cells.append(_cell(r))
                key[f'perfonly_abl:{fs}:{cond}:{cid}'] = r
                csv.append({'model': 'performance_only_ablation', 'featset': fs, 'cond': cond, 'id': cid, **r})
        rows.append(cells)
    md.append(L.md_table(['features', 'T2: vs sampled digital (P2)', 'T2: vs algorithmic perf (P1)', 'T2: vs modelled (P3)',
                          'T1: P2', 'T1: P1', 'T1: P3'], rows))
    return ''.join(md), runs, d


def iso_only(df, libv, key, csv):
    iso = df[df['kind'].isin(['real_isolated', 'algo_single'])].copy()
    lib_fold = {g: int(df.loc[df.grp == g, 'fold'].iloc[0]) for g in libv['grp'].unique()}
    ln = libv.copy()
    ln['y'], ln['kind'], ln['subkind'] = 1, 'library_note', 'library_note:' + ln['library']
    ln['fold'] = ln['grp'].map(lib_fold).astype(int)
    ln['file'] = [os.path.relpath(p, ROOT).replace('\\', '/') for p in ln['path']]
    cols = ['file', 'y', 'kind', 'subkind', 'grp', 'fold'] + FEATURE_COLS
    t1 = pd.concat([iso[cols], ln[cols]], ignore_index=True)
    iso = iso.reset_index(drop=True)
    res = {}
    for fs in ('all47', 'med16'):
        sc, ext = P.run_cv(iso, 'xgb', fs, 'T1', ext={'libnotes': libv})      # no library in training = "T2"
        res[(fs, 'T2')] = (sc, ext['libnotes'].mean(axis=0))
        sc1, _ = P.run_cv(t1, 'xgb', fs, 'T1')                               # library notes trained on = T1
        res[(fs, 'T1')] = (sc1[:len(iso)], sc1[len(iso):])
    cc = (iso.kind == 'real_isolated').values
    sk = iso['subkind'].values
    g_iso = iso['grp'].values
    mf = libv['mf_layer'].values
    libname = libv['library'].values
    rowsdef = [('SoundFont single notes', 'iso', sk == 'soundfont_single'), ('Otey single notes', 'iso', sk == 'otey_single'),
               ('algorithmic (both)', 'iso', np.isin(sk, ['soundfont_single', 'otey_single'])),
               ('library notes, 4 libraries, mf layer', 'lib', mf), ('library notes, 4 libraries, all layers', 'lib', np.ones(len(libv), bool))]
    for lb in ('Salamander', 'Iowa', 'K18', 'FreePats'):
        rowsdef += [(f'{lb}, mf layer', 'lib', (libname == lb) & mf), (f'{lb}, all layers', 'lib', libname == lb)]
    md = ['\n### 9.3 Isolated-only detector (I-only)\n\n']
    md.append(f'Training population (T2): {int(cc.sum())} CCMusic notes (6 brand groups) vs {int((~cc).sum())} single-note '
              'renders (10 SoundFonts + Otey); library notes (999 files, 4 libraries) scored zero-shot (mean of the 5 fold '
              'models). T1: all 999 library note files added to training, labelled synthetic; they are scored out-of-fold '
              '(each library is one group). Cells as 9.1; FA = CCMusic notes with P > 0.5.\n\n')
    rows = []
    for name, kind, m in rowsdef:
        cells = [name, f'{int(m.sum())}/{len(set((g_iso if kind == "iso" else libv["grp"].values)[m]))}']
        for fs, cond in (('all47', 'T2'), ('med16', 'T2'), ('all47', 'T1'), ('med16', 'T1')):
            sc, lsc = res[(fs, cond)]
            if kind == 'iso':
                r = L.compare(iso['y'].values, sc, g_iso, cc, m)
            else:
                yy = np.r_[np.zeros(cc.sum()), np.ones(m.sum())]
                r = L.compare(yy, np.r_[sc[cc], lsc[m]], np.r_[g_iso[cc], libv['grp'].values[m]], yy == 0, yy == 1)
            cells.append(_cell(r))
            key[f'isoonly:{fs}:{cond}:{name}'] = r
            csv.append({'model': 'isolated_only', 'featset': fs, 'cond': cond, 'id': name, **r})
        rows.append(cells)
    md.append(L.md_table(['CCMusic notes vs', 'files/groups', 'all47 T2', 'med16 T2', 'all47 T1', 'med16 T1'], rows))
    fa = {f'{fs} {cond}': float(np.mean(res[(fs, cond)][0][cc] > 0.5)) for fs in ('all47', 'med16') for cond in ('T2', 'T1')}
    key['isoonly:fa_ccmusic'] = fa
    md.append('\nCCMusic false alarms (P > 0.5): ' + ', '.join(f'{k} {L.fpct(v)}' for k, v in fa.items()) + '.\n')
    return ''.join(md), res, iso


def renders_effect(df, key, csv):
    md = ['\n### 9.4 Corpus-level detector with vs without the dry performance renders (T2)\n\n']
    d0 = df[df['kind'] != 'algo_performance'].reset_index(drop=True)
    rows = []
    out = {}
    for name, d in (('with renders', df), ('without renders', d0)):
        comps = {c[1]: c for c in P._comparisons(d)}
        y, grp = d['y'].values, d['grp'].values
        for fs in ('all47', 'med16'):
            sc = P.run_cv(d, 'xgb', fs, 'T2')[0]
            for cid in ('P2', 'P2a', 'P2d', 'P3', 'P4', 'P6', 'I1a', 'I1b'):
                if cid in comps and comps[cid][4].any():
                    out[(name, fs, cid)] = L.compare(y, sc, grp, comps[cid][3], comps[cid][4])
    for cid in ('P2', 'P2a', 'P2d', 'P3', 'P4', 'P6', 'I1a', 'I1b'):
        cells = [cid]
        for fs in ('all47', 'med16'):
            a, b = out.get(('without renders', fs, cid)), out.get(('with renders', fs, cid))
            if a is None or b is None:
                cells.append('n/a')
                continue
            cells.append(f"{L.fci(a)} (det {L.fpct(a['det_syn'])}) -> {L.fci(b)} (det {L.fpct(b['det_syn'])})")
            key[f'renders_effect:{fs}:{cid}'] = {'without': a, 'with': b}
            csv.append({'model': 'corpus_without_renders', 'featset': fs, 'cond': 'T2', 'id': cid, **a})
        rows.append(cells)
    md.append('Same corpus, folds and parameters as section 2; the only change is whether the 120 SoundFont / Otey '
              'performance renders are in the data (training and evaluation). P2-P6 sources are never trained on (T2).\n\n')
    md.append(L.md_table(['id', 'all47: without -> with renders', 'med16: without -> with renders'], rows))
    return ''.join(md)


def _mw(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan
    u, p = mannwhitneyu(b, a, alternative='two-sided', method='exact')
    return float(p), float(u / (len(a) * len(b)))


def instrument_tests(iso, libv, res, key):
    md = ['\n### 9.5 Instrument-level tests (one value per instrument)\n\n']
    piano = iso['file'].str.split('/').str[-1].str.split('_').str[0].values          # 8 CCMusic pianos
    cc = (iso.kind == 'real_isolated').values
    sf = (iso.subkind == 'soundfont_single').values
    ot = (iso.subkind == 'otey_single').values
    rows, vals = [], {}
    for fs in ('all47', 'med16'):
        for cond in ('T2', 'T1'):
            sc, lsc = res[(fs, cond)]
            v_p8 = pd.Series(sc[cc]).groupby(piano[cc]).median()
            v_g6 = pd.Series(sc[cc]).groupby(iso['grp'].values[cc]).median()
            v_sf = pd.Series(sc[sf]).groupby(iso['grp'].values[sf]).median()
            v_lib_all = pd.Series(lsc).groupby(libv['library'].values).median()
            v_lib_mf = pd.Series(lsc[libv['mf_layer'].values]).groupby(libv['library'].values[libv['mf_layer'].values]).median()
            v_ot = float(np.median(sc[ot]))
            vals[(fs, cond)] = {'pianos8': v_p8.to_dict(), 'libraries_all': v_lib_all.to_dict(), 'soundfonts': v_sf.to_dict(),
                                'otey': v_ot}
            cells = [f'{fs} {cond}']
            for (na, a), (nb, b) in ((('8 pianos', v_p8), ('4 libraries (all layers)', v_lib_all)),
                                      (('8 pianos', v_p8), ('4 libraries (mf)', v_lib_mf)),
                                      (('6 CCMusic groups', v_g6), ('4 libraries (all layers)', v_lib_all)),
                                      (('8 pianos', v_p8), ('10 SoundFonts', v_sf)),
                                      (('4 libraries (all layers)', v_lib_all), ('10 SoundFonts', v_sf))):
                p, auc = _mw(a.values, b.values)
                cells.append(f'{auc:.2f} (p={p:.3g})')
                key[f'inst:{fs}:{cond}:{na} vs {nb}'] = {'p': p, 'P_b_gt_a': auc}
            cells.append(f'{v_ot:.3f}')
            rows.append(cells)
    md.append('Median P(synthetic) of each instrument\'s notes under the I-only detector (9.3). Cells: P(an instrument '
              'of the second set has a higher median P than one of the first set) and exact two-sided Mann-Whitney p '
              '(n = 8 vs 4 gives a minimum p of 0.004; 6 vs 4: 0.0095; 8 vs 10: 4.6e-5).\n\n')
    md.append(L.md_table(['detector', '8 pianos vs 4 libraries (all)', '8 pianos vs 4 libraries (mf)',
                          '6 CCMusic groups vs 4 libraries', '8 pianos vs 10 SoundFonts', '4 libraries vs 10 SoundFonts',
                          'Otey (1 engine) median P'], rows))
    v = vals[('all47', 'T2')]
    md.append('\nPer-instrument medians (all47 T2): pianos ' + ', '.join(f'{k} {x:.3f}' for k, x in v['pianos8'].items()) +
              '; libraries ' + ', '.join(f'{k} {x:.3f}' for k, x in v['libraries_all'].items()) +
              '; SoundFonts ' + ', '.join(f"{k.split(':')[-1]} {x:.3f}" for k, x in v['soundfonts'].items()) + '.\n')
    key['inst_values'] = {f'{a}:{b}': x for (a, b), x in vals.items()}

    # ---- phaseogram irregularity per instrument (figure analysis output)
    p = os.path.join(ROOT, 'Stattia', 'revision', 'figures', 'drift_notes_all.csv')
    if not os.path.exists(p):
        md.append('\n(figures/drift_notes_all.csv not found: phaseogram test skipped.)\n')
        return ''.join(md)
    dr = pd.read_csv(p)
    dr = dr[dr['cls'].isin(['real', 'samplelib', 'soundfont', 'otey'])]
    rows = []
    for name, m in (('MIDI 48-76, pitch check passed (figure-analysis selection)',
                     (dr['ok'] == 1) & (dr['pitch_ok'] == 1) & dr['midi_nom'].between(48, 76)),
                    ('all analysable notes', dr['ok'] == 1)):
        g = dr[m].groupby(['cls', 'source'])['irreg'].median()
        a = g.loc['real'].values
        b = g.loc['samplelib'].values
        c = g.loc['soundfont'].values
        p1, u1 = _mw(a, b)
        p2, u2 = _mw(a, c)
        p3, u3 = _mw(b, c)
        o = g.loc['otey'].values
        rows.append([name, f'{np.median(a):.2f} ({len(a)})', f'{np.median(b):.2f} ({len(b)})', f'{np.median(c):.2f} ({len(c)})',
                     f'{o[0]:.3f}' if len(o) else 'n/a', f'{u1:.2f} (p={p1:.3g})', f'{u2:.2f} (p={p2:.3g})', f'{u3:.2f} (p={p3:.3g})'])
        key[f'phaseo_inst:{name}'] = {'pianos_vs_libs': (p1, u1), 'pianos_vs_sf': (p2, u2), 'libs_vs_sf': (p3, u3),
                                      'medians': {f'{a_}|{b_}': float(v_) for (a_, b_), v_ in g.items()}}
    md.append('\nPhaseogram drift irregularity (`figures/drift_notes_all.csv`, samples per cycle; one median per '
              'instrument). Cells for the tests: P(second set > first set) and exact two-sided Mann-Whitney p.\n\n')
    md.append(L.md_table(['note selection', '8 pianos: median of instrument medians (n)', '4 libraries', '10 SoundFonts',
                          'Otey', 'pianos vs libraries', 'pianos vs SoundFonts', 'libraries vs SoundFonts'], rows))
    return ''.join(md)


def format_counts(df, key):
    md = ['\n### 9.6 Zero IQR values and undefined Group B / C values, by source\n\n']
    B, C = list(BASS_FEATURE_COLS), list(TUNING)
    rows = []
    for lab, m in (('real performances', df.kind == 'real_performance'),
                   ('  manifest', df.subkind == 'manifest'), ('  youtube_extra', df.subkind == 'youtube_extra'),
                   ('SoundFont performance renders', df.subkind == 'sf2_perf'), ('Otey performance renders', df.subkind == 'otey_perf'),
                   ('sampled digital pianos', df.kind == 'sampled_digital_performance'),
                   ('modelled digital pianos', df.kind == 'modelled_digital'), ('Korg M1', df.kind == 'm1'),
                   ('CCMusic notes (isolated, ref.)', df.subkind == 'ccmusic'),
                   ('SoundFont single notes (ref.)', df.subkind == 'soundfont_single'),
                   ('Otey single notes (ref.)', df.subkind == 'otey_single')):
        g = df[m]
        if not len(g):
            continue
        z = g[IQR16].values == 0
        nb, nc = g[B].isna().values, g[C].isna().values
        rows.append([lab, len(g), f'{int(z.sum())} / {z.size}', int(z.any(axis=1).sum()), int(z.all(axis=1).sum()),
                     f'{int(nb.sum())} / {nb.size}', int(nb.any(axis=1).sum()),
                     f'{int(nc.sum())} / {nc.size}', int(nc.any(axis=1).sum())])
        key[f'counts:{lab.strip()}'] = {'files': len(g), 'iqr_zero_values': int(z.sum()), 'files_any_iqr_zero': int(z.any(axis=1).sum()),
                                        'files_all_iqr_zero': int(z.all(axis=1).sum()), 'B_nan_values': int(nb.sum()),
                                        'files_any_B_nan': int(nb.any(axis=1).sum()), 'C_nan_values': int(nc.sum()),
                                        'files_any_C_nan': int(nc.any(axis=1).sum()),
                                        'B_nan_by_feature': {c: int(g[c].isna().sum()) for c in B},
                                        'C_nan_by_feature': {c: int(g[c].isna().sum()) for c in C}}
    md.append(L.md_table(['source', 'files', 'zero IQR values / all IQR values', 'files with >= 1 zero IQR',
                          'files with all 16 IQR = 0', 'undefined B values / all', 'files with >= 1 undefined B',
                          'undefined C values / all', 'files with >= 1 undefined C'], rows))
    rp = df[df.kind == 'real_performance']
    md.append('\nReal performances, undefined values per feature: Group B ' +
              ', '.join(f'{c} {int(rp[c].isna().sum())}' for c in B) + '; Group C ' +
              ', '.join(f'{c} {int(rp[c].isna().sum())}' for c in C) + '. Zero IQR values per feature: ' +
              ', '.join(f'{c} {int((rp[c] == 0).sum())}' for c in IQR16 if (rp[c] == 0).any()) + '.\n')
    return ''.join(md)


def main(log=print):
    key, csv = {}, []
    df, info = P.build_corpus(log=log)
    import followup_v2 as F
    libv = F._libv(df)
    md = ['\n## 9. Format-specific detectors\n\n',
          'Each recording format gets its own detector, trained only on that format, so that every comparison in a '
          'row of the paper\'s Table 7 comes from one training population. XGBoost (train_realness.py parameters, '
          'n_jobs=1), protocol-v2 fixed folds, stratified grouped bootstrap (B = 2000, seed 0).\n\n']
    m1, _, _ = perf_only(df, key, csv)
    md.append(m1)
    log('  formats: performance-only done')
    m2, res, iso = iso_only(df, libv, key, csv)
    md.append(m2)
    log('  formats: isolated-only done')
    md.append(renders_effect(df, key, csv))
    log('  formats: renders effect done')
    md.append(instrument_tests(iso, libv, res, key))
    md.append(format_counts(df, key))
    pd.DataFrame(csv).to_csv(os.path.join(OUT, 'format_specific.csv'), index=False, float_format='%.6f')
    return ''.join(md), key


if __name__ == '__main__':
    t, _ = main()
    print(t)
