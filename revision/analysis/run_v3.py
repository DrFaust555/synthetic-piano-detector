"""Protocol v3 -- ONE command for every number in out_v3/RESULTS_v3.md.

    cd <project root>
    python Stattia/revision/analysis/run_v3.py

Needs xgboost 3.2.0, tinysoundfont (C1b re-renders), the MIT IR Survey in audio/_ir/mit_survey/Audio/
(control C3) and Stattia/revision/data_v2/scripts/otey_v0_build/piano_v0.exe (Otey C1b re-renders).
Reads only (apart from the IR download, which is described in RESULTS_v3); writes into
Stattia/revision/analysis/out_v3/.  Extractions, renders and loop indices are cached in out_v3/cache/.
v2 (run_all.py, out/) is untouched.
"""
from __future__ import annotations

import os
import sys
import json
import time
import platform

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import numpy as np            # noqa: E402
import pandas as pd           # noqa: E402
import pv2_lib as L           # noqa: E402
import v3_lib as V            # noqa: E402
from pv2_lib import FEATURE_COLS, ROOT   # noqa: E402

PERF_KINDS = ['real_performance', 'algo_performance', 'sampled_digital_performance', 'modelled_digital', 'm1',
              'sound_module', 'digital_other']


def iso_only(df, libv, key, csv):
    import protocol_v2 as P  # noqa: F401
    lib_fold = {g: int(df.loc[df.grp == g, 'fold'].iloc[0]) for g in libv['grp'].unique()}
    libv = libv.copy()
    libv['fold'] = libv['grp'].map(lib_fold).astype(int)
    libv['file'] = [os.path.relpath(p, ROOT).replace('\\', '/') for p in libv['path']]
    libv['y'], libv['kind'] = 1, 'library_note'
    res = {}
    for dups in (False, True):
        iso = df[df['kind'].isin(['real_isolated', 'algo_single']) & (dups | ~df['sf_dup'])].reset_index(drop=True)
        for fs in ('all47', 'med16') if not dups else ('all47',):
            r2 = V.run_cv3(iso, V.FS3[fs], 'T1', ext={'lib': libv}, ext_fold={'lib': libv['fold'].values})
            res[(fs, 'T2', dups)] = (iso, r2['own'], r2['ext_own']['lib'], r2['ext_mean5']['lib'])
            if dups:
                continue
            cols = ['file', 'y', 'kind', 'subkind', 'grp', 'fold'] + FEATURE_COLS
            lb = libv.copy()
            lb['subkind'] = 'library_note'
            t1 = pd.concat([iso[cols], lb[cols]], ignore_index=True)
            r1 = V.run_cv3(t1, V.FS3[fs], 'T1')
            res[(fs, 'T1', dups)] = (iso, r1['own'][:len(iso)], r1['own'][len(iso):], None)
    mf = libv['mf_layer'].values
    name_ = libv['library'].values
    defs = [('SoundFont single notes', 'iso', 'soundfont_single'), ('Otey single notes', 'iso', 'otey_single'),
            ('algorithmic (both)', 'iso', None), ('library notes, 4 libraries, mf layer', 'lib', mf),
            ('library notes, 4 libraries, all layers', 'lib', np.ones(len(libv), bool))]
    for lb_ in ('Salamander', 'Iowa', 'K18', 'FreePats'):
        defs += [(f'{lb_}, mf layer', 'lib', (name_ == lb_) & mf), (f'{lb_}, all layers', 'lib', name_ == lb_)]
    rows = []
    for name, kind, sel in defs:
        cells = [name]
        for fs, cond in (('all47', 'T2'), ('med16', 'T2'), ('all47', 'T1'), ('med16', 'T1')):
            iso, sc, lsc, lm5 = res[(fs, cond, False)]
            cc = (iso.kind == 'real_isolated').values
            g_iso = iso['grp'].values
            if kind == 'iso':
                sm = (iso.subkind == sel).values if sel else (iso.kind == 'algo_single').values
                r = V.cmp3(iso['y'].values, sc, g_iso, cc, sm)
                n = int(sm.sum())
            else:
                yy = np.r_[np.zeros(cc.sum()), np.ones(sel.sum())]
                gg = np.r_[g_iso[cc], libv['grp'].values[sel]]
                r = V.cmp3(yy, np.r_[sc[cc], lsc[sel]], gg, yy == 0, yy == 1)
                n = int(sel.sum())
                if cond == 'T2':
                    r5 = V.cmp3(yy, np.r_[sc[cc], lm5[sel]], gg, yy == 0, yy == 1)
                    key[f'T7iso:{fs}:T2:{name}:mean5'] = r5
                    csv.append({'table': 'iso', 'featset': fs, 'cond': 'T2_mean5', 'row': name, **r5})
            if len(cells) == 1:
                cells.append(f'{n}/{r["g_syn"]}')
            cells.append(V.fmt3(r, meanp=True))
            key[f'T7iso:{fs}:{cond}:{name}'] = r
            csv.append({'table': 'iso', 'featset': fs, 'cond': cond, 'row': name, **r})
        rows.append(cells)
    fa = {}
    for fs, cond in (('all47', 'T2'), ('med16', 'T2'), ('all47', 'T1'), ('med16', 'T1')):
        iso, sc, _, _ = res[(fs, cond, False)]
        fa[f'{fs} {cond}'] = float(np.mean(sc[(iso.kind == 'real_isolated').values] > 0.5))
    key['T7iso:fa'] = fa
    # sensitivity: velocity copies kept
    iso_d, scd, _, _ = res[('all47', 'T2', True)]
    rd = V.cmp3(iso_d['y'].values, scd, iso_d['grp'].values, (iso_d.kind == 'real_isolated').values,
                (iso_d.subkind == 'soundfont_single').values)
    key['T7iso:all47:T2:SoundFont incl. velocity copies'] = rd
    lib_m5 = {fs: key[f'T7iso:{fs}:T2:library notes, 4 libraries, mf layer:mean5'] for fs in ('all47', 'med16')}
    md = ['\n### 2.1 Isolated notes: I-only detector\n\n',
          f'Training (T2): {int((iso.kind == "real_isolated").sum())} CCMusic notes (6 brand groups) vs '
          f'{int((iso.subkind == "soundfont_single").sum())} SoundFont renders (velocity copies removed) + '
          f'{int((iso.subkind == "otey_single").sum())} Otey notes. Library notes (999 files) never trained (T2), scored '
          'by the model of their library\'s fold (own-fold). T1: library notes added to training labelled synthetic, '
          'scored out-of-fold. Cells: AUC [95% CI]; det = share of the synthetic notes with P > 0.5; P = mean '
          'P(synthetic). "real-side only" = one synthetic group: the interval resamples real groups only and does not '
          'generalise to other engines.\n\n']
    md.append(L.md_table(['CCMusic notes vs', 'notes/groups', 'all47 T2', 'med16 T2', 'all47 T1', 'med16 T1'], rows))
    md.append('\nCCMusic false alarms (P > 0.5): ' + ', '.join(f'{k} {L.fpct(v)}' for k, v in fa.items()) +
              f'. Sensitivity: SoundFont incl. the {int(df["sf_dup"].sum())} velocity copies (all47 T2): {V.fmt3(rd)}. '
              'Library notes with the v2 rule (mean of 5 fold models), mf layer: ' +
              ', '.join(f'{fs} {V.fmt3(r)}' for fs, r in lib_m5.items()) + '.\n')
    return ''.join(md)


def perf_only(df, key, csv):
    import protocol_v2 as P
    d = df[df['kind'].isin(PERF_KINDS)].reset_index(drop=True)
    comps = {c[1]: c for c in P._comparisons(d)}
    ids = [i for i in ('P1', 'P1a', 'P1b', 'P2', 'P2a', 'P2d', 'P2d-line', 'P2d-mic', 'P2d-unk', 'P2d-lt', 'P2d-nolt',
                       'P3', 'P3d', 'P4', 'P5', 'P6') if i in comps and comps[i][4].any()]
    y, grp = d['y'].values, d['grp'].values
    runs = {(fs, c): V.run_cv3(d, V.FS3[fs], c) for fs in ('all47', 'med16') for c in ('T2', 'T1')}
    rows = []
    for cid in ids:
        _, _, label, rm, sm = comps[cid]
        cells = [cid, label.replace('real_performance vs ', ''), f'{int(sm.sum())}/{len(set(grp[sm]))}']
        for fs, c in (('all47', 'T2'), ('med16', 'T2'), ('all47', 'T1'), ('med16', 'T1')):
            r = V.cmp3(y, runs[(fs, c)]['own'], grp, rm, sm)
            cells.append(V.fmt3(r, meanp=True))
            key[f'T7perf:{fs}:{c}:{cid}'] = r
            csv.append({'table': 'perf', 'featset': fs, 'cond': c, 'row': cid, **r})
        r5 = V.cmp3(y, runs[('all47', 'T2')]['mean5'], grp, rm, sm)
        cells.append(V.fmt3(r5))
        key[f'T7perf:all47:T2:{cid}:mean5'] = r5
        csv.append({'table': 'perf', 'featset': 'all47', 'cond': 'T2_mean5', 'row': cid, **r5})
        rows.append(cells)
    rp = (d.kind == 'real_performance').values
    fa = {f'{fs} {c}': float(np.mean(runs[(fs, c)]['own'][rp] > 0.5)) for fs in ('all47', 'med16') for c in ('T2', 'T1')}
    key['T7perf:fa'] = fa
    md = ['\n### 2.2 Performances: performance-only detector\n\n',
          f'Training (T2): {int(rp.sum())} real performances (cleaned; grouped by v2 group + YouTube video id) vs '
          f'{int((d.kind == "algo_performance").sum())} SoundFont / Otey performance renders (each with its engine\'s '
          'group). Digital pianos, M1, SC-55 and GP-3 never trained (T2), scored own-fold (the model of their group\'s '
          'fold); T1: trained, labelled synthetic. Digital pianos grouped by brand. Last column: the v2 scoring rule '
          '(mean of the 5 fold models) for comparison.\n\n']
    md.append(L.md_table(['id', 'real performances vs', 'files/groups', 'all47 T2', 'med16 T2', 'all47 T1', 'med16 T1',
                          'all47 T2, v2 rule (mean of 5)'], rows))
    md.append('\nReal-performance false alarms (P > 0.5): ' + ', '.join(f'{k} {L.fpct(v)}' for k, v in fa.items()) + '.\n')
    # ablation for P2 (T2, own fold)
    rows = []
    for fs in ('all47', 'only_A_med', 'only_A_iqr', 'only_B', 'only_C'):
        sc = runs[('all47', 'T2')]['own'] if fs == 'all47' else V.run_cv3(d, V.FS3[fs], 'T2')['own']
        cells = [fs]
        for cid in ('P2', 'P1', 'P3'):
            r = V.cmp3(y, sc, grp, comps[cid][3], comps[cid][4])
            cells.append(V.fmt3(r))
            key[f'T7perf_abl:{fs}:{cid}'] = r
        rows.append(cells)
    md.append('\nFeature-group ablation (performance-only, T2, own-fold):\n\n')
    md.append(L.md_table(['features', 'P2 sampled digital', 'P1 algorithmic perf', 'P3 modelled'], rows))
    return ''.join(md)


def _summary(key, ckey):
    def g(k):
        r = key.get(k) or ckey.get(k)
        return V.fmt3(r) if isinstance(r, dict) and 'auc' in r else 'n/a'
    rows = []
    for name, k in (('isolated: CCMusic vs SoundFont', 'T7iso:{fs}:T2:SoundFont single notes'),
                    ('isolated: CCMusic vs Otey', 'T7iso:{fs}:T2:Otey single notes'),
                    ('isolated: CCMusic vs library notes (mf)', 'T7iso:{fs}:T2:library notes, 4 libraries, mf layer'),
                    ('performance: real vs algorithmic renders', 'T7perf:{fs}:T2:P1'),
                    ('performance: real vs sampled digital pianos', 'T7perf:{fs}:T2:P2'),
                    ('performance: real vs modelled digital', 'T7perf:{fs}:T2:P3'),
                    ('performance: real vs Korg M1', 'T7perf:{fs}:T2:P4')):
        rows.append([name, g(k.format(fs='all47')), g(k.format(fs='med16'))])
    s = ['## 0. Key numbers (T2, own-fold scoring, cleaned real class)\n\n',
         L.md_table(['comparison', 'all 47', '16 Group-A medians'], rows)]
    rows = []
    for nm in ('SoundFont', 'SoundFont, unaltered sample', 'Otey', 'library notes, mf layer'):
        rows.append([nm] + [(f"{ckey[f'ctrl:all47:{c}:{nm}']['auc']:.3f}" if np.isfinite(ckey.get(f'ctrl:all47:{c}:{nm}', {}).get('auc') or np.nan) else 'n/a')
                            for c in ('orig', 'c1a', 'c1b', 'c3', 'c1a+c3', 'c1b+c3')])
    s.append('\nControl table B in brief (I-only, all 47, T2, AUC vs CCMusic):\n\n')
    s.append(L.md_table(['source', 'original', 'C1a 1.2 s', 'C1b 4.0 s', 'C3', 'C1a+C3', 'C1b+C3'], rows))
    return ''.join(s) + '\n'


def main():
    t0 = time.time()
    os.makedirs(V.OUT3, exist_ok=True)
    xv = L.check_xgboost()
    import followup_v2 as F
    import v3_render as R
    import v3_controls as C
    print('protocol v3: corpus ...', flush=True)
    df, info = V.build_corpus_v3()
    rep = info['v3']
    df[['file', 'y', 'kind', 'subkind', 'grp', 'fold', 'sf_dup']].to_csv(os.path.join(V.OUT3, 'folds.csv'), index=False)
    ex = [{'file': f, 'reason': r, 'source': 'CODE_AUDIT B1'} for f, r in rep['audit_B1']] + \
         [{'file': f, 'reason': r, 'source': 'real_class_exclusions.csv'} for f, r in rep['screen']] + \
         [{'file': f, 'reason': 'byte-identical duplicate of ' + g[0], 'source': 'md5'} for g in rep['duplicates'] for f in g[1:]]
    pd.DataFrame(ex).to_csv(os.path.join(V.OUT3, 'excluded_real_v3.csv'), index=False)
    key, csv = {}, []
    libv = F._libv(df)
    print('protocol v3: Table 7 ...', flush=True)
    t7 = ['\n## 2. Table 7, protocol v3: one detector per format (own-fold scoring)\n'] + \
        [iso_only(df, libv, key, csv), perf_only(df, key, csv)]
    pd.DataFrame(csv).to_csv(os.path.join(V.OUT3, 'table7_v3.csv'), index=False, float_format='%.6f')
    print('protocol v3: renders ...', flush=True)
    rr = {'sf': R.render_soundfonts(), 'otey': R.render_otey()}
    print('protocol v3: controls ...', flush=True)
    cmd, ckey, _ = C.main(df)
    corp = ['\n## 1. Corpus v3\n\n',
            f"Real class cleaned: {len(rep['audit_B1'])} non-acoustic files of CODE_AUDIT B1 removed "
            f"(missing from the corpus: {rep['audit_B1_missing'] or 'none'}); {rep['screen_note']} "
            f"({len(rep['screen'])} removed); {sum(len(g) - 1 for g in rep['duplicates'])} byte-identical duplicate "
            f"copies removed ({'; '.join(' = '.join(g) for g in rep['duplicates'])}). Real performances regrouped by "
            f"YouTube video id: {rep['n_real_groups_v2']} -> {rep['n_real_groups_v3']} groups. SoundFont velocity copies "
            f"(identical 47-feature vectors within an instrument): {rep['sf_dups']} files flagged and left out of the "
            'isolated analyses (sensitivity row keeps them). Folds: v2 rule re-applied to the v3 groups '
            f"(`out_v3/folds.csv`). Corpus: {len(df)} files; " +
            ', '.join(f'{k} {v}' for k, v in df.groupby('kind').size().items()) + '.\n',
            '\nC1b re-renders: ' + ', '.join(f"{k} max|d| {v.get('repro_maxdiff_1p4s', 'cached')}" for k, v in rr['sf'].items()) +
            f"; Otey reproduction of out_full at 3.0 s: max|d| {rr['otey'].get('repro_maxdiff_3s')}.\n"]
    files = ['folds.csv', 'table7_v3.csv', 'controls_scores.csv', 'c2_soundfont_categories.csv',
             'c4_phaseogram_c3.csv', 'c5_loop_index.csv', 'excluded_real_v3.csv']
    ck = os.path.join(V.OUT3, 'checksums.txt')
    prev = {}
    if os.path.exists(ck):
        for line in open(ck, encoding='utf-8'):
            if '  ' in line:
                h, f = line.strip().split('  ', 1)
                prev[f] = h
    now = {f: L.md5_of(os.path.join(V.OUT3, f)) for f in files if os.path.exists(os.path.join(V.OUT3, f))}
    open(ck, 'w', encoding='utf-8').write(''.join(f'{h}  {f}\n' for f, h in now.items()))
    changed = [f for f in now if f in prev and prev[f] != now[f]]
    det = (f"Determinism check vs the previous run: {sum(prev.get(f) == h for f, h in now.items())} of {len(now)} output "
           f"CSVs byte-identical" + (f"; changed: {', '.join(changed)}" if changed else '') + '.') if prev else \
        'Determinism check: first run (no previous out_v3/checksums.txt).'
    json.dump({'table7': key, 'controls': ckey, 'corpus': {k: v for k, v in rep.items() if k != 'screen'}},
              open(os.path.join(V.OUT3, 'key_numbers_v3.json'), 'w'), indent=1,
              default=lambda o: o.item() if isinstance(o, (np.floating, np.integer, np.bool_)) else str(o))
    head = ['# RESULTS v3 -- protocol v3 (after the independent code audit)\n\n',
            f'Generated {time.strftime("%Y-%m-%d %H:%M")} by one command, from the project root:\n\n',
            '```\npython Stattia/revision/analysis/run_v3.py\n```\n\n',
            f'Environment: Python {platform.python_version()}, numpy {np.__version__}, pandas {pd.__version__}, '
            f'xgboost {xv} (n_jobs=1, random_state=0). Scripts: `Stattia/revision/analysis/{{run_v3, v3_lib, v3_sf2, '
            f'v3_render, v3_controls}}.py` (+ v2 modules for the corpus). Runtime {time.time() - t0:.0f} s. {det}\n\n',
            'Fixes against v2 (CODE_AUDIT.md): A1 own-fold scoring of every never-trained row (v2 rule as secondary '
            'column); A2 cleaned real class + video-id grouping; A3 SoundFont velocity copies removed; A4 one-group '
            'synthetic sides get "real-side only" intervals. Outputs: `out_v3/` (folds.csv, table7_v3.csv, '
            'controls_scores.csv, c2_soundfont_categories.csv, c4_phaseogram_c3.csv, c5_loop_index.csv, '
            'excluded_real_v3.csv, key_numbers_v3.json, cache/).\n\n']
    text = ''.join(head) + _summary(key, ckey) + ''.join(corp) + ''.join(t7) + cmd
    open(os.path.join(V.OUT3, 'RESULTS_v3.md'), 'w', encoding='utf-8').write(text)
    print(f'wrote {os.path.join(V.OUT3, "RESULTS_v3.md")} ({time.time() - t0:.0f} s)')


if __name__ == '__main__':
    main()
