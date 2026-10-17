"""FINAL protocol-v3 run: every number of manuscript_v2.md Tables 7-10 and Sections 5.1-5.6.

    cd <project root>
    python Stattia/revision/analysis/final_v3.py

1. run_v3.main() on the PRIMARY corpus: CODE_AUDIT B1 files + every 'exclude' row of
   audit/real_class_exclusions.csv removed, real performances regrouped by group_id / video id
   (writes out_v3/RESULTS_v3.md, key_numbers_v3.json; controls C1-C5).
2. SENSITIVITY corpus: also every 'review' row removed -> Table 7 again.
3. Items (a)-(f): eGeMAPS / CNN / fusion library-note AUCs with own-fold scoring; effect of the dry
   performance renders (own-fold); the 30 listening clips scored by the isolated-note detector
   (leave-own-group-out); f0 / register restriction of the isolated comparisons; zero-IQR / undefined counts.
4. out_v3/FINAL_NUMBERS.md (+ final_numbers.json): primary and sensitivity values side by side.
All scoring own-fold (a never-trained row is scored by the single model of its own fold).
"""
from __future__ import annotations

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import pv2_lib as L                       # noqa: E402
import v3_lib as V                        # noqa: E402
from pv2_lib import FEATURE_COLS, ROOT, KIT   # noqa: E402

REV = os.path.join(ROOT, 'Stattia', 'revision')
IQR16 = [c for c in FEATURE_COLS if c.endswith('_iqr')]


def fa3(r, ci=True):
    if not isinstance(r, dict) or not np.isfinite(r.get('auc', np.nan)):
        return 'n/a'
    if r.get('one_syn_group') or r.get('g_syn') == 1:
        return f"{r['auc']:.3f}*"
    return f"{r['auc']:.3f} [{r['ci_lo']:.2f}, {r['ci_hi']:.2f}]" if ci else f"{r['auc']:.3f}"


def det(r):
    return L.fpct(r['det_syn']) if isinstance(r, dict) and np.isfinite(r.get('det_syn', np.nan)) else 'n/a'


def mw(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    u, p = mannwhitneyu(b, a, alternative='two-sided', method='exact')
    return float(p), float(u / (len(a) * len(b)))


# ----------------------------------------------------------------------------- isolated I-only scores
def iso_scores(df, libv, fs='all47', exclude_grp=()):
    lib_fold = {g: int(df.loc[df.grp == g, 'fold'].iloc[0]) for g in libv['grp'].unique()}
    iso = df[df['kind'].isin(['real_isolated', 'algo_single']) & ~df['sf_dup']].reset_index(drop=True)
    iso['trainable'] = ~iso['grp'].isin(list(exclude_grp))
    lf = libv['grp'].map(lib_fold).values
    return iso, V.run_cv3(iso, V.FS3[fs], 'T1', ext={'lib': libv}, ext_fold={'lib': lf}), lf


# ----------------------------------------------------------------------------- eGeMAPS (88 functionals, cached extraction)
def load_egemaps():
    eg = pd.read_csv(os.path.join(REV, 'egemaps', 'out', 'cache', 'egemaps_main.csv'))
    want = np.where(eg['file'].str.startswith('audio/ref_'), 'mono', 'file')   # run_egemaps.py l.85/93
    eg = eg[(eg['variant'] == 'main') & (eg['loader'] == want)].drop_duplicates('file').set_index('file')
    feg = [c for c in eg.columns if c not in ('file', 'sig', 'loader', 'variant', 'sr', 'channels', 'duration_s', 'peak',
                                              'n_wrap', 'error')]
    return eg, feg


def egemaps_lib(iso, libv, lf, EG, ci=True):
    eg, FEG = EG

    def egx(files):
        rel = [f if f.startswith(('audio/', 'otey/')) else 'audio/' + f for f in files]
        return pd.DataFrame(eg[FEG].reindex(rel).values, columns=FEG)
    cc = (iso.kind == 'real_isolated').values
    d = iso[['file', 'y', 'kind', 'subkind', 'grp', 'fold']].copy()
    d[FEG] = egx(list(d['file'])).values
    ld = libv[['grp']].copy()
    ld[FEG] = egx([os.path.relpath(p_, ROOT).replace('\\', '/') for p_ in libv['path']]).values
    r_eg = V.run_cv3(d, FEG, 'T1', ext={'lib': ld}, ext_fold={'lib': lf})
    egres = {}
    for name, m in (('library mf', libv['mf_layer'].values), ('library all', np.ones(len(libv), bool))):
        for rule, s2 in (('own', r_eg['ext_own']['lib']), ('mean5', r_eg['ext_mean5']['lib'])):
            yy = np.r_[np.zeros(cc.sum()), np.ones(m.sum())]
            egres[f'{name}|{rule}'] = V.cmp3(yy, np.r_[r_eg['own'][cc], s2[m]],
                                            np.r_[d['grp'].values[cc], libv['grp'].values[m]], yy == 0, yy == 1)
    alg = (d.kind == 'algo_single').values
    egres['algorithmic'] = V.cmp3(d['y'].values, r_eg['own'], d['grp'].values, cc, alg)
    egres['n_missing_features'] = int(d[FEG].isna().all(axis=1).sum() + ld[FEG].isna().all(axis=1).sum())
    return egres



# ----------------------------------------------------------------------------- fold-seed sweep
N_SEEDS = 10


def refold(df, seed):
    fmap = L.assign_folds(df.loc[df.base, 'grp'], df.loc[~df.base, 'grp'], seed=seed)
    d = df.copy()
    d['fold'] = d['grp'].map(fmap).astype(int)
    return d


def _auc(r):
    return float(r['auc']) if isinstance(r, dict) and np.isfinite(r.get('auc', np.nan)) else np.nan


def sweep(df, libv, EG, log=print):
    """Primary corpus, seeds 0..N_SEEDS-1 of the greedy fold rule (tie order permuted by the seed).
    Per seed: Table 7 (all47 + med16, T2, own-fold), Table 8 controls (all47), instrument-level tests,
    eGeMAPS library rows.  Seed 0 must reproduce the primary run exactly (checked)."""
    import run_v3 as R3
    import v3_controls as C
    src0 = C.iso_sources(df, libv)
    cats = C.sf2_categories(src0)
    feats = {'orig': C.orig_features(df, src0, libv)}
    for c in C.CONDS[1:]:
        feats[c] = C.features(src0, c, log=lambda *a, **k: None)
    lev = feats['c1b']['level_4s_db'].values
    keep_real = src0.src.isin(['CCMusic', 'library']).values & np.isfinite(lev) & (lev >= -60.0)
    keep = {c: np.ones(len(src0), bool) for c in C.CONDS}
    for c in ('c1b', 'c1b+c3'):
        keep[c] = src0.src.isin(['SoundFont', 'Otey']).values | keep_real
    per = []
    for seed in range(N_SEEDS):
        ds = refold(df, seed)
        if seed == 0:
            assert (ds['fold'].values == df['fold'].values).all(), 'seed 0 does not reproduce the primary folds'
        v = {}
        ks = {}
        R3.iso_only(ds, libv, ks, [])
        R3.perf_only(ds, ks, [])
        for k, r in ks.items():
            if ':T2:' in k and not k.endswith(':mean5'):
                v[k] = _auc(r)
        for k in ('all47 T2', 'med16 T2'):
            v[f'T7iso:fa:{k}'] = ks['T7iso:fa'][k]
            v[f'T7perf:fa:{k}'] = ks['T7perf:fa'][k]
        fmap = dict(zip(ds['grp'], ds['fold']))
        src = src0.copy()
        src['fold'] = src['grp'].map(fmap).astype(int)
        for c in C.CONDS:
            m = keep[c]
            allr = C.run_iso(src[m].reset_index(drop=True), feats[c][m].reset_index(drop=True), 'all47')
            t = C.iso_table(allr, cats, c)
            for nm in ('SoundFont', 'SoundFont, unaltered sample', 'Otey', 'library notes, all layers',
                       'library notes, mf layer'):
                v[f'ctrl:all47:{c}:{nm}'] = _auc(t[nm])
            v[f'ctrl:all47:{c}:CCMusic FA'] = t['CCMusic FA']
        iso, res, lf = iso_scores(ds, libv)
        sc, lsc = res['own'], res['ext_own']['lib']
        cc = (iso.kind == 'real_isolated').values
        sfm = (iso.subkind == 'soundfont_single').values
        piano = iso['file'].str.split('/').str[-1].str.split('_').str[0].values
        v_p = pd.Series(sc[cc]).groupby(piano[cc]).median()
        v_l = pd.Series(lsc).groupby(libv['library'].values).median()
        v_s = pd.Series(sc[sfm]).groupby(iso['grp'].values[sfm]).median()
        for nm, b in (('libraries', v_l), ('soundfonts', v_s)):
            p_, P_ = mw(v_p.values, b.values)
            v[f'instr:pianos_vs_{nm}:p'], v[f'instr:pianos_vs_{nm}:P'] = p_, P_
        eg = egemaps_lib(iso, libv, lf, EG)
        for k in ('library mf|own', 'library all|own', 'algorithmic'):
            v[f'egemaps:{k}'] = _auc(eg[k])
        per.append(v)
        log(f'final: sweep seed {seed} done')
    keys = list(per[0])
    tab = {k: [p_.get(k, np.nan) for p_ in per] for k in keys}
    pd.DataFrame({'seed': range(N_SEEDS), **tab}).to_csv(os.path.join(V.OUT3, 'fold_seed_sweep.csv'), index=False,
                                                         float_format='%.6f')
    summ = {k: {'seed0': x[0], 'median': float(np.nanmedian(x)), 'min': float(np.nanmin(x)), 'max': float(np.nanmax(x)),
                'values': x} for k, x in tab.items()}
    return summ



def main():
    t0 = time.time()
    L.check_xgboost()
    import run_v3 as R3
    import followup_v2 as F
    out = {}
    print('final: primary run_v3 ...', flush=True)
    R3.main()
    kp = json.load(open(os.path.join(V.OUT3, 'key_numbers_v3.json')))
    T7p, CT = kp['table7'], kp['controls']
    df, info = V.build_corpus_v3()
    libv = F._libv(df)
    print('final: sensitivity corpus ...', flush=True)
    dfs, infos = V.build_corpus_v3(include_review=True)
    libs_s = F._libv(dfs)
    ks, cs = {}, []
    R3.iso_only(dfs, libs_s, ks, cs)
    R3.perf_only(dfs, ks, cs)
    T7s = json.loads(json.dumps(ks, default=lambda o: o.item() if isinstance(o, (np.floating, np.integer, np.bool_)) else str(o)))

    # ---------------- 5.1 format: counts and the mixed-format detector
    def counts(d):
        rp = d[d.kind == 'real_performance']
        sf = d[d.subkind == 'soundfont_single']
        z = rp[IQR16].values == 0
        return {'sf_all16_zero': int((sf[IQR16].values == 0).all(axis=1).sum()), 'sf_n': len(sf),
                'perf_zero_iqr_share': float(z.mean()), 'perf_zero_iqr_values': int(z.sum()), 'perf_values': int(z.size),
                'perf_all16_zero': int(z.all(axis=1).sum()), 'perf_n': len(rp),
                'perf_any_B_undefined_files': int(rp[list(V.FS3['only_B'])].isna().any(axis=1).sum()),
                'perf_B_undefined_values': int(rp[list(V.FS3['only_B'])].isna().values.sum()),
                'perf_any_C_undefined_files': int(rp[list(V.FS3['only_C'])].isna().any(axis=1).sum()),
                'perf_C_undefined_values': int(rp[list(V.FS3['only_C'])].isna().values.sum())}
    out['counts'] = {'primary': counts(df), 'sensitivity': counts(dfs)}

    def mixed(d):
        import protocol_v2 as P
        comps = {c[1]: c for c in P._comparisons(d)}
        r = {}
        for fs in ('all47', 'med16'):
            sc = V.run_cv3(d, V.FS3[fs], 'T2')['own']
            r[fs] = V.cmp3(d['y'].values, sc, d['grp'].values, comps['C1'][3], comps['C1'][4])
        # (c) renders effect, own-fold
        d0 = d[d.kind != 'algo_performance'].reset_index(drop=True)
        c0 = {c[1]: c for c in P._comparisons(d0)}
        sc0 = V.run_cv3(d0, V.FS3['all47'], 'T2')['own']
        sc1 = V.run_cv3(d, V.FS3['all47'], 'T2')['own']
        r['renders'] = {'without': V.cmp3(d0['y'].values, sc0, d0['grp'].values, c0['P2'][3], c0['P2'][4]),
                        'with': V.cmp3(d['y'].values, sc1, d['grp'].values, comps['P2'][3], comps['P2'][4])}
        return r
    print('final: mixed-format and renders effect ...', flush=True)
    out['mixed'] = {'primary': mixed(df), 'sensitivity': mixed(dfs)}

    # ---------------- 5.2 instrument-level, f0 restriction, clips (primary; isolated data unchanged by the screen)
    print('final: isolated extras ...', flush=True)
    iso, res, lf = iso_scores(df, libv)
    sc, lsc = res['own'], res['ext_own']['lib']
    cc = (iso.kind == 'real_isolated').values
    sfm = (iso.subkind == 'soundfont_single').values
    piano = iso['file'].str.split('/').str[-1].str.split('_').str[0].values
    v_p = pd.Series(sc[cc]).groupby(piano[cc]).median()
    v_l = pd.Series(lsc).groupby(libv['library'].values).median()
    v_s = pd.Series(sc[sfm]).groupby(iso['grp'].values[sfm]).median()
    out['instrument'] = {'pianos_vs_libraries': mw(v_p.values, v_l.values), 'pianos_vs_soundfonts': mw(v_p.values, v_s.values)}

    # (e) f0 / register restriction (f0 audit caches of protocol v2)
    a_iso = pd.read_csv(os.path.join(L.FEAT_CACHE, 'f0_isolated.csv'))
    a_lib = pd.read_csv(os.path.join(L.FEAT_CACHE, 'f0_libnotes.csv'))
    a = pd.concat([a_iso, a_lib]).drop_duplicates('file').set_index('file')

    def okmask(files):
        rel = [f if f.startswith(('audio/', 'otey/')) else 'audio/' + f for f in files]
        e = a['primary_err'].reindex(rel).values
        n = a['nominal_midi'].reindex(rel).values
        return (np.abs(e) < 1) & (n >= 48) & (n <= 76)
    ok_iso = okmask(list(iso['file']))
    ok_lib = okmask([os.path.relpath(p, ROOT).replace('\\', '/') for p in libv['path']])
    f0r = {}
    for name, m in (('SoundFont', sfm), ('Otey', (iso.subkind == 'otey_single').values)):
        for tag, rr in (('all', np.ones(len(iso), bool)), ('f0-valid & 48-76', ok_iso)):
            f0r[f'{name}|{tag}'] = V.cmp3(iso['y'].values, sc, iso['grp'].values, cc & rr, m & rr)
    for name, m in (('library mf', libv['mf_layer'].values), ('library all', np.ones(len(libv), bool))):
        for tag, rl, rr in (('all', np.ones(len(libv), bool), np.ones(len(iso), bool)), ('f0-valid & 48-76', ok_lib, ok_iso)):
            rm, sm = cc & rr, m & rl
            yy = np.r_[np.zeros(rm.sum()), np.ones(sm.sum())]
            f0r[f'{name}|{tag}'] = V.cmp3(yy, np.r_[sc[rm], lsc[sm]], np.r_[iso['grp'].values[rm], libv['grp'].values[sm]], yy == 0, yy == 1)
    out['f0_restriction'] = f0r
    out['f0_counts'] = {'ccmusic_ok_48_76': int(ok_iso[cc].sum()), 'sf_ok': int(ok_iso[sfm].sum()),
                        'otey_ok': int(ok_iso[(iso.subkind == 'otey_single').values].sum()), 'lib_ok': int(ok_lib.sum())}

    # (d) listening clips, I-only detector, leave-own-group-out
    key = pd.read_csv(os.path.join(KIT, 'answer_key.csv'))
    clips = L.extract_cached([os.path.join(KIT, 'web', c + '.wav') for c in key['clip']], 'stimuli_clips')
    cls = key['source'].map({'Iowa Steinway': 'Iowa', 'K18 upright': 'K18', 'Salamander (sampled)': 'Salamander',
                             'soundfont': 'SoundFont', 'Otey physical model': 'Otey'}).values
    casio = 'syn:sf2:' + L.norm_token('Casio_CA_110_Piano_FM')
    clip_res = {}
    for fs in ('all47', 'med16'):
        p = np.full(len(key), np.nan)
        for ex, m in ((('syn:otey',), cls == 'Otey'), ((casio,), cls == 'SoundFont'), ((), np.isin(cls, ['Iowa', 'K18', 'Salamander']))):
            d = iso.copy()
            d = d[~d['grp'].isin(list(ex))].reset_index(drop=True)
            spw = float((d.y == 0).sum()) / max(1, int((d.y == 1).sum()))
            _, (pp,) = L.fit_predict('xgb', d[V.FS3[fs]].values.astype(float), d['y'].values,
                                     [clips[V.FS3[fs]].values.astype(float)], spw=spw)
            p[m] = pp[m]
        rec = np.isin(cls, ['Iowa', 'K18', 'Salamander'])
        clip_res[fs] = {'counts': {c: f"{int((p[cls == c] > 0.5).sum())}/{int((cls == c).sum())}" for c in
                                   ('Iowa', 'K18', 'Salamander', 'SoundFont', 'Otey')},
                        'auc': L.auc_fast((~rec).astype(int), p), 'p': p.tolist()}
    out['clips'] = clip_res

    # (a) eGeMAPS library notes, own-fold (XGBoost, v3 primary folds)
    print('final: eGeMAPS / CNN own-fold ...', flush=True)
    EG = load_egemaps()
    egres = egemaps_lib(iso, libv, lf, EG)
    out['egemaps'] = egres

    # (b) CNN / fusion library notes, own-fold (cnn_iso per-fold jobs, constrained-f0 variant)
    meta = pd.read_csv(os.path.join(REV, 'cnn_iso', 'cache', 'phaseo_meta.csv'))
    cn = {}
    for m_ in ('cnn', 'fus47', 'fus16'):
        J = [os.path.join(REV, 'cnn_iso', 'jobs', 'constr', f'{m_}_T2_f{k}.npz') for k in range(5)]
        if not all(os.path.exists(j) for j in J):
            cn[m_] = 'jobs missing'
            continue
        full = np.full((5, len(meta)), np.nan)
        for k, j in enumerate(J):
            z = np.load(j)
            full[k, z['rows']] = z['p']
        fo = meta['fold'].fillna(0).values.astype(int)
        own = full[fo, np.arange(len(meta))]
        m5 = np.nanmean(full, axis=0)
        src = meta['source'].values
        ccm = src == 'ccmusic'
        for name, lm in (('library mf', (src == 'library') & meta['mf_layer'].fillna(False).astype(bool).values),
                         ('library all', src == 'library'), ('algorithmic', np.isin(src, ['soundfont', 'otey']))):
            for rule, s2 in (('own', own), ('mean5', m5)):
                s_ = np.where(ccm, own, s2)
                cn[f'{m_}|{name}|{rule}'] = V.cmp3(meta['y'].values, s_, meta['grp'].values, ccm, lm)
        for lb in ('Salamander', 'FreePats', 'K18', 'Iowa'):
            lm = (src == 'library') & (meta['sub'].values == lb) & meta['mf_layer'].fillna(False).astype(bool).values
            cn[f'{m_}|{lb} mf|own'] = V.cmp3(meta['y'].values, own, meta['grp'].values, ccm, lm)
    out['cnn'] = cn

    # (5.5) bass / tuning numbers and the tuning-only AUC vs top-rated real pianos (performance-only, own-fold)
    import protocol_v2 as P
    pd_ = df[df['kind'].isin(R3.PERF_KINDS)].reset_index(drop=True)
    rc = pd.read_csv(os.path.join(L.QCACHE, 'features_cal.csv'), usecols=['file', 'rating_cal']).set_index('file')['rating_cal']
    rating = pd_['file'].map(rc).values.astype(float)
    rp = (pd_.kind == 'real_performance').values
    top = rp & (rating >= np.nanpercentile(rating[rp], 90))
    comps = {c[1]: c for c in P._comparisons(pd_)}
    scC = V.run_cv3(pd_, V.FS3['only_C'], 'T2')['own']
    sm = comps['P2'][4]
    out['tuning_top10'] = V.cmp3(pd_['y'].values, scC, pd_['grp'].values, top, sm)
    out['bass'] = {k: float(np.nanmedian(df.loc[m, c])) for k, m, c in (
        ('fund real perf', df.kind == 'real_performance', 'bass_fund_strength'),
        ('fund sampled digital', df.kind == 'sampled_digital_performance', 'bass_fund_strength'),
        ('fund renders', df.kind == 'algo_performance', 'bass_fund_strength'),
        ('fund ccmusic', df.subkind == 'ccmusic', 'bass_fund_strength'),
        ('h2h1 sampled digital', df.kind == 'sampled_digital_performance', 'bass_h2_over_h1'),
        ('detune sampled digital v2', (df.subkind == 'digital_v2') & (df.kind == 'sampled_digital_performance'), 'detune_final'),
        ('detune real top10', pd.Series(False, index=df.index), 'detune_final'))}
    out['bass']['detune real top10'] = float(np.nanmedian(pd_.loc[top, 'detune_final']))
    out['bass']['h2h1 real top10 / bottom quarter'] = (float(np.nanmedian(pd_.loc[top, 'bass_h2_over_h1'])),
                                                       float(np.nanmedian(pd_.loc[rp & (rating < np.nanpercentile(rating[rp], 25)), 'bass_h2_over_h1'])))

    # (5.4) phaseogram numbers of the 190-note selection (figure analysis) + C4
    c4 = pd.read_csv(os.path.join(V.OUT3, 'c4_phaseogram_c3.csv'))
    dr = pd.read_csv(os.path.join(REV, 'figures', 'drift_notes_all.csv')).drop_duplicates('file').set_index('file')
    c4 = c4.join(dr[['r_row', 'irreg_early', 'irreg_late']], on='file')
    ph = {}
    for c_ in ('real', 'samplelib', 'soundfont', 'otey'):
        g = c4[c4.cls == c_]
        ph[c_] = {'n': len(g), 'irreg_med': float(g['irreg'].median()), 'q1': float(g['irreg'].quantile(.25)),
                  'q3': float(g['irreg'].quantile(.75)), 'c3_med': float(g['irreg_c3'].median()),
                  'r_row_med': float(g['r_row'].median())}
    real = c4[c4.cls == 'real']
    ph['real_early_late'] = (float(real['irreg_early'].median()), float(real['irreg_late'].median()))
    for c_ in ('samplelib', 'soundfont', 'otey'):
        g = c4[c4.cls == c_]
        u, p_ = mannwhitneyu(real['irreg'], g['irreg'], alternative='two-sided')
        ph[f'real_vs_{c_}'] = {'auc': float(u / (len(real) * len(g))), 'p': float(p_)}
    out['phaseogram'] = ph

    print('final: fold-seed sweep ...', flush=True)
    out['sweep'] = sweep(df, libv, EG)
    sw0 = {k: v['seed0'] for k, v in out['sweep'].items()}
    chk = [(k, sw0[k], T7p[k]['auc']) for k in sw0 if k in T7p and abs(sw0[k] - T7p[k]['auc']) > 1e-12]
    assert not chk, f'seed-0 sweep differs from the primary run: {chk[:3]}'
    json.dump(out, open(os.path.join(V.OUT3, 'final_numbers.json'), 'w'), indent=1,
              default=lambda o: o.item() if isinstance(o, (np.floating, np.integer, np.bool_)) else str(o))
    write_md(out, T7p, T7s, CT, info, infos, time.time() - t0)
    print(f'final: wrote FINAL_NUMBERS.md ({time.time() - t0:.0f} s)')


def write_md(o, T7p, T7s, CT, info, infos, secs):
    def pr(k, fs='all47', cond='T2', what='auc'):
        a, b = T7p.get(f'{k.split("|")[0]}:{fs}:{cond}:{k.split("|")[1]}'), T7s.get(f'{k.split("|")[0]}:{fs}:{cond}:{k.split("|")[1]}')
        return a, b
    rows = []
    for label, k in (('Iso: SoundFont', 'T7iso|SoundFont single notes'), ('Iso: Otey', 'T7iso|Otey single notes'),
                     ('Iso: algorithmic (both)', 'T7iso|algorithmic (both)'),
                     ('Iso: library notes, mf (224)', 'T7iso|library notes, 4 libraries, mf layer'),
                     ('Iso: library notes, all layers', 'T7iso|library notes, 4 libraries, all layers'),
                     ('Iso: Salamander mf', 'T7iso|Salamander, mf layer'), ('Iso: Iowa mf', 'T7iso|Iowa, mf layer'),
                     ('Iso: K18 mf', 'T7iso|K18, mf layer'), ('Iso: FreePats mf', 'T7iso|FreePats, mf layer'),
                     ('Perf: SoundFont renders (P1a)', 'T7perf|P1a'), ('Perf: Otey renders (P1b)', 'T7perf|P1b'),
                     ('Perf: algorithmic renders (P1)', 'T7perf|P1'),
                     ('Perf: sampled digital (P2)', 'T7perf|P2'), ('Perf: sampled digital, long tails', 'T7perf|P2d-lt'),
                     ('Perf: sampled digital, no long tails', 'T7perf|P2d-nolt'), ('Perf: digital, line capture', 'T7perf|P2d-line'),
                     ('Perf: modelled digital (P3)', 'T7perf|P3'), ('Perf: Korg M1 (P4)', 'T7perf|P4')):
        a47, s47 = pr(k)
        a16, s16 = pr(k, 'med16')
        t1 = pr(k, 'all47', 'T1')[0]
        rows.append([label, f"{a47['n_syn'] if a47 else ''}/{a47['g_syn'] if a47 else ''}",
                     f"{fa3(a47)}; det {det(a47)}", f"{fa3(s47)}; det {det(s47)}", fa3(a16, False), fa3(s16, False),
                     fa3(t1, False)])
    md = ['# FINAL NUMBERS -- protocol v3, final run\n\n',
          'Command: `python Stattia/revision/analysis/final_v3.py` (runs run_v3.py first). Own-fold scoring throughout; '
          'T2 unless stated. PRIMARY = CODE_AUDIT B1 files and every "exclude" row of audit/real_class_exclusions.csv '
          f"removed, real performances grouped by group_id / video id ({info['v3']['screen_note']}; "
          f"{int(info['v3']['n_real_groups_v3'])} real-performance groups). SENSITIVITY = also every \"review\" row "
          f"removed ({infos['v3']['screen_note']}; {int(infos['v3']['n_real_groups_v3'])} groups). "
          '`*` = one synthetic group (no general interval). Runtime ' + f'{secs:.0f} s.\n\n',
          'Caveat: the screen changes only real performances, but the greedy GroupKFold assignment is recomputed over the '
          'whole corpus, so 5 of the 11 synthetic single-note groups (and the libraries tied to their folds) moved to other '
          'folds relative to the pre-screen v3 run (and differ again in the sensitivity corpus). All isolated-note changes '
          'versus pre-screen v3 (e.g. library mf 0.507 -> 0.621, Iowa mf 0.363 -> 0.608) are therefore fold-assignment '
          'variability, not an effect of the screen; they lie inside the grouped-bootstrap intervals.\n\n',
          '## Table 7 (detection by format)\n\n']
    fa = (T7p.get('T7iso:fa', {}), T7s.get('T7iso:fa', {}), T7p.get('T7perf:fa', {}), T7s.get('T7perf:fa', {}))
    md.append(L.md_table(['row', 'files/groups', '47 feat. primary', '47 feat. sensitivity', '16 med. primary',
                          '16 med. sensitivity', '47 feat. T1 primary'], rows))
    c = o['counts']
    md.append(f"\nFalse alarms: CCMusic {L.fpct(fa[0].get('all47 T2', np.nan))} (sens. {L.fpct(fa[1].get('all47 T2', np.nan))}), "
              f"med16 {L.fpct(fa[0].get('med16 T2', np.nan))}; T1 {L.fpct(fa[0].get('all47 T1', np.nan))}. Real performances "
              f"{L.fpct(fa[2].get('all47 T2', np.nan))} (sens. {L.fpct(fa[3].get('all47 T2', np.nan))}), med16 "
              f"{L.fpct(fa[2].get('med16 T2', np.nan))}. Real performances: {c['primary']['perf_n']} files primary, "
              f"{c['sensitivity']['perf_n']} sensitivity.\n")
    mx = o['mixed']
    md.append(f"Formats mixed (corpus-level detector, T2): real performances vs algorithmic single notes "
              f"{fa3(mx['primary']['all47'])}; det {det(mx['primary']['all47'])} (sens. {fa3(mx['sensitivity']['all47'])}); "
              f"med16 {fa3(mx['primary']['med16'], False)} (sens. {fa3(mx['sensitivity']['med16'], False)}).\n")
    md.append('\n## 5.1 Format\n\n'
              f"SoundFont single-note files with all 16 IQR = 0: {c['primary']['sf_all16_zero']} of {c['primary']['sf_n']}. "
              f"Real performances: {100 * c['primary']['perf_zero_iqr_share']:.1f}% of IQR values zero "
              f"({c['primary']['perf_zero_iqr_values']} of {c['primary']['perf_values']}); files with all 16 zero: "
              f"{c['primary']['perf_all16_zero']}; files with an undefined Group B value: {c['primary']['perf_any_B_undefined_files']} "
              f"({c['primary']['perf_B_undefined_values']} values); with an undefined Group C value: "
              f"{c['primary']['perf_any_C_undefined_files']} ({c['primary']['perf_C_undefined_values']} values). Sensitivity: "
              f"{100 * c['sensitivity']['perf_zero_iqr_share']:.1f}% zero, {c['sensitivity']['perf_all16_zero']} all-zero, "
              f"{c['sensitivity']['perf_any_B_undefined_files']} / {c['sensitivity']['perf_any_C_undefined_files']} files with "
              f"undefined B / C.\n")
    ins = o['instrument']
    md.append('\n## 5.2 Isolated notes (extras)\n\n'
              f"Instrument level (median P per instrument, exact Mann-Whitney): 8 pianos vs 4 libraries p = {ins['pianos_vs_libraries'][0]:.3g} "
              f"(P = {ins['pianos_vs_libraries'][1]:.2f}); 8 pianos vs 10 SoundFonts p = {ins['pianos_vs_soundfonts'][0]:.2g} "
              f"(P = {ins['pianos_vs_soundfonts'][1]:.2f}).\n\n")
    f0 = o['f0_restriction']
    md.append('f0 / register restriction (evaluation restricted to files with a correct primary f0 and nominal MIDI 48-76; '
              f"remaining: CCMusic {o['f0_counts']['ccmusic_ok_48_76']}, SoundFont {o['f0_counts']['sf_ok']}, Otey "
              f"{o['f0_counts']['otey_ok']}, library notes {o['f0_counts']['lib_ok']}): " +
              '; '.join(f"{n}: {fa3(f0[n + '|all'], False)} -> {fa3(f0[n + '|f0-valid & 48-76'], False)} "
                        f"(change {f0[n + '|f0-valid & 48-76']['auc'] - f0[n + '|all']['auc']:+.3f})"
                        for n in ('SoundFont', 'Otey', 'library mf', 'library all')) + '.\n\n')
    eg = o['egemaps']
    md.append(f"eGeMAPS (88 features, same XGBoost, v3 folds): algorithmic {fa3(eg['algorithmic'])}; library notes mf "
              f"own-fold {fa3(eg['library mf|own'])} (v2 rule {fa3(eg['library mf|mean5'], False)}); all layers own-fold "
              f"{fa3(eg['library all|own'])} (v2 rule {fa3(eg['library all|mean5'], False)}).\n")
    md.append('\n## 5.3 Controls (Table 8; primary corpus)\n\n')
    rows = []
    for nm in ('SoundFont', 'SoundFont, unaltered sample', 'Otey', 'library notes, mf layer'):
        rows.append([nm] + [f"{CT[f'ctrl:all47:{cd}:{nm}']['auc']:.3f} ({L.fpct(CT[f'ctrl:all47:{cd}:{nm}']['det_syn'])})"
                            for cd in ('orig', 'c1a', 'c1b', 'c3', 'c1a+c3', 'c1b+c3')])
    rows.append(['false alarms'] + [L.fpct(CT[f'ctrl:all47:{cd}:CCMusic FA']) for cd in ('orig', 'c1a', 'c1b', 'c3', 'c1a+c3', 'c1b+c3')])
    rows.append(['Otey, 16 medians'] + [f"{CT[f'ctrl:med16:{cd}:Otey']['auc']:.3f}" for cd in ('orig', 'c1a', 'c1b', 'c3', 'c1a+c3', 'c1b+c3')])
    md.append(L.md_table(['source', 'original', '1.2 s', '4 s', 'room+noise', '1.2 s+room', '4 s+room'], rows))
    md.append('\nLoop index (C5, share r > 0.9999): ' + ', '.join(f"{k.split(':', 1)[1]} {100 * v['share_09999']:.0f}%"
                                                              for k, v in CT.items() if k.startswith('c5:')) + '.\n')
    ph = o['phaseogram']
    md.append('\n## 5.4 Phaseogram and CNN (Table 9)\n\n')
    md.append(L.md_table(['source', 'notes', 'irregularity median [IQR]', 'with room+noise', 'row-to-row r'],
                         [[k, v['n'], f"{v['irreg_med']:.3f} [{v['q1']:.2f}, {v['q3']:.2f}]", f"{v['c3_med']:.2f}",
                           f"{v['r_row_med']:.3f}"] for k, v in ph.items() if isinstance(v, dict) and 'n' in v]))
    md.append(f"\nReal vs Otey AUC {ph['real_vs_otey']['auc']:.2f} (p = {ph['real_vs_otey']['p']:.1g}); vs SoundFont "
              f"{ph['real_vs_soundfont']['auc']:.2f}; vs libraries {ph['real_vs_samplelib']['auc']:.2f} (p = {ph['real_vs_samplelib']['p']:.2g}). "
              f"Real early / late half: {ph['real_early_late'][0]:.2f} / {ph['real_early_late'][1]:.2f}. After room+noise: "
              f"P(class > real) SoundFont {CT['c4:soundfont']['P_gt_real_c3']:.2f}, Otey {CT['c4:otey']['P_gt_real_c3']:.2f}.\n\n")
    cn = o['cnn']
    if isinstance(cn.get('cnn|library mf|own'), dict):
        md.append('CNN / fusion (isolated training, T2, constrained f0), own-fold: ' + '; '.join(
            f"{m_}: algorithmic {fa3(cn[f'{m_}|algorithmic|own'], False)}, library mf {fa3(cn[f'{m_}|library mf|own'])} "
            f"(v2 rule {fa3(cn[f'{m_}|library mf|mean5'], False)})" for m_ in ('cnn', 'fus47', 'fus16') if isinstance(cn.get(f'{m_}|library mf|own'), dict)) +
            '. CNN per library (mf, own-fold): ' + ', '.join(f"{lb} {cn[f'cnn|{lb} mf|own']['auc']:.2f}" for lb in ('Salamander', 'FreePats', 'K18', 'Iowa')) + '.\n')
    b = o['bass']
    r_ = o['mixed']
    md.append('\n## 5.5 Performances (extras)\n\n'
              f"Bass fundamental share: real performances {b['fund real perf']:.3f}, sampled digital {b['fund sampled digital']:.3f}, "
              f"renders {b['fund renders']:.3f}, CCMusic {b['fund ccmusic']:.3f}; bass log(h2/h1): digital {b['h2h1 sampled digital']:.2f} vs real "
              f"top-10% / bottom quarter {b['h2h1 real top10 / bottom quarter'][0]:.2f} / {b['h2h1 real top10 / bottom quarter'][1]:.2f}. "
              f"Detuning: digital_v2 {b['detune sampled digital v2']:.2f} vs top-10%-rated real {b['detune real top10']:.2f}; "
              f"tuning-only performance detector vs top-10% real: {fa3(o['tuning_top10'])}. "
              f"Dry renders effect (corpus detector, T2, own-fold): P2 {fa3(r_['primary']['renders']['without'])} -> "
              f"{fa3(r_['primary']['renders']['with'])} (sens. {fa3(r_['sensitivity']['renders']['without'], False)} -> "
              f"{fa3(r_['sensitivity']['renders']['with'], False)}).\n")
    cl = o['clips']
    md.append('\n## 5.6 Listening clips (Table 10 detector column; isolated-note detector, leave-own-group-out)\n\n' +
              '; '.join(f"{fs}: " + ', '.join(f'{k} {v}' for k, v in cl[fs]['counts'].items()) + f", AUC {cl[fs]['auc']:.3f}"
                        for fs in ('all47', 'med16')) + '. Listener numbers: unchanged (RESULTS_v2 section 6).\n')
    md.append(sweep_md(o, T7p, CT))
    open(os.path.join(V.OUT3, 'FINAL_NUMBERS.md'), 'w', encoding='utf-8').write(''.join(md))


def sweep_md(o, T7p, CT):
    sw = o['sweep']

    def mm(k, pct=False, p=False):
        v = sw[k]
        if p:
            return f"{v['median']:.2g} [{v['min']:.2g}-{v['max']:.2g}]"
        f = (lambda x: f'{100 * x:.1f}%') if pct else (lambda x: f'{x:.3f}')
        return f"{f(v['median'])} [{f(v['min'])}-{f(v['max'])}]"

    def s0(r):
        return fa3(r) if isinstance(r, dict) else 'n/a'
    md = [f'\n## Over {N_SEEDS} fold assignments (primary corpus)\n\n'
          f'Seeds 0-{N_SEEDS - 1} of the same greedy GroupKFold rule (largest group first, ties broken in a seed-permuted '
          'order); seed 0 = the assignment used everywhere above. Each cell: median [min-max] over the seeds; the seed-0 '
          'column gives the value with its grouped-bootstrap 95% CI. Per-seed values: `out_v3/fold_seed_sweep.csv`. '
          "The listening-clip detector (section 5.6) is trained on all isolated notes except the clip's own group, "
          'so it does not use folds: its values are the same for every seed.\n\n']
    rows = []
    t7 = [('Iso: SoundFont', 'T7iso:{}:T2:SoundFont single notes'), ('Iso: Otey', 'T7iso:{}:T2:Otey single notes'),
          ('Iso: algorithmic (both)', 'T7iso:{}:T2:algorithmic (both)'),
          ('Iso: library notes, mf', 'T7iso:{}:T2:library notes, 4 libraries, mf layer'),
          ('Iso: library notes, all layers', 'T7iso:{}:T2:library notes, 4 libraries, all layers')]
    t7 += [(f'Iso: {lb} mf', 'T7iso:{}:T2:' + lb + ', mf layer') for lb in ('Salamander', 'Iowa', 'K18', 'FreePats')]
    t7 += [(f'Perf: {c}', 'T7perf:{}:T2:' + c) for c in ('P1a', 'P1b', 'P1', 'P2', 'P2d-lt', 'P2d-nolt', 'P2d-line',
                                                       'P3', 'P4')]
    for lab, k in t7:
        ka, km = k.format('all47'), k.format('med16')
        if ka in sw:
            rows.append([lab, s0(T7p.get(ka)), mm(ka), f"{sw[km]['seed0']:.3f}" if km in sw else 'n/a',
                         mm(km) if km in sw else 'n/a'])
    for lab, k in (('False alarms, CCMusic', 'T7iso:fa:{} T2'), ('False alarms, real performances', 'T7perf:fa:{} T2')):
        rows.append([lab, f"{100 * sw[k.format('all47')]['seed0']:.1f}%", mm(k.format('all47'), pct=True),
                     f"{100 * sw[k.format('med16')]['seed0']:.1f}%", mm(k.format('med16'), pct=True)])
    md.append('### Table 7\n\n' + L.md_table(['row', '47 feat., seed 0 [CI]', '47 feat., median [min-max]',
                                              '16 med., seed 0', '16 med., median [min-max]'], rows))
    rows = []
    for nm in ('SoundFont', 'SoundFont, unaltered sample', 'Otey', 'library notes, mf layer', 'library notes, all layers',
               'CCMusic FA'):
        cells = [nm]
        for c in ('orig', 'c1a', 'c1b', 'c3', 'c1a+c3', 'c1b+c3'):
            cells.append(mm(f'ctrl:all47:{c}:{nm}', pct=(nm == 'CCMusic FA')))
        rows.append(cells)
    md.append('\n### Table 8 controls (47 features), median [min-max]\n\n' +
              L.md_table(['source', 'original', '1.2 s', '4 s', 'room+noise', '1.2 s+room', '4 s+room'], rows))
    rows = []
    for nm, lab in (('libraries', '8 pianos vs 4 libraries'), ('soundfonts', '8 pianos vs 10 SoundFonts')):
        rows.append([lab, f"p = {sw[f'instr:pianos_vs_{nm}:p']['seed0']:.3g}, P = {sw[f'instr:pianos_vs_{nm}:P']['seed0']:.2f}",
                     'p ' + mm(f'instr:pianos_vs_{nm}:p', p=True) + '; P ' + mm(f'instr:pianos_vs_{nm}:P')])
    eg = o['egemaps']
    for k, lab in (('library mf|own', 'eGeMAPS: library notes, mf'), ('library all|own', 'eGeMAPS: library notes, all layers'),
                   ('algorithmic', 'eGeMAPS: algorithmic')):
        rows.append([lab, s0(eg[k]), mm(f'egemaps:{k}')])
    for fs in ('all47', 'med16'):
        rows.append([f'Listening clips, detector AUC ({fs})', f"{o['clips'][fs]['auc']:.3f}",
                     'fold-independent (same for every seed)'])
    md.append('\n### Instrument-level tests, eGeMAPS, clips\n\n' + L.md_table(['number', 'seed 0', 'median [min-max]'], rows))
    return ''.join(md)


if __name__ == '__main__':
    main()
