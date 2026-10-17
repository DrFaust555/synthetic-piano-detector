"""Protocol v2 follow-up -- what carries the performance-regime separations (P1, P2)?

Called by run_all.py (section 7 of RESULTS_v2.md); same corpus, fixed folds, XGBoost (n_jobs=1),
T1 / T2 and stratified grouped bootstrap as protocol_v2.py.

  7.1 feature-group ablation: train on ONLY one group (A medians 16, A IQRs 16, B register 8,
      C tuning 7) or on all 47 MINUS one group; plus two decay-specific sets (the 4 decay/sustain
      medians alone, and the 12 other Group-A medians).
  7.2 attribution for P1 / P2 under T2: exact TreeSHAP from XGBoost itself (pred_contribs; the `shap`
      package is not needed), computed with the same fold models that produce the scores, plus
      total-gain importance averaged over the 5 fold models.
  7.3 "too-perfect tuning" test: Group C distributions per source kind; Mann-Whitney real
      performances vs sample-based demos.
  7.4 "dry channel" test: decay / sustain medians per source kind; same tests.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

import pv2_lib as L
import protocol_v2 as P
from pv2_lib import FEATURE_COLS, MED16, FEATSETS, OUT
from quality.pipeline import BASS_FEATURE_COLS, TUNING

IQR16 = [c for c in FEATURE_COLS if c.endswith('_iqr')]
GROUPS = {'A_med': MED16, 'A_iqr': IQR16, 'B': list(BASS_FEATURE_COLS), 'C': list(TUNING)}
DECAY4 = ['log_t60_med', 'sustain_db_1s_med', 'hf_decay_excess_med', 'bright_decay_med']
ABL = {'all47': list(FEATURE_COLS)}
for g, cols in GROUPS.items():
    ABL[f'only_{g}'] = cols
for g, cols in GROUPS.items():
    ABL[f'all_minus_{g}'] = [c for c in FEATURE_COLS if c not in cols]
ABL['only_decay4'] = DECAY4
ABL['A_med_minus_decay4'] = [c for c in MED16 if c not in DECAY4]
ABL_LABEL = {'all47': 'all 47', 'only_A_med': 'only A medians (16)', 'only_A_iqr': 'only A IQRs (16)',
             'only_B': 'only B register (8)', 'only_C': 'only C tuning (7)',
             'all_minus_A_med': 'all minus A medians (31)', 'all_minus_A_iqr': 'all minus A IQRs (31)',
             'all_minus_B': 'all minus B (39)', 'all_minus_C': 'all minus C (40)',
             'only_decay4': 'only 4 decay/sustain medians', 'A_med_minus_decay4': '12 other A medians'}
GROUP_OF = {c: g for g, cols in GROUPS.items() for c in cols}

CMP_IDS = ['P1', 'P1a', 'P1b', 'P2', 'P2a', 'P2d', 'P3', 'P4', 'I1a', 'I1b']
CMP_V2 = []


def _libv(df):
    libs = P.library_note_files()
    lf = L.extract_cached(list(libs['path']), 'library_notes')
    libs = pd.concat([libs.reset_index(drop=True), lf], axis=1)
    libs['grp'] = 'syn:lib:' + libs['library']
    libs['valid'] = (libs[FEATURE_COLS].notna().sum(axis=1) >= 30) & (libs['error'].fillna('') == '')
    return libs[libs['valid']].reset_index(drop=True)


def _kinds(df):
    """(label, mask) of the source kinds compared in 7.3 / 7.4."""
    sk, k = df['subkind'].values, df['kind'].values
    cap = df['capture'].fillna('').astype(str).values
    out = [('real perf: manifest', sk == 'manifest'), ('real perf: youtube_extra', sk == 'youtube_extra'),
           ('real perf: all', k == 'real_performance'),
           ('sampled demos: old', (sk == 'old_demo') & (k == 'sampled_digital_performance'))]
    if ((sk == 'digital_v2') & (k == 'sampled_digital_performance')).any():
        out.append(('sampled demos: digital_v2', (sk == 'digital_v2') & (k == 'sampled_digital_performance')))
        for c in ('line', 'mic', 'unknown'):
            m = (sk == 'digital_v2') & (k == 'sampled_digital_performance') & (cap == c)
            if m.any():
                out.append((f'sampled demos: digital_v2, {c}', m))
    out += [('modelled digital', k == 'modelled_digital'), ('digital other (GP-3)', k == 'digital_other'),
            ('SC-55 module', k == 'sound_module'),
            ('SoundFont perf renders', sk == 'sf2_perf'), ('Otey perf renders', sk == 'otey_perf'),
            ('Korg M1', k == 'm1'), ('CCMusic isolated (ref.)', sk == 'ccmusic')]
    return [(a, b) for a, b in out if b.any()]


def _q(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if not len(v):
        return 'n/a'
    q1, md, q3 = np.percentile(v, [25, 50, 75])
    return f'{md:.3g} [{q1:.3g}, {q3:.3g}]'


def _mw(a, b):
    """two-sided Mann-Whitney + P(b > a) (ties 1/2)."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan, len(b)
    u, p = mannwhitneyu(b, a, alternative='two-sided')
    return float(p), float(u / (len(a) * len(b))), len(b)


def _dist_section(df, feats, title, note, tag, key):
    kinds = _kinds(df)
    md = [f'### {title}\n\n{note}\n\n']
    rows = []
    for f in feats:
        rows.append([f] + [_q(df.loc[m, f]) for _, m in kinds])
    md.append('Median [Q1, Q3] per file (n files: ' + ', '.join(f'{a} {int(m.sum())}' for a, m in kinds) + '):\n\n')
    md.append(L.md_table(['feature'] + [a for a, _ in kinds], rows))
    real = (df['kind'] == 'real_performance').values
    refs = [(a, m) for a, m in kinds if a.startswith('sampled demos') or a in
            ('SoundFont perf renders', 'Otey perf renders', 'Korg M1', 'modelled digital')]
    rows, csv = [], []
    for f in feats:
        cells = [f]
        for a, m in refs:
            p, pb, nb = _mw(df.loc[real, f], df.loc[m, f])
            cells.append('n/a' if not np.isfinite(pb) else f'{pb:.2f} (p={p:.2g})')
            csv.append({'test': tag, 'feature': f, 'vs': a, 'n_real': int(np.isfinite(df.loc[real, f]).sum()),
                        'n_other': nb, 'mw_p_two_sided': p, 'P_other_gt_real': pb})
            key[f'{tag}:{f}:{a}'] = {'p': p, 'P_gt': pb}
        pm, pbm, _ = _mw(df.loc[df.subkind == 'youtube_extra', f], df.loc[df.subkind == 'manifest', f])
        cells.append(f'{pbm:.2f} (p={pm:.2g})')
        rows.append(cells)
    md.append('\nMann-Whitney against ALL real performances (2576 files): cell = P(other > real) '
              '[0.5 = same distribution; < 0.5 = the other kind is lower] and two-sided p. File-level test: '
              'demos cluster in 5 brands, so the p-values overstate the independent evidence. '
              'Last column: real manifest vs real youtube_extra (within-real heterogeneity).\n\n')
    md.append(L.md_table(['feature'] + [f'vs {a}' for a, _ in refs] + ['manifest vs youtube_extra'], rows))
    return ''.join(md), csv


def _shap_cv(df, cond, cols):
    """Re-fit the fold models of P.run_cv (identical data / params / seed -> identical models) and
    return scores, exact TreeSHAP contributions (log-odds) and fold-averaged total gain."""
    import xgboost as xgb
    X = df[cols].values.astype(float)
    y = df['y'].values
    fold = df['fold'].values
    trainable = np.ones(len(df), bool) if cond == 'T1' else ~df['kind'].isin(L.SB_KINDS).values
    spw = float(np.sum(y[trainable] == 0)) / max(1, int(np.sum(y[trainable] == 1)))
    oof = np.full(len(df), np.nan); zs = np.zeros(len(df))
    con = np.zeros((len(df), len(cols) + 1)); zc = np.zeros_like(con)
    gain = np.zeros(len(cols))
    dm = xgb.DMatrix(X, missing=np.nan)
    for k in range(L.N_FOLDS):
        tr = trainable & (fold != k)
        te = fold == k
        m = L.make_model('xgb', spw)
        m.fit(X[tr], y[tr])
        pa = m.predict_proba(X)[:, 1]
        oof[te] = pa[te]; zs += pa / L.N_FOLDS
        c = m.get_booster().predict(dm, pred_contribs=True)
        con[te] = c[te]; zc += c / L.N_FOLDS
        g = m.get_booster().get_score(importance_type='total_gain')
        tot = sum(g.values()) or 1.0
        for f, v in g.items():
            gain[int(f[1:])] += v / tot / L.N_FOLDS
    scores = np.where(trainable, oof, zs)
    contrib = np.where(trainable[:, None], con, zc)
    return scores, contrib, gain


def main(log=print):
    md, key, csv_rows = ['\n## 7. Follow-up: what carries the performance-regime separations (P1, P2)?\n\n'], {}, []
    for k_, v in ABL.items():
        FEATSETS[k_] = v                                   # shared dict -> visible to protocol_v2.run_cv
    df, info = P.build_corpus(corrected=True, log=log)
    libv = _libv(df)
    comps = {c[1]: c for c in P._comparisons(df)}
    ids = [i for i in CMP_IDS + CMP_V2 if i in comps and comps[i][4].any()]
    lib_fold = {g: int(df.loc[df.grp == g, 'fold'].iloc[0]) for g in libv['grp'].unique()}
    cc = (df.kind == 'real_isolated').values
    mf = libv['mf_layer'].values
    y, grp = df['y'].values, df['grp'].values

    # ---------------- 7.1 ablation
    res = []
    scores = {}
    for fs in ABL:
        for cond in ('T1', 'T2'):
            sc, ext = P.run_cv(df, 'xgb', fs, cond, ext={'libnotes': libv})
            scores[(fs, cond)] = sc
            for cid in ids:
                _, _, label, rm, sm = comps[cid]
                res.append({'featset': fs, 'cond': cond, 'id': cid, **L.compare(y, sc, grp, rm, sm)})
            lsc = (ext['libnotes'][libv['grp'].map(lib_fold).values, np.arange(len(libv))] if cond == 'T1'
                   else ext['libnotes'].mean(axis=0))
            yy = np.r_[np.zeros(cc.sum()), np.ones(mf.sum())]
            r = L.compare(yy, np.r_[sc[cc], lsc[mf]], np.r_[grp[cc], libv['grp'].values[mf]], yy == 0, yy == 1)
            res.append({'featset': fs, 'cond': cond, 'id': 'libnotes_mf', **r})
        log(f'  ablation {fs} done')
    ab = pd.DataFrame(res)
    ab.to_csv(os.path.join(OUT, 'followup_ablation.csv'), index=False)
    cols_out = ids + ['libnotes_mf']
    md.append('### 7.1 Feature-group ablation (XGBoost, same fixed folds)\n\n')
    md.append('Groups: A medians = per-note descriptors 1-16 (median over the file\'s notes); A IQRs = 17-32; '
              'B = register descriptors 33-40 (bass / treble / deep-bass, not detrended); C = tuning 41-47 '
              '(file level). "decay/sustain medians" = log_t60_med, sustain_db_1s_med, hf_decay_excess_med, '
              'bright_decay_med. Comparison ids as in section 2 (P1 real performances vs algorithmic performance renders; '
              'P1a SoundFont, P1b Otey; P2 vs sample-based digital pianos (old demos + digital_v2), P2a old demos only, P2d '
              'digital_v2 only; P3 modelled digital pianos (Roland); P4 Korg M1; '
              'I1a/I1b CCMusic notes vs SoundFont / Otey single notes; libnotes_mf = CCMusic notes vs library notes, mf '
              'layer, as section 3). n (real / synthetic files): ' +
              ', '.join(f"{i} {int(comps[i][3].sum())}/{int(comps[i][4].sum())}" for i in ids) +
              f', libnotes_mf {int(cc.sum())}/{int(mf.sum())}.\n\n')
    for cond in ('T1', 'T2'):
        md.append(f'**{cond}: AUC [95% CI]**\n\n')
        tb = []
        for fs in ABL:
            row = [ABL_LABEL[fs]]
            for cid in cols_out:
                r = ab[(ab.featset == fs) & (ab.cond == cond) & (ab.id == cid)].iloc[0]
                row.append(L.fci(r))
            tb.append(row)
        md.append(L.md_table(['feature set'] + cols_out, tb))
        md.append('\n')
    md.append('**Detection rate (P > 0.5) of the synthetic files, T1 / T2**\n\n')
    tb = []
    for fs in ABL:
        row = [ABL_LABEL[fs]]
        for cid in cols_out:
            a = ab[(ab.featset == fs) & (ab.cond == 'T1') & (ab.id == cid)].iloc[0]
            b = ab[(ab.featset == fs) & (ab.cond == 'T2') & (ab.id == cid)].iloc[0]
            row.append(f"{L.fpct(a['det_syn'])} / {L.fpct(b['det_syn'])}")
        tb.append(row)
    md.append(L.md_table(['feature set'] + cols_out, tb))
    fa = []
    for fs in ABL:
        a = ab[(ab.featset == fs) & (ab.cond == 'T2') & (ab.id == 'P1')].iloc[0]
        i = ab[(ab.featset == fs) & (ab.cond == 'T2') & (ab.id == 'I1a')].iloc[0]
        fa.append(f"{ABL_LABEL[fs]} {L.fpct(a['fa_real'])} / {L.fpct(i['fa_real'])}")
    md.append('\nReal false alarms under T2 (real performances / CCMusic notes): ' + '; '.join(fa) + '.\n\n')
    for _, r in ab.iterrows():
        key[f"abl:{r['featset']}:{r['cond']}:{r['id']}"] = {'auc': r['auc'], 'ci': (r['ci_lo'], r['ci_hi']),
                                                             'det': r['det_syn'], 'fa': r['fa_real']}

    # ---------------- 7.2 attribution (T2, all 47)
    sc, contrib, gain = _shap_cv(df, 'T2', list(FEATURE_COLS))
    chk = P.run_cv(df, 'xgb', 'all47', 'T2')[0]
    same = float(np.nanmax(np.abs(chk - sc)))
    md.append('### 7.2 Attribution for P1 and P2 (XGBoost all 47, T2)\n\n')
    md.append('Exact TreeSHAP values from XGBoost (`pred_contribs`, log-odds units) of the very fold models that '
              f'produce the section-2 scores (max |score difference| to section 2 = {same:.1e}); out-of-fold rows use '
              'their fold model, rows excluded from training under T2 (demos, M1) the mean over the 5 fold models. '
              '"Separation" = mean SHAP over the synthetic files minus mean SHAP over the real performances '
              '(positive = the feature pushes this synthetic kind towards "synthetic" relative to real performances). '
              '"gain" = total-gain share, mean over the 5 fold models (model-wide, not comparison-specific).\n\n')
    shap_rows = []
    for cid in ('P1', 'P1a', 'P1b', 'P2', 'P2d', 'P3'):
        if cid not in comps or not comps[cid][4].any():
            continue
        _, _, label, rm, sm = comps[cid]
        sep = contrib[sm, :-1].mean(axis=0) - contrib[rm, :-1].mean(axis=0)
        mabs = np.abs(contrib[rm | sm, :-1]).mean(axis=0)
        for j, f in enumerate(FEATURE_COLS):
            shap_rows.append({'id': cid, 'feature': f, 'group': GROUP_OF[f], 'separation': sep[j],
                              'mean_abs_shap': mabs[j], 'gain_share': gain[j]})
        order = np.argsort(-sep)[:10]
        tot = sep[sep > 0].sum()
        grp_share = {g: float(sep[[FEATURE_COLS.index(c) for c in cols]].sum()) for g, cols in GROUPS.items()}
        key[f'shap:{cid}:group_sep'] = grp_share
        key[f'shap:{cid}:top10'] = [(FEATURE_COLS[j], float(sep[j])) for j in order]
        md.append(f'**{cid}** ({label}; {int(rm.sum())} real vs {int(sm.sum())} synthetic files). Separation summed by '
                  'group (log-odds): ' + ', '.join(f'{g} {v:+.2f}' for g, v in grp_share.items()) +
                  f'; total {sep.sum():+.2f}.\n\n')
        md.append(L.md_table(['rank', 'feature', 'group', 'separation', 'share of positive separation',
                              'mean |SHAP| (both classes)', 'gain share'],
                             [[i + 1, FEATURE_COLS[j], GROUP_OF[FEATURE_COLS[j]], f'{sep[j]:+.3f}',
                               f'{100 * sep[j] / tot:.0f}%' if tot > 0 and sep[j] > 0 else '-',
                               f'{mabs[j]:.3f}', f'{100 * gain[j]:.1f}%'] for i, j in enumerate(order)]))
        md.append('\n')
    pd.DataFrame(shap_rows).to_csv(os.path.join(OUT, 'followup_shap.csv'), index=False, float_format='%.5f')
    og = np.argsort(-gain)[:10]
    md.append('Model-wide total-gain top 10 (T2, all 47): ' +
              ', '.join(f'{FEATURE_COLS[j]} ({GROUP_OF[FEATURE_COLS[j]]}) {100 * gain[j]:.1f}%' for j in og) + '.\n\n')

    # ---------------- 7.3 tuning, 7.4 channel
    t_md, t_csv = _dist_section(
        df, list(TUNING), '7.3 "Too-perfect tuning" test: Group C (tuning) features',
        'All seven are "more detuned = larger": detune_final (integral detuning 0-100, 0 = perfect), unison_index '
        '(unison spread 0-100), global_index (note-to-note inconsistency 0-100, 0 below 12 cents scatter), '
        'med_spread_cents (median resolved unison split), frac_split (share of notes with a resolved split), '
        'global_scatter (residual pitch scatter, cents), ref_offset_abs (|reference - A440|, cents). The hypothesis '
        'predicts digital pianos BELOW real performances (P(other > real) < 0.5).', 'tuning', key)
    c_md, c_csv = _dist_section(
        df, DECAY4 + ['sustain_db_1s_iqr', 'log_t60_iqr'],
        '7.4 "Dry channel" test: decay / sustain features',
        'Group-A medians are register-detrended residuals (0 = the corpus trend); log_t60 = log10 T60 of the note '
        'window, sustain_db_1s = level 1 s after onset re peak, hf_decay_excess = HF minus LF energy change over the '
        'note, bright_decay = change of centroid/f0 early -> late. In performances the analysis window ends at the next '
        'onset (<= 2.5 s), so these measure the early decay as recorded, including the room. A dry (direct / line) '
        'signal predicts faster, cleaner decays than room-miked real pianos.', 'channel', key)
    md.append(t_md + '\n' + c_md)
    pd.DataFrame(t_csv + c_csv).to_csv(os.path.join(OUT, 'followup_tuning_channel.csv'), index=False, float_format='%.5g')

    # per-file table of the demos (transparency)
    dm = df[(df.subkind == 'old_demo')]
    show = ['detune_final', 'unison_index', 'global_index', 'frac_split', 'global_scatter'] + DECAY4
    md.append('\n**Per file, old digital demos** (P = XGB all 47 T2 score):\n\n')
    md.append(L.md_table(['file', 'kind', 'group', 'P'] + show,
                         [[r['file'].split('/')[-1][:38], r['kind'].replace('_performance', ''), r['grp'].split(':', 1)[1],
                           f"{sc[i]:.3f}"] + [L.f3(r[c]) for c in show] for i, r in dm.iterrows()]))
    rp = df.kind == 'real_performance'
    md.append('Real performances, median: ' + ', '.join(f"{c} {df.loc[rp, c].median():.3g}" for c in show) + '.\n')
    v2m = df['subkind'].isin(['digital_v2', 'modelled_v2']).values
    if v2m.any():
        show2 = ['bass_fund_strength'] + show
        rows = []
        for (b, kd), g in df[v2m].groupby([df.loc[v2m, 'grp'], df.loc[v2m, 'kind']]):
            rows.append([b.split(':', 1)[1], kd.replace('_performance', ''), len(g), g['model_key'].nunique(),
                         f"{np.median(sc[g.index]):.3f}"] + [f'{g[c].median():.3g}' for c in show2])
        rows.append(['real performances', '', int(rp.sum()), '', f'{np.median(sc[rp.values]):.3f}'] +
                    [f'{df.loc[rp, c].median():.3g}' for c in show2])
        md.append('\n**digital_v2 by brand** (medians; P = XGB all 47 T2 score):\n\n')
        md.append(L.md_table(['brand', 'kind', 'files', 'models', 'median P'] + show2, rows))

    # ---------------- 7.5 "too perfect" or just "in tune"?  rating-stratified real class
    md.append(_rating_section(df, scores, comps, key))
    return ''.join(md), key


def _rating_section(df, scores, comps, key):
    """Real performances split by their calibrated quality rating (features_cal.csv `rating_cal`, read-only):
    are the digital sources better tuned than real pianos, or only than the average YouTube piano?"""
    rc = pd.read_csv(os.path.join(L.QCACHE, 'features_cal.csv'), usecols=['file', 'rating_cal'])
    rating = df['file'].map(rc.set_index('file')['rating_cal']).values.astype(float)
    rp = (df.kind == 'real_performance').values & np.isfinite(rating)
    qs = np.percentile(rating[rp], [25, 50, 75, 90])
    bins = [('rating < q25', rp & (rating < qs[0])), ('q25-q50', rp & (rating >= qs[0]) & (rating < qs[1])),
            ('q50-q75', rp & (rating >= qs[1]) & (rating < qs[2])), ('q75-q90', rp & (rating >= qs[2]) & (rating < qs[3])),
            ('top 10% rated', rp & (rating >= qs[3]))]
    sk, k = df['subkind'].values, df['kind'].values
    others = [('sampled demos (old)', (sk == 'old_demo') & (k == 'sampled_digital_performance')),
              ('SoundFont perf', sk == 'sf2_perf'), ('Otey perf', sk == 'otey_perf'), ('Korg M1', k == 'm1'),
              ('CCMusic (studio, isolated)', sk == 'ccmusic')]
    if ((sk == 'digital_v2') & (k == 'sampled_digital_performance')).any():
        others.insert(1, ('sampled demos (digital_v2)', (sk == 'digital_v2') & (k == 'sampled_digital_performance')))
    feats = ['detune_final', 'unison_index', 'frac_split', 'global_scatter', 'bass_fund_strength', 'bass_h2_over_h1',
             'hf_decay_excess_iqr', 'bright_decay_med']
    md = ['\n### 7.5 "Too-perfect" tuning or simply "in tune"? Real performances split by quality rating\n\n']
    md.append('`rating_cal` = the calibrated 0-99 instrument-quality rating stored with every real file in '
              f'`features_cal.csv` (real performances: median {np.median(rating[rp]):.1f}; bin edges at the 25/50/75/90th '
              f'percentiles = {", ".join(f"{q:.1f}" for q in qs)}). Raters hear tuning, so a negative rating-tuning '
              'correlation is expected; the question is whether well-rated real pianos reach the tuning values of '
              'the digital sources. Median per group; last column = Spearman rho of the feature with rating_cal over '
              'all real performances. bass_fund_strength / bass_h2_over_h1 (Group B: share of the fundamental in bass '
              'notes, log h2/h1) are included as recording-chain indicators.\n\n')
    rows = []
    from scipy.stats import spearmanr
    for f in feats:
        rho = spearmanr(rating[rp], df.loc[rp, f], nan_policy='omit').correlation
        rows.append([f] + [f'{np.nanmedian(df.loc[m, f]):.3g}' for _, m in bins] +
                    [f'{np.nanmedian(df.loc[m, f]):.3g}' if m.any() else 'n/a' for _, m in others] + [f'{rho:+.2f}'])
        key[f'rating:{f}'] = {'by_bin': [float(np.nanmedian(df.loc[m, f])) for _, m in bins], 'rho': float(rho)}
    md.append(L.md_table(['feature'] + [f'real: {a} (n={int(m.sum())})' for a, m in bins] +
                         [f'{a} (n={int(m.sum())})' for a, m in others] + ['rho(rating)'], rows))
    top = bins[-1][1]
    bot = bins[0][1]
    md.append('\nAUC [95% CI] (T2) with the REAL class restricted by rating; synthetic class unchanged. If the separation '
              'were "too-perfect tuning", it should collapse against the well-tuned top-rated pianos for the tuning-only '
              'model, and so should the all-47 model if tuning carried it.\n\n')
    rows = []
    y, grp = df['y'].values, df['grp'].values
    for cid in ('P1', 'P1a', 'P1b', 'P2', 'P2d', 'P3'):
        if cid not in comps or not comps[cid][4].any():
            continue
        sm = comps[cid][4]
        for fs in ('all47', 'only_C', 'only_B', 'only_A_iqr', 'only_A_med', 'all_minus_C', 'all_minus_B'):
            sc = scores[(fs, 'T2')]
            cells = [cid, ABL_LABEL[fs]]
            for lbl, rmask in (('all real perf', rp), ('top 10% rated', top), ('rating < q25', bot)):
                r = L.compare(y, sc, grp, rmask, sm)
                cells.append(L.fci(r))
                key[f'ratingauc:{cid}:{fs}:{lbl}'] = {'auc': r['auc'], 'ci': (r['ci_lo'], r['ci_hi'])}
            rows.append(cells)
    md.append(L.md_table(['id', 'feature set', f'vs all real perf (n={int(rp.sum())})', f'vs top 10% rated (n={int(top.sum())})',
                          f'vs rating < q25 (n={int(bot.sum())})'], rows))
    return ''.join(md)


if __name__ == '__main__':
    text, _ = main()
    print(text)
