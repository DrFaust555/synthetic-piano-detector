"""Evaluate the format-matched (isolated-note) CNN / fusion runs; writes every table of RESULTS_CNN_ISO.md.

Scoring rules (protocol v2, RESULTS_v2 s.3-4):
  * CCMusic, SoundFont and Otey notes: out-of-fold (the fold model whose test fold holds the file).
  * library notes: T1 -> the fold model whose test fold holds that library (folds.csv: FreePats/Iowa fold 1,
    K18/Salamander fold 2); T2 -> mean of the 5 fold models (no library in training).
  * listening clips: primary = leave-own-group-out models on the T2 population (Otey clips: model without
    Otey; Casio CA-110 clips: model without that SoundFont; library clips: full model); sensitivity = the CV
    fold models (Otey clips by the fold model whose test fold holds Otey, Casio clips likewise, library clips
    as library notes).
Metrics: pv2_lib.compare (AUC, stratified grouped bootstrap B=2000 seed 0, detection P>0.5, mean P), clips:
AUC with the s.4 clip bootstrap (clips resampled within each of the 5 classes, B=2000, seed 0).
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

import numpy as np                     # noqa: E402
import pandas as pd                    # noqa: E402

import cnn_iso_lib as CL               # noqa: E402
from cnn_iso_lib import L              # noqa: E402

VARIANTS = ['constr', 'det48']
MODELS = ['cnn', 'fus47', 'fus16']
MLAB = {'cnn': 'CNN only', 'fus47': 'Fusion, CNN + 47 features', 'fus16': 'Fusion, CNN + 16 Group-A medians'}
CONDS = ['T1', 'T2']
COMPS = (['SoundFont renders', 'Otey notes', 'algorithmic (SoundFont + Otey)',
          'library notes: all 4 libraries, mf layer', 'library notes: all 4 libraries, all layers']
         + [f'library notes: {lb}, mf layer' for lb in CL.LIBS]
         + [f'library notes: {lb}, all layers' for lb in CL.LIBS])
KEY_COMPS = COMPS[:5]


# ----------------------------------------------------------------------------- assemble scores
def _job(v, m, c, s):
    p = os.path.join(CL.JOBS, v, f'{m}_{c}_{s}.npz')
    return np.load(p) if os.path.exists(p) else None


def _mean(v):
    v = v[np.isfinite(v)]
    return float(v.mean()) if len(v) else np.nan


def assemble(meta):
    N = len(meta)
    fold = meta['fold'].values
    src = meta['source'].values
    sub = meta['sub'].values
    lib_fold = meta[src == 'library'].groupby('sub')['fold'].first().to_dict()
    gfold = meta[src != 'clip'].groupby('grp')['fold'].first().to_dict()
    clip_fold = np.array([gfold[CL.OTEY_G] if s == 'Otey' else gfold[CL.CASIO_G] if s == CL.ALG[0]
                          else lib_fold.get(s, -1) for s in sub])
    key = pd.read_csv(os.path.join(CL.KIT, 'answer_key.csv'))
    src_of = {'Stattia/listening_kit/web/' + c + '.wav': f for c, f in zip(key['clip'], key['src'])}
    row_of = {f: i for i, f in enumerate(meta['file'])}
    src_row = np.array([row_of.get(src_of.get(f, ''), -1) for f in meta['file']])
    scores, info = {}, {}
    for v in VARIANTS:
        for m in MODELS:
            for c in CONDS:
                J = [_job(v, m, c, f'f{k}') for k in range(5)]
                if any(j is None for j in J):
                    continue
                full = np.full((5, N), np.nan)
                for k, j in enumerate(J):
                    full[k, j['rows']] = j['p']
                s = np.full(N, np.nan)
                for i in range(N):
                    if src[i] in ('ccmusic', 'soundfont', 'otey'):
                        s[i] = full[fold[i], i]
                    elif src[i] == 'library':
                        s[i] = full[fold[i], i] if c == 'T1' else _mean(full[:, i])
                    else:                                     # clip, fold-model rule (sensitivity)
                        if sub[i] in CL.REC:
                            s[i] = full[clip_fold[i], i] if c == 'T1' else _mean(full[:, i])
                        else:
                            s[i] = full[clip_fold[i], i]
                scores[(v, m, c)] = s
                if c == 'T2':                                 # the clip's unprocessed source note, same models
                    s2 = np.full(N, np.nan)
                    for i in np.flatnonzero(src == 'clip'):
                        r = src_row[i]
                        if r >= 0:
                            s2[i] = s[r]
                    scores[(v, m, 'T2src')] = s2
                info[(v, m, c)] = {'seconds': float(sum(float(j['seconds']) for j in J)),
                                   'n_train': [int(j['n_train']) for j in J]}
        for m in MODELS:                                      # LOGO clip models (T2 population)
            J = {s: _job(v, m, 'T2', s) for s in ('logo_full', 'logo_noOtey', 'logo_noCasio')}
            if any(j is None for j in J.values()):
                continue
            s = np.full(N, np.nan)
            for i in np.flatnonzero(src == 'clip'):
                key = 'logo_noOtey' if sub[i] == 'Otey' else 'logo_noCasio' if sub[i] == CL.ALG[0] else 'logo_full'
                j = J[key]
                pos = np.flatnonzero(j['rows'] == i)
                if len(pos):
                    s[i] = j['p'][pos[0]]
            scores[(v, m, 'LOGO')] = s
            s2 = np.full(N, np.nan)                         # unprocessed source note, same LOGO model
            for i in np.flatnonzero(src == 'clip'):
                key = 'logo_noOtey' if sub[i] == 'Otey' else 'logo_noCasio' if sub[i] == CL.ALG[0] else 'logo_full'
                pos = np.flatnonzero(J[key]['rows'] == src_row[i])
                if src_row[i] >= 0 and len(pos):
                    s2[i] = J[key]['p'][pos[0]]
            scores[(v, m, 'LOGOsrc')] = s2
            info[(v, m, 'LOGO')] = {'seconds': float(sum(float(j['seconds']) for j in J.values())),
                                    'n_train': [int(j['n_train']) for j in J.values()]}
    return scores, info


# ----------------------------------------------------------------------------- comparisons
def comparisons(meta, s, restrict=None, boot=True):
    y, grp, src, sub = meta['y'].values, meta['grp'].values.astype(str), meta['source'].values, meta['sub'].values
    mf = meta['mf_layer'].values.astype(bool)
    base = np.isfinite(s)
    if restrict is not None:
        base &= restrict
    cc = base & (src == 'ccmusic')
    lib = base & (src == 'library')
    masks = [base & (src == 'soundfont'), base & (src == 'otey'), base & np.isin(src, ['soundfont', 'otey']),
             lib & mf, lib] + [lib & mf & (sub == lb) for lb in CL.LIBS] + [lib & (sub == lb) for lb in CL.LIBS]
    out = {}
    pp = np.where(np.isfinite(s), s, 0.0)
    for name, msk in zip(COMPS, masks):
        out[name] = L.compare(y, pp, grp, cc, msk, boot=boot)
    return out


def clip_metrics(meta, s):
    cl = meta['source'].values == 'clip'
    cls = meta['sub'].values[cl]
    p = s[cl]
    rec = np.isin(cls, CL.REC)
    alg = ~rec
    oldreal = np.isin(cls, ['Iowa', 'K18'])
    sal = cls == 'Salamander'
    cnt = {c: f"{int((p[cls == c] > 0.5).sum())}/{int((cls == c).sum())}" for c in CL.REC + CL.ALG}
    a1 = L.auc_fast(alg[rec | alg].astype(int), p[rec | alg])
    rng = np.random.default_rng(L.SEED)
    idx_by = {c: np.where(cls == c)[0] for c in CL.REC + CL.ALG}
    v = np.empty(L.N_BOOT)
    for b in range(L.N_BOOT):
        ii = np.concatenate([rng.choice(ix, len(ix), replace=True) for ix in idx_by.values()])
        mm = rec[ii] | alg[ii]
        v[b] = L.auc_fast(alg[ii][mm].astype(int), p[ii][mm])
    a2 = L.auc_fast(alg[oldreal | alg].astype(int), p[oldreal | alg])
    a3 = L.auc_fast(sal[oldreal | sal].astype(int), p[oldreal | sal])
    return {'counts': cnt, 'auc_rec_vs_alg': a1, 'ci': (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))),
            'auc_iowak18_vs_alg': a2, 'auc_sal_vs_iowak18': a3,
            'det_rec': float(np.mean(p[rec] > 0.5)), 'det_alg': float(np.mean(p[alg] > 0.5)),
            'meanP': {c: float(p[cls == c].mean()) for c in CL.REC + CL.ALG}}


# ----------------------------------------------------------------------------- recording-noise proxy
def _snr_one(args):
    path, onset_s, span_s = args
    sys.path.insert(0, os.path.join(CL.ROOT, 'Stattia', 'revision', 'figures'))
    from detune.segment import load_mono
    from phaseo_lib import noise_floor_rms
    try:
        x, sr = load_mono(path)
        a = int((onset_s + 0.03) * sr)
        w = x[a:a + max(1, int(span_s * sr))]
        nf = noise_floor_rms(x, sr)
        r = float(np.sqrt(np.mean(w ** 2))) if len(w) else np.nan
        return 20 * np.log10(r / nf) if (np.isfinite(nf) and nf > 0 and r > 0) else np.nan
    except Exception:
        return np.nan


def snr_table(meta, workers=4):
    """SNR proxy per file, as in the figure analysis (FIGURE_REPORT s.6.2): RMS of the phaseogram window
    (onset + 30 ms, 48 periods of the constrained f0; the -45 dB trim offset is ignored) over the 5th-percentile
    RMS of 20-ms frames of the whole file (figures/phaseo_lib.noise_floor_rms).  Cached in cache/snr.csv."""
    p = os.path.join(CL.CACHE, 'snr.csv')
    if os.path.exists(p):
        d = pd.read_csv(p)
        if len(d) == len(meta) and (d['file'].values == meta['file'].values).all():
            return d['snr_db'].values
    args = list(zip(meta['path'], meta['con_onset_s'].astype(float), meta['con_img_span_s'].astype(float)))
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=workers) as ex:
        snr = np.array(list(ex.map(_snr_one, args, chunksize=8)), dtype=float)
    pd.DataFrame({'file': meta['file'], 'snr_db': snr}).to_csv(p, index=False)
    return snr


def snr_section(meta, scores, snr):
    from scipy.stats import spearmanr
    src, sub = meta['source'].values, meta['sub'].values
    groups = [('CCMusic notes', src == 'ccmusic'), ('SoundFont renders', src == 'soundfont'), ('Otey notes', src == 'otey')] + \
             [(f'library {lb} (all layers)', (src == 'library') & (sub == lb)) for lb in CL.LIBS]
    cc = src == 'ccmusic'
    rows = []
    for name, msk in groups:
        ok = msk & np.isfinite(snr)
        row = [name, int(ok.sum()), f'{np.median(snr[ok]):.1f}' if ok.any() else 'n/a']
        if name.startswith('library'):
            yy = np.r_[np.zeros((cc & np.isfinite(snr)).sum()), np.ones(ok.sum())]
            row.append(f'{L.auc_fast(yy, np.r_[snr[cc & np.isfinite(snr)], snr[ok]]):.2f}')
        else:
            row.append('-')
        for key in (('constr', 'cnn', 'T2'), ('constr', 'fus47', 'T2')):
            s = scores.get(key)
            if s is None:
                row.append('n/a')
                continue
            m2 = ok & np.isfinite(s)
            rho = spearmanr(snr[m2], s[m2]).correlation if m2.sum() > 5 else np.nan
            row.append(f'{rho:+.2f}')
        rows.append(row)
    return rows


# ----------------------------------------------------------------------------- matched XGB control
def xgb_matched(meta, X47):
    """XGBoost (protocol-v2 settings: train_realness hyper-parameters, n_jobs=1, random_state=0, global
    scale_pos_weight) on EXACTLY the CNN's isolated-note populations and folds.  Optional (needs xgboost)."""
    try:
        L.check_xgboost()
    except SystemExit:
        return None
    src, fold, grp = meta['source'].values, meta['fold'].values, meta['grp'].values
    y = meta['y'].values
    mf = meta['mf_layer'].values.astype(bool)
    corpus = np.isin(src, ['ccmusic', 'soundfont', 'otey'])
    i16 = [CL.FEATURE_COLS.index(c) for c in CL.MED16]
    out = {}
    for fs, X in (('all47', X47.astype(float)), ('med16', X47[:, i16].astype(float))):
        for c in CONDS:
            trainable = corpus | ((src == 'library') & mf) if c == 'T1' else corpus.copy()
            spw = float(np.sum(y[trainable] == 0)) / max(1, int(np.sum(y[trainable] == 1)))
            full = np.zeros((5, len(meta)))
            for k in range(5):
                tr = trainable & (fold != k)
                _, (pa,) = L.fit_predict('xgb', X[tr], y[tr], [X], spw=spw)
                full[k] = pa
            s = np.full(len(meta), np.nan)
            for i in range(len(meta)):
                if corpus[i]:
                    s[i] = full[fold[i], i]
                elif src[i] == 'library':
                    s[i] = full[fold[i], i] if c == 'T1' else full[:, i].mean()
            out[(fs, c)] = s
        # LOGO on the isolated T2 population for the clips
        s = np.full(len(meta), np.nan)
        spw_all = float(np.sum(y[corpus] == 0)) / max(1, int(np.sum(y[corpus] == 1)))
        preds = {}
        for name, ex in (('full', None), ('noOtey', CL.OTEY_G), ('noCasio', CL.CASIO_G)):
            tr = corpus & (grp != ex) if ex else corpus
            _, (pa,) = L.fit_predict('xgb', X[tr], y[tr], [X], spw=float(np.sum(y[tr] == 0)) / max(1, int(np.sum(y[tr] == 1))))
            preds[name] = pa
        for i in np.flatnonzero(src == 'clip'):
            sb = meta['sub'].values[i]
            s[i] = preds['noOtey' if sb == 'Otey' else 'noCasio' if sb == CL.ALG[0] else 'full'][i]
        out[(fs, 'LOGO')] = s
    return out


# ----------------------------------------------------------------------------- RESULTS_v2 XGB numbers
def xgb_v2():
    pn = pd.read_csv(os.path.join(CL.ANA_OUT, 'pernote_comparisons.csv'))
    cm = pd.read_csv(os.path.join(CL.ANA_OUT, 'comparisons.csv'))
    f0s = pd.read_csv(os.path.join(CL.ANA_OUT, 'f0_sensitivity.csv'))
    st = pd.read_csv(os.path.join(CL.ANA_OUT, 'stimuli_detector_scores.csv'))
    name_map = {'SoundFont renders': 'CCMusic notes vs SoundFont renders', 'Otey notes': 'CCMusic notes vs Otey notes'}
    for c in COMPS[3:]:
        name_map[c] = 'CCMusic notes vs ' + c
    out = {}
    for fs in ('all47', 'med16'):
        for mode in ('T1', 'T2', 'Ionly'):
            run = f'xgb_{fs}_{mode}'
            for c in COMPS:
                if c == 'algorithmic (SoundFont + Otey)':
                    r = cm[(cm.run == run) & (cm.id == 'I1')]
                else:
                    r = pn[(pn.run == run) & (pn.comparison == name_map[c])]
                if len(r):
                    out[(fs, mode, c)] = r.iloc[0].to_dict()
    f0map = {'SoundFont renders': 'SoundFont renders', 'Otey notes': 'Otey notes',
             'algorithmic (SoundFont + Otey)': 'algorithmic (both)',
             'library notes: all 4 libraries, mf layer': 'library notes, mf layer',
             'library notes: all 4 libraries, all layers': 'library notes, all layers'}
    for _, r in f0s.iterrows():
        for k, v in f0map.items():
            if r['comparison'] == v:
                out[('f0sens', r['run'], r['restriction'], k)] = r.to_dict()
    return out, st


def _v2_stamp():
    p = os.path.join(CL.ANA_OUT, 'RESULTS_v2.md')
    gen = ''
    if os.path.exists(p):
        for line in open(p, encoding='utf-8'):
            if line.startswith('Generated '):
                gen = line.split(' by ')[0].replace('Generated ', '').strip()
                break
    return (f'RESULTS_v2 version read: generated {gen or "unknown"}. Its T1 detector is trained on the whole v2 corpus, '
            'including any data_v2 rows present (e.g. the digital_v2 sampled-digital performances labelled synthetic), so '
            'its T1 numbers move when data_v2 grows; its T2 and I-only numbers do not involve those rows.')


# ----------------------------------------------------------------------------- formatting
def cell(r, det=True):
    if r is None or not np.isfinite(r.get('auc', np.nan)):
        return 'n/a'
    s = f"{r['auc']:.3f} [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]"
    return s + (f"; det {100 * r['det_syn']:.1f}%" if det else '')


def fa(r):
    return f"{100 * r['fa_real']:.1f}%" if r is not None and np.isfinite(r.get('fa_real', np.nan)) else 'n/a'


def main(log=print):
    t0 = time.time()
    meta = pd.read_csv(os.path.join(CL.CACHE, 'phaseo_meta.csv'))
    u = CL.universe()
    assert (u['file'].values == meta['file'].values).all()
    X47 = CL.features(u)
    scores, info = assemble(meta)
    log(f'assembled {len(scores)} score sets')
    v2, st_v2 = xgb_v2()
    xm = xgb_matched(meta, X47)
    log('matched XGB control: ' + ('done' if xm is not None else 'skipped (xgboost not importable)'))

    # ---------------- comparisons
    rows, R = [], {}
    in4876 = meta['in_48_76'].values.astype(bool)
    near = (meta['con_err_st'].abs() < 0.5).values
    restr = {'all': None, 'MIDI 48-76': in4876, 'abs(f0_constr - nominal) < 0.5 st': near}
    for (v, m, c), s in scores.items():
        if c in ('LOGO', 'LOGOsrc', 'T2src'):
            continue
        rlist = ['all', 'MIDI 48-76', 'abs(f0_constr - nominal) < 0.5 st'] if v == 'constr' else ['all']
        for rn_ in rlist:
            res = comparisons(meta, s, restr[rn_])
            for name, r in res.items():
                R[(v, m, c, rn_, name)] = r
                rows.append({'variant': v, 'model': m, 'cond': c, 'restriction': rn_, 'comparison': 'CCMusic notes vs ' + name, **r})
    if xm is not None:
        for (fs, c), s in xm.items():
            if c == 'LOGO':
                continue
            for rn_ in ('all', 'MIDI 48-76'):
                res = comparisons(meta, s, restr[rn_])
                for name, r in res.items():
                    R[('xgbm', fs, c, rn_, name)] = r
                    rows.append({'variant': 'xgb_matched', 'model': f'xgb_{fs}', 'cond': c, 'restriction': rn_,
                                 'comparison': 'CCMusic notes vs ' + name, **r})
    comp_df = pd.DataFrame(rows)
    comp_df.to_csv(os.path.join(HERE, 'comparisons_cnn_iso.csv'), index=False)

    # ---------------- per-file scores
    oof = meta[['file', 'source', 'sub', 'grp', 'fold', 'y', 'nominal_midi', 'in_48_76', 'layer', 'mf_layer',
                'det_f0', 'det_err_st', 'con_f0', 'con_err_st', 'con_fallback', 'det_mode', 'con_mode', 'det_n_segs',
                'con_n_segs', 'con_zero_rows']].copy()
    for (v, m, c), s in scores.items():
        oof[f'{v}_{m}_{c}'] = s
    if xm is not None:
        for (fs, c), s in xm.items():
            oof[f'xgbmatched_{fs}_{c}'] = s
    oof.to_csv(os.path.join(HERE, 'oof_scores_cnn_iso.csv'), index=False, float_format='%.6f')

    # ---------------- clips
    clip_rows, CM = [], {}
    clip_variants = []
    for m in MODELS:
        clip_variants += [(('constr', m, 'LOGO'), f'{MLAB[m]}, LOGO (primary)'),
                          (('constr', m, 'LOGOsrc'), f'{MLAB[m]}, LOGO, unprocessed source note files'),
                          (('constr', m, 'T2'), f'{MLAB[m]}, CV fold models (T2)'),
                          (('constr', m, 'T2src'), f'{MLAB[m]}, CV fold models (T2), unprocessed source note files')]
    for m in MODELS:
        clip_variants += [(('det48', m, 'T2'), f'{MLAB[m]}, MIDI 48-76 variant, CV fold models (T2)')]
    for key, lab in clip_variants:
        if key in scores:
            CM[key] = clip_metrics(meta, scores[key])
            clip_rows.append((lab, CM[key]))
    cl_mask = (meta['source'] == 'clip').values
    clip_ord = meta[cl_mask].reset_index(drop=True)
    stv2 = st_v2.set_index('clip')
    clip_names = [os.path.basename(f)[:-4] for f in clip_ord['file']]
    xgb_clip = {}
    for col, lab in (('xgb_all47_clip_LOGO', 'XGB all47, LOGO, T2 population incl. performances (RESULTS_v2 s.4)'),
                     ('xgb_med16_clip_LOGO', 'XGB med16, LOGO, T2 population incl. performances (RESULTS_v2 s.4)')):
        s = np.full(len(meta), np.nan)
        s[cl_mask] = stv2.loc[clip_names, col].values
        xgb_clip[col] = s
        clip_rows.append((lab, clip_metrics(meta, s)))
    if xm is not None:
        for fs in ('all47', 'med16'):
            clip_rows.append((f'XGB {fs}, LOGO, isolated-note population (computed here)', clip_metrics(meta, xm[(fs, 'LOGO')])))
    clip_tab = clip_ord[['file', 'sub', 'nominal_midi']].copy()
    clip_tab['clip'] = clip_names
    for key, _ in clip_variants:
        if key in scores:
            clip_tab['_'.join(key)] = scores[key][cl_mask]
    for col, s in xgb_clip.items():
        clip_tab[col + '_v2'] = s[cl_mask]
    if xm is not None:
        clip_tab['xgbmatched_all47_LOGO'] = xm[('all47', 'LOGO')][cl_mask]
    clip_tab.to_csv(os.path.join(HERE, 'stimuli_cnn_iso_scores.csv'), index=False, float_format='%.6f')

    # ================================================================= markdown
    md = []
    md.append('# Phaseogram CNN and fusion, FORMAT-MATCHED (isolated notes only)\n\n')
    md.append(f"Generated {time.strftime('%Y-%m-%d %H:%M')} by one command, from the project root:\n\n"
              "```\npython Stattia/revision/cnn_iso/run_all.py\n```\n\n"
              "(builds the phaseogram cache, trains every model that is not yet in `cnn_iso/jobs/` -- resumable -- and "
              "evaluates. The matched-XGBoost control additionally needs xgboost 3.2.0 importable, as protocol v2.)\n\n")
    md.append(f"Environment: Python {platform.python_version()}, torch {__import__('torch').__version__} (CPU only), numpy "
              f"{np.__version__}, pandas {pd.__version__}. Scripts: `Stattia/revision/cnn_iso/{{run_all, build_phaseograms, "
              f"train_cnn_iso, evaluate_cnn_iso, cnn_iso_lib}}.py`. Outputs: `RESULTS_CNN_ISO.md`, `oof_scores_cnn_iso.csv` "
              f"(per-file scores of every model/condition), `comparisons_cnn_iso.csv`, `stimuli_cnn_iso_scores.csv`, "
              f"`cache/` (phaseograms + `phaseo_meta.csv`), `jobs/` (one .npz per trained model), `logs/`.\n\n")
    import train_cnn_iso as T
    alljobs = T.job_list('all')
    done = [j for j in alljobs if os.path.exists(T.job_path(j))]
    if len(done) < len(alljobs):
        miss = sorted({f'{v}/{m}/{c}' + ('/LOGO' if s.startswith('logo') else '') for v, m, c, s in alljobs
                       if (v, m, c, s) not in set(done)})
        md.append(f'> **INTERIM ({time.strftime("%Y-%m-%d %H:%M")}): {len(done)} of {len(alljobs)} model fits finished.** '
                  f'Tables show only complete model/condition sets (all 5 folds); missing sets are shown as n/a or omitted: '
                  f'{", ".join(miss)}.\n\n')
    reading = os.path.join(HERE, 'reading.md')
    if os.path.exists(reading):
        md.append(open(reading, encoding='utf-8').read().rstrip() + '\n\n')

    # ---- section 0: key numbers
    md.append('## 0. Key numbers (constrained-f0 phaseograms, all registers)\n\n')
    md.append('Cells: AUC [95% CI, stratified grouped bootstrap]; det = share of synthetic files with P(synthetic) > 0.5. '
              'XGB columns are copied from `analysis/out/` (RESULTS_v2 s.2-3): T1/T2 = corpus-level training (performances '
              'included), I-only = trained on isolated notes only (CCMusic vs SoundFont + Otey, same folds; library notes by '
              'the mean of the 5 fold models -- the XGBoost counterpart of this study\'s T2; shown from the matched '
              'control of s.3.4, which reproduces the RESULTS_v2 I-only AUCs exactly and adds the "algorithmic" row). '
              + _v2_stamp() + '\n\n')
    for c in CONDS:
        hdr = ['CCMusic notes vs'] + [f'{MLAB[m]} {c}' for m in MODELS]
        if c == 'T2':
            hdr += ['XGB all47 I-only', 'XGB all47 T2', 'XGB med16 I-only']
        else:
            hdr += ['XGB all47 T1', 'XGB med16 T1']
        tb = []
        for name in KEY_COMPS:
            row = [name] + [cell(R.get(('constr', m, c, 'all', name))) for m in MODELS]
            if c == 'T2':
                # I-only = the matched control's T2 (identical training rows, folds and scoring; equal to the
                # RESULTS_v2 I-only numbers); RESULTS_v2 has no I-only row for 'algorithmic (both)'
                row += [cell(R.get(('xgbm', 'all47', 'T2', 'all', name)) or v2.get(('all47', 'Ionly', name))),
                        cell(v2.get(('all47', 'T2', name))),
                        cell(R.get(('xgbm', 'med16', 'T2', 'all', name)) or v2.get(('med16', 'Ionly', name)))]
            else:
                row += [cell(v2.get(('all47', 'T1', name))), cell(v2.get(('med16', 'T1', name)))]
            tb.append(row)
        frow = ['(CCMusic false alarms, P > 0.5)'] + [fa(R.get(('constr', m, c, 'all', 'SoundFont renders'))) for m in MODELS]
        if c == 'T2':
            frow += [fa(v2.get(('all47', 'Ionly', 'SoundFont renders'))), fa(v2.get(('all47', 'T2', 'SoundFont renders'))),
                     fa(v2.get(('med16', 'Ionly', 'SoundFont renders')))]
        else:
            frow += [fa(v2.get(('all47', 'T1', 'SoundFont renders'))), fa(v2.get(('med16', 'T1', 'SoundFont renders')))]
        tb.append(frow)
        md.append(f'**{c}** ' + ('(libraries excluded from training; library notes = mean of the 5 fold models)'
                                 if c == 'T2' else '(library mf-layer notes in training, labelled synthetic; each '
                                 'library note scored by the fold model whose test fold holds its library)') + '\n\n')
        md.append(L.md_table(hdr, tb))
        md.append('\n')
    if clip_rows:
        md.append('Listening clips (out of sample): ' + '; '.join(
            f"{lab}: recorded det {int(round(100 * r['det_rec']))}% / algorithmic det {int(round(100 * r['det_alg']))}%, "
            f"AUC {r['auc_rec_vs_alg']:.3f}" for lab, r in clip_rows
            if ('LOGO (primary)' in lab or ('CV fold models (T2)' in lab and 'MIDI' not in lab)
                or ('RESULTS_v2' in lab and 'all47' in lab)))
            + '.\n\n')

    # ---- section 1: data
    md.append('## 1. Data, phaseograms, models, protocol\n\n')
    comp = []
    for sname, msk in (('CCMusic notes (real)', meta.source == 'ccmusic'), ('SoundFont renders', meta.source == 'soundfont'),
                       ('Otey notes', meta.source == 'otey')) + tuple(
            (f'library {lb} (all layers / mf layer)', (meta.source == 'library') & (meta['sub'] == lb)) for lb in CL.LIBS) + \
            (('listening clips', meta.source == 'clip'),):
        g = meta[msk]
        nf = f"{len(g)} / {int(g.mf_layer.sum())}" if sname.startswith('library') else str(len(g))
        comp.append([sname, nf, g.grp.nunique() if sname != 'listening clips' else 5,
                     ', '.join(str(x) for x in sorted(g.fold.unique())) if sname != 'listening clips' else '-',
                     int(g.in_48_76.sum())])
    md.append(L.md_table(['source', 'files', 'CV groups', 'fold(s)', 'files with nominal MIDI 48-76'], comp))
    md.append(
        "\n- **One file = one struck note, one phaseogram per file**, in every class: the primary segment (largest energy "
        "among segments with 24 < f0 < 4500 Hz) of `detune.segment.build_segments`, the rule the detector uses for the "
        "sample-library notes (`quality/extract_samplelib_features.py`). Construction otherwise as the detector "
        "(`quality/phasefeat.compute_phaseogram`, P = 256, C = 48; spectral-flux onset, -45 dB trim, 30-ms skip, "
        "inharmonicity-corrected f0 from `refine_f0_multipartial`). Check: for the three files the figure analysis used "
        "(S90ES C4 v64, Otey C4 v75, Otey G3 v127) the unconstrained phaseograms equal the detector's own cached arrays in "
        "`quality/cache/phase/` (max |difference| = 0).\n"
        "- **Constrained f0 (variant `constr`, primary).** `detune.segment.estimate_pitches` is replaced, in the worker "
        "process only, by the same spectrum and harmonic-sum salience evaluated only at MIDI nominal-1, nominal, nominal+1, "
        "then the same parabolic peak refinement limited to [f(nominal-1), f(nominal+1)]; the sub-octave guard is dropped "
        "(it can only leave the window). The detector's B estimate and `refine_f0_multipartial` then run unchanged; if the "
        "refined f0 leaves the window the window-limited peak is kept ('fallback', counted below). Identical for all classes. "
        "Because the pitch also decides how the segmenter merges repeated onsets of the same note, constraining it can merge "
        "spurious re-onsets into one longer ring; in files where the detector's own f0 was already right this changes the "
        "image in 11% of files (f0 shift median < 1 cent).\n"
        "- **Unconstrained detector f0 (variant `det48`).** The detector unchanged; all classes restricted to nominal MIDI "
        "48-76, where its f0 is mostly right; trained and evaluated on that subset only.\n"
        "- **Nominal pitch.** CCMusic: the HuggingFace dataset's `pitch` class label (`ccmusic-database/pianos`, config "
        "`8_class`, 88 Helmholtz names, e.g. `c1` = C4, `C` = C2, `A2` = A0), which the WAV export wrote into each file name "
        "(`<Piano>_<pitch>_<split>_<index>.wav`); parsed with `pv2_lib.helmholtz_to_midi` (sharps take the first spelling, "
        "`c1s-d1b` = C#4). The convention is the one the figure analysis established; its own f0 agrees with it in 211 of "
        "220 mid-register files. SoundFont and Otey renders: note name in the file name (`C4_v127.wav`); libraries: note "
        "name in the file name (Iowa `Piano.mf.C4.aiff`, Salamander `C4v8.ogg`, K18 `C4.mf.wav`, FreePats `C4.wav`); "
        "clips: `listening_kit/answer_key.csv` column `note`. All via `pv2_lib.nominal_midi` / `sci_to_midi`.\n"
        "- **Models** = `quality.realness_net._train_fold` / `_predict` unchanged (PhaseCNN 16-32-64-128 + BN, emb 128; "
        "head 128-ReLU-Dropout 0.4-1; harmonic EQ, phase roll, per-cycle gain jitter, noise 0.02, row masking; "
        "BCE(pos_weight = N_real/N_synth of the training fold) + 1.0 x MSE consistency of two views; Adam lr 1e-3, "
        "weight decay 1e-4; batch 24 files; 40 epochs, final weights; torch Generator seed 0). (a) CNN only; (b) fusion "
        "with the 47 features (median-imputed, standardised on the training fold); (c) fusion with the 16 Group-A "
        "medians (the head's input size comes from `len(realness_net.FEATURE_COLS)`, which is set to the 16 names during "
        "that call). Each fit starts from `torch.manual_seed(0)`, `np.random.seed(0)`.\n"
        "- **File pooling.** Kept the code's [mean; std] pooling; every file has one note, so the code's own single-note "
        "branch makes std = 0 for every file of every class (the head sees [embedding; 0]). Nothing class-dependent.\n"
        "- **47 features** for fusion = the stored rows: `quality/cache/features_cal.csv` (CCMusic), `synth_features.csv` "
        "(SoundFont, Otey); library notes and clips from `analysis/out/cache/{library_notes, stimuli_clips}.csv` (same "
        "single-file pipeline). They use the detector's unconstrained f0.\n"
        "- **Folds** = `analysis/out/folds.csv` (CCMusic, SoundFont, Otey and the four library groups). **T2**: train = "
        "CCMusic + SoundFont + Otey notes outside the test fold; library notes = mean of the 5 fold models. **T1**: as T2 "
        "plus the libraries' mf-equivalent-layer notes outside the test fold (Salamander v8, Iowa mf, K18 mf, FreePats), "
        "labelled synthetic; each library note (mf or any layer) scored by the fold model whose test fold holds its "
        "library (FreePats, Iowa: fold 1; K18, Salamander: fold 2). Fold 2 holds no CCMusic group, so its real notes "
        "never enter the comparisons; the same holds for the XGB numbers.\n\n")

    # ---- section 2: f0 diagnostics
    md.append('## 2. f0 and phaseogram diagnostics per source\n\n')
    dg = []
    meta['_src'] = np.where(meta.source == 'library', 'library ' + meta['sub'].astype(str), meta.source)
    Xc = np.load(os.path.join(CL.CACHE, 'phaseo_con.npy'), mmap_mode='r')
    Xd = np.load(os.path.join(CL.CACHE, 'phaseo_det.npy'), mmap_mode='r')
    same = np.array([np.array_equal(Xc[i], Xd[i]) for i in range(len(meta))])
    for sname, g in meta.groupby('_src', sort=False):
        ii = g.index.values
        dg.append([sname, len(g), int((g.det_err_st.abs() >= 1).sum()),
                   f"{int((g[g.in_48_76].det_err_st.abs() >= 1).sum())}/{int(g.in_48_76.sum())}",
                   int((g.con_fallback > 0).sum()), int((g.con_err_st.abs() >= 0.5).sum()),
                   f'{same[ii].mean():.2f}', f"{(g.det_mode == 'piece').mean():.2f}",
                   f'{g.det_n_segs.median():.0f} / {g.con_n_segs.median():.0f}', f'{g.con_seg_len_s.median():.2f}',
                   f'{g.con_zero_rows.mean():.1f}'])
    md.append(L.md_table(['source', 'files', 'detector f0 off >= 1 st (primary)', '... in MIDI 48-76',
                          'constrained: fallback', 'constrained: abs(f0 - nominal) >= 0.5 st', 'image identical det = constr',
                          'piece mode (several onsets)', 'segments det / constr (median)', 'primary ring length (s, median)',
                          'all-zero rows of 48 (mean, constr)'], dg))
    cc = (meta.source == 'ccmusic').values
    zr = []
    for nm, msk in (('SoundFont', meta.source == 'soundfont'), ('Otey', meta.source == 'otey'),
                    ('library notes (all layers)', meta.source == 'library')):
        msk = msk.values
        yy = np.r_[np.zeros(cc.sum()), np.ones(msk.sum())]
        zr.append(f"{nm} {L.auc_fast(yy, np.r_[meta.loc[cc, 'con_zero_rows'].values, meta.loc[msk, 'con_zero_rows'].values]):.2f}")
    md.append('\nAll-zero rows = cycles past the end of the primary ring (zero padding of `compute_phaseogram`): in piece '
              'mode a spurious later onset ends the ring early. Used alone as a score, the number of zero rows separates '
              'CCMusic notes from the other sources at AUC ' + ', '.join(zr) + ' (0.5 = no information), so the padding is '
              'not a usable format cue here. The detector-f0 failure counts reproduce the f0 audit of RESULTS_v2 s.3.6 '
              'exactly (CCMusic 329/668, SoundFont 135, Otey 18, Salamander 228, Iowa 136, K18 88, FreePats 35).\n\n')

    snr = snr_table(meta)
    md.append('**Level-ratio proxy ("SNR").** RMS of the phaseogram window (onset + 30 ms, 48 periods) over the '
              '5th-percentile RMS of 20-ms frames of the whole file (the figure analysis\' SNR definition, a lower bound; '
              'for a long decaying note it also measures how far the file decays, e.g. into digital silence). '
              '"Alone" = AUC of the ratio as a score, CCMusic notes vs that library (higher = more synthetic). '
              'rho = Spearman correlation between the ratio and the model\'s P(synthetic) WITHIN the source (T2 scores).\n\n')
    md.append(L.md_table(['source', 'files', 'median ratio (dB)', 'ratio alone: AUC vs CCMusic', 'rho(ratio, P) CNN T2',
                          'rho(ratio, P) fusion-47 T2'], snr_section(meta, scores, snr)))
    md.append('\n')

    import build_phaseograms as BP
    r0 = BP.row0_table(meta)['row0_ms'].values
    src_, sub_ = meta['source'].values, meta['sub'].values
    cc_ = src_ == 'ccmusic'
    s_cnn = scores.get(('constr', 'cnn', 'T2'))
    rt = []
    for nm, msk in ([('CCMusic notes', cc_), ('SoundFont renders', src_ == 'soundfont'), ('Otey notes', src_ == 'otey')]
                    + [(f'library {lb} (all layers)', (src_ == 'library') & (sub_ == lb)) for lb in CL.LIBS]
                    + [('listening clips', src_ == 'clip')]):
        q = np.nanpercentile(r0[msk], [10, 50, 90])
        a = '-' if nm in ('CCMusic notes', 'listening clips') else \
            f"{L.auc_fast(np.r_[np.zeros(cc_.sum()), np.ones(msk.sum())], np.r_[r0[cc_], r0[msk]]):.2f}"
        dd = '-'
        if s_cnn is not None and nm != 'listening clips':
            dd = ' / '.join(f"{100 * np.mean(s_cnn[msk & (r0 >= lo) & (r0 < hi)] > 0.5):.0f}% (n={int((msk & (r0 >= lo) & (r0 < hi)).sum())})"
                            if (msk & (r0 >= lo) & (r0 < hi)).any() else 'n/a'
                            for lo, hi in ((-1e9, 30), (30, 60), (60, 1e9)))
        rt.append([nm, int(msk.sum()), f'{q[1]:.0f} ({q[0]:.0f}-{q[2]:.0f})', a, dd])
    md.append('**Where the 48-cycle window starts.** row 0 minus the attack (first sample with |x| > 0.02 x peak, the '
              'listening kit\'s rule), in ms; the detector puts row 0 30 ms after its -45 dB trim point, so pre-attack '
              'noise above -45 dB moves the window towards the attack. "Alone" = AUC of this offset as a score vs '
              'CCMusic. Last column: share with P(synthetic) > 0.5 (CNN only, T2) for windows starting < 30 / 30-60 / '
              '>= 60 ms after the attack (for CCMusic = false alarms).\n\n')
    md.append(L.md_table(['source', 'files', 'row 0 - attack, ms: median (10-90%)', 'alone: AUC vs CCMusic',
                          'CNN T2 P > 0.5 by window start'], rt))
    md.append('\n')

    # ---- section 3: full tables
    md.append('## 3. Full results, constrained-f0 phaseograms (all registers)\n\n')
    for k_, m in enumerate(MODELS):
        fsx = 'med16' if m == 'fus16' else 'all47'
        md.append(f'### 3.{k_ + 1} {MLAB[m]}\n\n')
        tb = []
        for name in COMPS:
            r1, r2 = R.get(('constr', m, 'T1', 'all', name)), R.get(('constr', m, 'T2', 'all', name))
            ref = r2 or r1
            if ref is None:
                continue
            row = [name, f"{ref['n_real']}/{ref['g_real']} vs {ref['n_syn']}/{ref['g_syn']}"]
            for r in (r1, r2):
                row += [cell(r, det=False), L.fpct(r['det_syn']) if r else 'n/a', L.f3(r['meanP_syn']) if r else 'n/a']
            for mode in ('T1', 'T2', 'Ionly'):
                row.append(cell(v2.get((fsx, mode, name))))
            tb.append(row)
        md.append(L.md_table(['CCMusic notes vs', 'files/groups', 'T1 AUC [CI]', 'T1 det.', 'T1 mean P', 'T2 AUC [CI]',
                              'T2 det.', 'T2 mean P', f'XGB {fsx} T1 (v2)', f'XGB {fsx} T2 (v2)', f'XGB {fsx} I-only (v2)'], tb))
        r1, r2 = R.get(('constr', m, 'T1', 'all', 'SoundFont renders')), R.get(('constr', m, 'T2', 'all', 'SoundFont renders'))
        md.append(f"\nCCMusic false alarms (P > 0.5) / mean P: T1 {fa(r1)} / {L.f3(r1['meanP_real']) if r1 else 'n/a'}; "
                  f"T2 {fa(r2)} / {L.f3(r2['meanP_real']) if r2 else 'n/a'}.\n\n")
    if xm is not None:
        md.append('### 3.4 Control: XGBoost on exactly these isolated-note populations and folds (computed here)\n\n')
        md.append('Same rows, folds, T1/T2 definitions and scoring rules as the CNN; protocol-v2 XGBoost settings. T2 here '
                  'is the I-only detector of RESULTS_v2 (identical training set and scoring).\n\n')
        tb = []
        for name in COMPS:
            row = [name]
            for fs in ('all47', 'med16'):
                for c in CONDS:
                    r = R.get(('xgbm', fs, c, 'all', name))
                    row.append(cell(r))
            tb.append(row)
        tb.append(['(CCMusic false alarms)'] + [fa(R.get(('xgbm', fs, c, 'all', 'SoundFont renders')))
                                                for fs in ('all47', 'med16') for c in CONDS])
        md.append(L.md_table(['CCMusic notes vs', 'XGB all47 T1', 'XGB all47 T2', 'XGB med16 T1', 'XGB med16 T2'], tb))
        md.append('\n')

    # ---- stability across individual models of the T2 population
    md.append('### 3.5 Library notes under each individual model trained on the T2 population (constrained f0)\n\n')
    md.append('All library notes are out of sample for all 8 models (5 CV fold models + 3 LOGO models). det = share of '
              'mf-layer library notes with P > 0.5, per library and pooled; AUC = CCMusic notes of the model\'s own test '
              'fold vs all mf-layer library notes (fold models only; fold 2 holds no CCMusic).\n\n')
    srcv, subv, foldv, mfv = meta['source'].values, meta['sub'].values, meta['fold'].values, meta['mf_layer'].values.astype(bool)
    st_rows = []
    for m in MODELS:
        for s_ in [f'f{k}' for k in range(5)] + ['logo_full', 'logo_noOtey', 'logo_noCasio']:
            j = _job('constr', m, 'T2', s_)
            if j is None:
                continue
            Pm = np.full(len(meta), np.nan)
            Pm[j['rows']] = j['p']
            lm = (srcv == 'library') & mfv
            row = [MLAB[m], s_, f'{100 * np.mean(Pm[lm] > 0.5):.0f}%'] + \
                  [f'{100 * np.mean(Pm[lm & (subv == lb)] > 0.5):.0f}%' for lb in CL.LIBS]
            if s_.startswith('f') and ((srcv == 'ccmusic') & (foldv == int(s_[1:]))).any():
                ccm = (srcv == 'ccmusic') & (foldv == int(s_[1:]))
                row += [f"{L.auc_fast(np.r_[np.zeros(ccm.sum()), np.ones(lm.sum())], np.r_[Pm[ccm], Pm[lm]]):.3f}",
                        f'{100 * np.mean(Pm[ccm] > 0.5):.0f}% (n={int(ccm.sum())})']
            else:
                row += ['-', '-']
            st_rows.append(row)
    md.append(L.md_table(['model', 'fit', 'library mf det.'] + [f'{lb} det.' for lb in CL.LIBS] +
                         ['AUC own-fold CCMusic vs library mf', 'own-fold CCMusic false alarms'], st_rows))
    md.append('\n')

    # ---- section 4: register / f0 sensitivity
    md.append('## 4. Register and f0 sensitivity\n\n')
    md.append('Rows: `constr` = constrained-f0 models (all registers) evaluated on all files, on nominal MIDI 48-76 only '
              '(both classes), or on files whose constrained f0 lies within 0.5 semitone of the nominal pitch; `det48` = '
              'models trained and evaluated on MIDI 48-76 with the detector\'s unconstrained f0. XGB rows: RESULTS_v2 '
              's.3.6 (corpus-level training, evaluation restricted to MIDI 48-76) and the matched control restricted to '
              '48-76.\n\n')
    for c in CONDS:
        tb = []
        for m in MODELS:
            for lab, key in ((f'{MLAB[m]}, constr, all', ('constr', m, c, 'all')),
                             (f'{MLAB[m]}, constr, MIDI 48-76', ('constr', m, c, 'MIDI 48-76')),
                             (f'{MLAB[m]}, constr, abs(f0 - nominal) < 0.5 st', ('constr', m, c, 'abs(f0_constr - nominal) < 0.5 st')),
                             (f'{MLAB[m]}, det48 (unconstrained f0, MIDI 48-76)', ('det48', m, c, 'all'))):
                rr = [R.get(key + (n,)) for n in KEY_COMPS]
                if all(r is None for r in rr):
                    continue
                n0 = next(r for r in rr if r is not None)
                tb.append([lab, str(n0['n_real'])] + [cell(r) if r else 'n/a' for r in rr] + [fa(n0)])
        for fs in ('all47', 'med16'):
            for rs in ('MIDI 48-76',):
                rr = [v2.get(('f0sens', f'xgb_{fs}_{c}', rs, n)) for n in KEY_COMPS]
                if any(r is not None for r in rr):
                    n0 = next(r for r in rr if r is not None)
                    tb.append([f'XGB {fs} {c} (RESULTS_v2 s.3.6), {rs}', str(n0['n_real'])] +
                              [cell(r) if r else 'n/a' for r in rr] + [fa(n0)])
            if xm is not None:
                rr = [R.get(('xgbm', fs, c, 'MIDI 48-76', n)) for n in KEY_COMPS]
                n0 = rr[0]
                tb.append([f'XGB {fs} {c}, matched control, MIDI 48-76', str(n0['n_real'])] + [cell(r) for r in rr] + [fa(n0)])
        md.append(f'**{c}**\n\n')
        md.append(L.md_table(['model / restriction', 'CCMusic files'] + [n for n in KEY_COMPS] + ['CCMusic false alarms'], tb))
        md.append('\n')

    # ---- section 5: clips
    md.append('## 5. Listening-test clips scored out of sample\n\n')
    md.append('The 30 clips the listeners heard (`listening_kit/web/clip_XX.wav`), phaseogram with the constrained f0 '
              '(nominal note from `answer_key.csv`), 47 features from `analysis/out/cache/stimuli_clips.csv`. LOGO (primary, '
              'as RESULTS_v2 s.4): models trained on the whole T2 isolated-note population (CCMusic + SoundFont + Otey) '
              'minus the clip\'s own group -- Otey clips by a model without Otey, Casio CA-110 clips by a model without '
              'that SoundFont, library clips (Iowa, K18, Salamander) by the full model. CV-fold rows: the fold models '
              '(Otey clips by the fold-2 model, Casio clips by the fold-0 model, library clips by the mean of the 5). '
              'Detection = P > 0.5; AUC CI = clips resampled within each of the 5 classes (B = 2000, seed 0). '
              '"Unprocessed source note files" = the same models applied to the 30 original note files the clips were '
              'cut from (`answer_key.csv` column `src`; out of sample under the same rule). This isolates the effect of '
              'the clip processing (resampling to 44.1 kHz, onset trim with 10-ms pre-roll, 3-s cut, 10-ms fades, RMS '
              'normalisation, 16-bit; facts_C s.9.3).\n\n')
    tb = []
    for lab, r in clip_rows:
        tb.append([lab] + [r['counts'][c] for c in CL.REC + CL.ALG] +
                  [f"{r['auc_rec_vs_alg']:.3f} [{r['ci'][0]:.3f}, {r['ci'][1]:.3f}]", f"{r['auc_iowak18_vs_alg']:.3f}",
                   f"{r['auc_sal_vs_iowak18']:.3f}"])
    md.append(L.md_table(['scoring', 'Iowa det.', 'K18 det.', 'Salamander det.', 'SoundFont det.', 'Otey det.',
                          'AUC recorded(20) vs algorithmic(10) [95% CI]', 'AUC Iowa+K18 vs algorithmic',
                          'AUC Salamander vs Iowa+K18'], tb))
    # clip vs its source note: where the window falls, and how similar the two phaseograms are
    key_ = pd.read_csv(os.path.join(CL.KIT, 'answer_key.csv'))
    row_of = {f: i for i, f in enumerate(meta['file'])}
    Xc_ = np.load(os.path.join(CL.CACHE, 'phaseo_con.npy'), mmap_mode='r')
    al = []
    for c_, f_ in zip(key_['clip'], key_['src']):
        i, j = row_of['Stattia/listening_kit/web/' + c_ + '.wav'], row_of[f_]
        a_, b_ = np.asarray(Xc_[i], dtype=np.float64), np.asarray(Xc_[j], dtype=np.float64)
        a_, b_ = a_ - a_.mean(), b_ - b_.mean()
        # best circular shift along the within-cycle axis (one shift for all rows; the CNN is trained with a
        # random phase roll), Pearson correlation at that shift
        xc = np.fft.irfft(np.sum(np.conj(np.fft.rfft(a_, axis=1)) * np.fft.rfft(b_, axis=1), axis=0), n=a_.shape[1])
        al.append({'cls': meta['sub'].values[i], 'delta_ms': r0[i] - r0[j],
                   'corr': float(xc.max() / (np.linalg.norm(a_) * np.linalg.norm(b_) + 1e-30))})
    al = pd.DataFrame(al)
    md.append('\n**Clip vs its source note.** Shift of the phaseogram window (row 0 relative to the attack, clip minus source, '
              'ms) and correlation of the two constrained-f0 phaseograms at the best common circular phase shift, '
              'median per class: ' + '; '.join(
                  f"{c_} {g['delta_ms'].median():+.0f} ms, r = {g['corr'].median():.2f}" for c_, g in al.groupby('cls'))
              + '. A large shift and a low correlation mean that the CNN sees a different stretch of the note in the '
                'clip than in the source file. In the Iowa source files the noise floor lies within 45 dB of the peak, so '
                'the -45 dB trim starts before the attack and the 30-ms skip ends near it; the clip\'s onset trim and '
                '10-ms fade-in remove that noise and push the window about 40 ms later.\n')
    md.append('\nPer-clip P(synthetic):\n\n')
    pcols = [c for c in ('constr_cnn_LOGO', 'constr_cnn_LOGOsrc', 'constr_fus47_LOGO', 'constr_fus47_LOGOsrc',
                         'constr_fus16_LOGO', 'constr_cnn_T2', 'constr_cnn_T2src', 'xgb_all47_clip_LOGO_v2',
                         'xgbmatched_all47_LOGO') if c in clip_tab]
    md.append(L.md_table(['clip', 'class', 'MIDI'] + pcols,
                         [[r['clip'], r['sub'], int(r['nominal_midi'])] + [f'{r[c]:.3f}' for c in pcols]
                          for _, r in clip_tab.iterrows()]))

    # ---- section 6: runtime
    md.append('\n## 6. Runtime\n\n')
    rt = []
    for (v, m, c), d in sorted(info.items()):
        rt.append([v, m, c, f"{d['seconds'] / 60:.0f}", ', '.join(str(n) for n in d['n_train'])])
    md.append(L.md_table(['variant', 'model', 'condition', 'CPU-process minutes (sum over fits)', 'training files per fit'], rt))
    tot = sum(d['seconds'] for d in info.values())
    logf = os.path.join(CL.LOGS, 'train.log')
    wall = ''
    if os.path.exists(logf):
        lines = [l for l in open(logf, encoding='utf-8').read().splitlines() if l.strip()]
        ts = [pd.Timestamp(l[:19]) for l in lines if l[:4].isdigit()]
        if ts:
            wall = f' Training log `logs/train.log`: first entry {ts[0]}, last entry {ts[-1]} ({(ts[-1] - ts[0]).total_seconds() / 3600:.2f} h wall clock).'
    md.append(f'\nSum over all fits: {tot / 3600:.1f} h of per-process time (6 worker processes x 2 torch threads on an '
              f'8-core i9-9900K, CPU only).{wall} Evaluation: {time.time() - t0:.0f} s.\n')
    open(os.path.join(HERE, 'RESULTS_CNN_ISO.md'), 'w', encoding='utf-8').write(''.join(md))
    key = {f'{"|".join(k)}': {kk: (vv if not isinstance(vv, (np.floating,)) else float(vv)) for kk, vv in r.items()}
           for k, r in R.items()}
    json.dump({'comparisons': key, 'clips': {lab: r for lab, r in clip_rows}}, open(os.path.join(HERE, 'key_numbers_cnn_iso.json'), 'w'),
              indent=1, default=float)
    log(f'wrote RESULTS_CNN_ISO.md ({time.time() - t0:.0f} s)')


if __name__ == '__main__':
    main()
