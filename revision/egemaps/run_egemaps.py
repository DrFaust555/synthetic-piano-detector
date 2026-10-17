"""eGeMAPS baseline redone under protocol v2, format-matched (one isolated note = one unit).

ONE command (from the project root), writes Stattia/revision/egemaps/RESULTS_EGEMAPS.md:

    python Stattia/revision/egemaps/run_egemaps.py

Needs
  * xgboost 3.2.0 importable (system site-packages, PYTHONPATH, or EGEMAPS_XGB_PATH=<folder containing it>);
  * opensmile 2.6.0 installed into a separate folder (EGEMAPS_OPENSMILE_PATH=<folder>), e.g.
        python -m pip install --target <folder> opensmile==2.6.0
    The folder is used only by the extraction subprocess (egemaps_extract.py); the system numpy is
    imported before it, so the system Python is not changed and the models run on the same
    numpy / pandas / scikit-learn / xgboost as protocol v2.
  If the two variables are not set, the session scratch folders used for the first run are tried.

Reads (never writes) Stattia/revision/analysis/out/{folds.csv, pernote_library_scores.csv,
pernote_comparisons.csv, comparisons.csv, oof_scores_isolated_only.csv, stimuli_detector_scores.csv},
Stattia/listening_kit/answer_key.csv and the audio.  Writes only into Stattia/revision/egemaps/.

Design (protocol v2, section 3 file lists):
  real        = 668 CCMusic note files                                    (folds.csv, real_isolated)
  algorithmic = 762 SoundFont single-note renders + 120 Otey notes        (folds.csv, algo_single)
  libraries   = 999 library note files (224 in the mf-equivalent layer)   (pernote_library_scores.csv)
  folds       = the FIXED folds of analysis/out/folds.csv (library notes take their library's fold)
  T1 = library notes in training, labelled synthetic; every file scored by the fold model whose
       test fold holds it;  T2 = libraries excluded from training, scored by the mean of the 5 fold models.
  Models = pv2_lib.make_model (XGBoost n_jobs=1 random_state=0 / LR median-impute + standardise),
           scale_pos_weight = global real/synthetic ratio of the training population (as protocol v2).
  CI = pv2_lib.grouped_boot_ci (stratified grouped bootstrap, B=2000, seed 0).
Sensitivities: T1mf (only mf-layer library notes in training); sr44k (library notes resampled to 44.1 kHz);
clip format (every unit through quality/build_listening_kit.process: 44.1 kHz mono, onset-trimmed, 3 s,
RMS-normalised).  Listening clips: leave-own-group-out models on the T2 population (protocol v2 s.4).
"""
from __future__ import annotations

import os
import sys
import json
import time
import hashlib
import platform
import subprocess
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
ANA = os.path.join(ROOT, 'Stattia', 'revision', 'analysis')
AOUT = os.path.join(ANA, 'out')
KIT = os.path.join(ROOT, 'Stattia', 'listening_kit')
OUT = os.path.join(HERE, 'out')
CACHE = os.path.join(OUT, 'cache')
SCRATCH = os.environ.get('EGEMAPS_PYLIB_ROOT') or os.path.join(HERE, 'cache')
WORKERS = 8
LIBS = ['Salamander', 'Iowa', 'K18', 'FreePats']          # order of protocol_v2.LIB_SPECS
META = ['file', 'sig', 'loader', 'variant', 'sr', 'channels', 'duration_s', 'peak', 'n_wrap', 'error']


def _setup_paths():
    try:
        import xgboost  # noqa: F401
    except ImportError:
        d = os.environ.get('EGEMAPS_XGB_PATH') or os.path.join(SCRATCH, 'pylib')
        if os.path.isdir(d):
            sys.path.append(d)
    osp = os.environ.get('EGEMAPS_OPENSMILE_PATH') or os.path.join(SCRATCH, 'pylib_os')
    if not os.path.isdir(os.path.join(osp, 'opensmile')):
        raise SystemExit(f'opensmile not found in {osp}; set EGEMAPS_OPENSMILE_PATH (see the docstring).')
    for p in (ANA, ROOT):
        if p not in sys.path:
            sys.path.insert(0, p)
    return osp


OSP = _setup_paths()
import pv2_lib as L          # noqa: E402  (protocol v2 helpers: models, AUC, bootstrap, tables)


# ============================================================================ file lists
def build_lists():
    folds = pd.read_csv(os.path.join(AOUT, 'folds.csv'))
    base = folds[folds.kind.isin(['real_isolated', 'algo_single'])].reset_index(drop=True).copy()
    base['path'] = [os.path.join(ROOT, 'audio', f) if f.startswith('ccmusic_pianos/') else os.path.join(ROOT, f)
                    for f in base['file']]
    base['loader'] = 'file'                         # opensmile_baseline.py l.59: smile.process_file
    n = base.subkind.value_counts().to_dict()
    assert n == {'soundfont_single': 762, 'ccmusic': 668, 'otey_single': 120}, n

    libs = pd.read_csv(os.path.join(AOUT, 'pernote_library_scores.csv'))[['library', 'path', 'layer', 'mf_layer', 'grp']]
    libs = libs.rename(columns={'path': 'file'})
    libs['mf_layer'] = libs['mf_layer'].astype(str).str.lower().eq('true')
    libs['path'] = [os.path.join(ROOT, f) for f in libs['file']]
    libs['loader'] = 'mono'                         # opensmile_baseline.py l.95/101: load_mono + process_signal
    gf = folds.drop_duplicates('grp').set_index('grp')['fold']
    libs['fold'] = libs['grp'].map(gf).astype(int)
    libs['y'] = 1
    libs['kind'], libs['subkind'] = 'library_note', 'library:' + libs['library']
    # cross-check against protocol_v2's own library list (all files valid there)
    import protocol_v2
    ref = protocol_v2.library_note_files()
    a = sorted(os.path.normcase(os.path.normpath(p)) for p in ref['path'])
    b = sorted(os.path.normcase(os.path.normpath(p)) for p in libs['path'])
    assert a == b, 'library note list differs from protocol_v2.library_note_files()'
    assert len(libs) == 999 and int(libs.mf_layer.sum()) == 224, (len(libs), libs.mf_layer.sum())

    key = pd.read_csv(os.path.join(KIT, 'answer_key.csv'))
    clips = key[['clip', 'source', 'note', 'src']].copy()
    clips['path'] = [os.path.join(KIT, 'web', c + '.wav') for c in clips['clip']]
    clips['loader'] = 'file'
    srcs = clips[['clip', 'src']].copy()
    srcs['path'] = [os.path.join(ROOT, s) for s in srcs['src']]
    srcs['loader'] = ['mono' if s.startswith('audio/ref_') else 'file' for s in srcs['src']]
    return base, libs, clips, srcs, folds


# ============================================================================ extraction (subprocess)
def _sig(path):
    st = os.stat(path)
    return f'{st.st_size}:{int(st.st_mtime)}'


def extract(paths, loaders, variant):
    os.makedirs(CACHE, exist_ok=True)
    jobs = pd.DataFrame({'path': list(paths), 'loader': list(loaders)}).drop_duplicates().reset_index(drop=True)
    jp = os.path.join(CACHE, f'_jobs_{variant}.csv')
    jobs.to_csv(jp, index=False)
    env = dict(os.environ)
    env['EGEMAPS_OPENSMILE_PATH'] = OSP
    subprocess.run([sys.executable, os.path.join(HERE, 'egemaps_extract.py'), '--jobs', jp, '--variant', variant,
                    '--workers', str(WORKERS)], env=env, check=True)
    cache = pd.read_csv(os.path.join(CACHE, f'egemaps_{variant}.csv'), float_precision='round_trip')
    cache['sig'] = cache['sig'].astype(str)
    cache = cache.set_index(['file', 'sig', 'loader'])
    rel = [os.path.relpath(p, ROOT).replace('\\', '/') for p in paths]
    idx = list(zip(rel, [_sig(p) for p in paths], loaders))
    df = cache.loc[idx].reset_index()
    feats = [c for c in df.columns if c not in META]
    df[feats] = df[feats].apply(pd.to_numeric, errors='coerce')
    return df, feats


def _lead_one(path):
    from detune.segment import load_mono
    x, sr = load_mono(path)
    pk = np.max(np.abs(x)) + 1e-9
    return float(np.argmax(np.abs(x) > 0.02 * pk) / sr)        # = build_listening_kit.process onset rule


def lead_in(paths):
    """Time (s) from file start to the first sample above 2% of the peak (cached)."""
    cp = os.path.join(CACHE, 'lead_in.csv')
    rel = [os.path.relpath(p, ROOT).replace('\\', '/') for p in paths]
    sig = [_sig(p) for p in paths]
    c = pd.read_csv(cp) if os.path.exists(cp) else pd.DataFrame(columns=['file', 'sig', 'lead_s'])
    have = {(f, str(s)): v for f, s, v in zip(c['file'], c['sig'], c['lead_s'])}
    todo = [i for i, k in enumerate(zip(rel, sig)) if k not in have]
    if todo:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=WORKERS) as ex:
            res = list(ex.map(_lead_one, [paths[i] for i in todo], chunksize=8))
        for i, v in zip(todo, res):
            have[(rel[i], sig[i])] = v
        pd.DataFrame([{'file': f, 'sig': s, 'lead_s': v} for (f, s), v in have.items()]).to_csv(cp, index=False)
    return np.array([have[k] for k in zip(rel, sig)])


# ============================================================================ CV
def cv_scores(X, y, fold, trainable, zero_shot, mk, ext=()):
    """Fold model k trained on trainable rows with fold != k.  Score of row i = prediction of the fold
    model whose test fold holds it (fold[i]); rows in `zero_shot` = mean of the 5 fold models.
    ext = extra matrices scored by every fold model -> list of (5, n) arrays."""
    spw = float(np.sum(y[trainable] == 0)) / max(1, int(np.sum(y[trainable] == 1)))
    allp = np.zeros((L.N_FOLDS, len(y)))
    extp = [np.zeros((L.N_FOLDS, len(e))) for e in ext]
    for k in range(L.N_FOLDS):
        tr = trainable & (fold != k)
        _, preds = L.fit_predict(mk, X[tr], y[tr], [X] + list(ext), spw=spw)
        allp[k] = preds[0]
        for j in range(len(ext)):
            extp[j][k] = preds[1 + j]
    sc = allp[fold, np.arange(len(y))]
    sc = np.where(zero_shot, allp.mean(axis=0), sc)
    return sc, extp


def comparison_sets(sub, lib, mf):
    cc = sub == 'ccmusic'
    S = [('CCMusic notes vs SoundFont renders', cc, sub == 'soundfont_single'),
         ('CCMusic notes vs Otey notes', cc, sub == 'otey_single'),
         ('CCMusic notes vs algorithmic (SoundFont + Otey)', cc, np.isin(sub, ['soundfont_single', 'otey_single'])),
         ('CCMusic notes vs library notes: all 4 libraries, mf layer', cc, mf),
         ('CCMusic notes vs library notes: all 4 libraries, all layers', cc, lib != '')]
    S += [(f'CCMusic notes vs library notes: {lb}, mf layer', cc, (lib == lb) & mf) for lb in LIBS]
    S += [(f'CCMusic notes vs library notes: {lb}, all layers', cc, lib == lb) for lb in LIBS]
    return S


def short(name):
    return name.replace('CCMusic notes vs ', '').replace('library notes: ', 'library notes, ')


# ============================================================================ main
def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    xv = L.check_xgboost()
    import sklearn
    import scipy
    base, libs, clips, srcs, folds = build_lists()
    print(f'files: base {len(base)}, library notes {len(libs)}, clips {len(clips)}, clip sources {len(srcs)}', flush=True)

    # ---------------- extraction
    allp = list(base['path']) + list(libs['path']) + list(clips['path']) + list(srcs['path'])
    alll = list(base['loader']) + list(libs['loader']) + list(clips['loader']) + list(srcs['loader'])
    # loader-equivalence check: the first 20 files (sorted) of each base source also via load_mono + process_signal
    chk = pd.concat([g.sort_values('file').head(20) for _, g in base.groupby('subkind')])
    allp += list(chk['path'])
    alll += ['mono'] * len(chk)
    t1 = time.time()
    feat_all, FEATS = extract(allp, alll, 'main')
    t_ex = time.time() - t1
    nb, nl, nc, ns = len(base), len(libs), len(clips), len(srcs)
    fb = feat_all.iloc[:nb].reset_index(drop=True)
    fl = feat_all.iloc[nb:nb + nl].reset_index(drop=True)
    fc = feat_all.iloc[nb + nl:nb + nl + nc].reset_index(drop=True)
    fs = feat_all.iloc[nb + nl + nc:nb + nl + nc + ns].reset_index(drop=True)
    fchk = feat_all.iloc[nb + nl + nc + ns:].reset_index(drop=True)
    assert len(FEATS) == 88, len(FEATS)
    # sensitivity: library notes resampled to 44.1 kHz (K18 and Salamander are 48 kHz)
    t1 = time.time()
    fl44, _ = extract(list(libs['path']), list(libs['loader']), 'sr44k')
    # sensitivity: every unit in the listening-clip format (quality/build_listening_kit.process)
    cf_paths = list(base['path']) + list(libs['path']) + list(srcs['path'])
    fcf, _ = extract(cf_paths, ['clip'] * len(cf_paths), 'main')
    t_ex += time.time() - t1
    fbc = fcf.iloc[:len(base)].reset_index(drop=True)
    flc = fcf.iloc[len(base):len(base) + len(libs)].reset_index(drop=True)
    fsc = fcf.iloc[len(base) + len(libs):].reset_index(drop=True)

    # loader equivalence
    ref = fb.set_index('file').loc[[os.path.relpath(p, ROOT).replace('\\', '/') for p in chk['path']], FEATS].values
    eq_diff = float(np.nanmax(np.abs(ref - fchk[FEATS].values)))
    eq_nan = int((np.isnan(ref) != np.isnan(fchk[FEATS].values)).sum())
    sr44_same = {}
    for lb in LIBS:
        m = (libs['library'] == lb).values
        d = np.abs(fl.loc[m, FEATS].values - fl44.loc[m, FEATS].values)
        sr44_same[lb] = float(np.nanmax(d))

    # ---------------- extraction diagnostics
    fb['lead_s'] = lead_in(list(base['path']))
    fl['lead_s'] = lead_in(list(libs['path']))
    fc['lead_s'] = lead_in(list(clips['path']))
    fs['lead_s'] = lead_in(list(srcs['path']))
    pk_cc = float(fb.loc[(base.subkind == 'ccmusic').values, 'peak'].median())
    lvl = {lb: 20 * np.log10(float(fl.loc[(libs.library == lb).values, 'peak'].median()) / pk_cc) for lb in LIBS}
    diag = []
    src_tables = ([(sk, fb[(base.subkind == sk).values]) for sk in ('ccmusic', 'soundfont_single', 'otey_single')]
                  + [(f'library {lb} (all layers)', fl[(libs.library == lb).values]) for lb in LIBS]
                  + [('listening clips', fc), ('clip source files', fs)])
    for name, d in src_tables:
        diag.append([name, len(d), int((d['error'].fillna('') != '').sum()),
                     '/'.join(str(int(x)) for x in sorted(d['sr'].dropna().unique())),
                     '/'.join(str(int(x)) for x in sorted(d['channels'].dropna().unique())),
                     ', '.join(sorted(set(d['loader']))),
                     f"{d['duration_s'].median():.1f}", f"{d['lead_s'].median():.3f}", f"{d['peak'].median():.2f}",
                     int((d['n_wrap'] > 0).sum()), f"{d[FEATS].isna().values.mean():.3f}"])
    src_tables += [('sensitivity sr44k: library notes', fl44), ('sensitivity clip format: corpus notes', fbc),
                   ('sensitivity clip format: library notes', flc), ('sensitivity clip format: clip sources', fsc)]
    for name, d in src_tables[-4:]:
        diag.append([name, len(d), int((d['error'].fillna('') != '').sum()),
                     ('44100 (48000 resampled)' if 'sr44k' in name else
                      '/'.join(str(int(x)) for x in sorted(d['sr'].dropna().unique()))), '1' if 'clip' in name else '2',
                     ', '.join(sorted(set(d['loader']))) + (' + resample' if 'sr44k' in name else ''),
                     f"{d['duration_s'].median():.1f}", '0.010' if 'clip' in name else 'as native',
                     f"{d['peak'].median():.2f}",
                     int((d['n_wrap'] > 0).sum()), f"{d[FEATS].isna().values.mean():.3f}"])
    err_rows = pd.concat([d.assign(source=n) for n, d in src_tables])
    err_rows = err_rows[err_rows['error'].fillna('') != '']

    # ---------------- design matrices
    Xb, Xl, Xl44 = fb[FEATS].values.astype(float), fl[FEATS].values.astype(float), fl44[FEATS].values.astype(float)
    Xc, Xs = fc[FEATS].values.astype(float), fs[FEATS].values.astype(float)
    X = np.vstack([Xb, Xl])
    X44 = np.vstack([Xb, Xl44])
    Xbc, Xsc = fbc[FEATS].values.astype(float), fsc[FEATS].values.astype(float)
    XC = np.vstack([Xbc, flc[FEATS].values.astype(float)])
    y = np.r_[base['y'].values, libs['y'].values].astype(int)
    fold = np.r_[base['fold'].values, libs['fold'].values].astype(int)
    grp = np.r_[base['grp'].values, libs['grp'].values].astype(str)
    sub = np.r_[base['subkind'].values, ['library'] * nl].astype(str)
    lib = np.r_[[''] * nb, libs['library'].values].astype(str)
    mf = np.r_[np.zeros(nb, bool), libs['mf_layer'].values]
    isbase = np.r_[np.ones(nb, bool), np.zeros(nl, bool)]
    islib = ~isbase
    COND = {  # name: (X, trainable, zero_shot)
        'T1': (X, np.ones(len(y), bool), np.zeros(len(y), bool)),
        'T2': (X, isbase, islib),
        'T1mf': (X, isbase | mf, np.zeros(len(y), bool)),
        'T1_sr44k': (X44, np.ones(len(y), bool), np.zeros(len(y), bool)),
        'T2_sr44k': (X44, isbase, islib),
        'T1_clipfmt': (XC, np.ones(len(y), bool), np.zeros(len(y), bool)),
        'T2_clipfmt': (XC, isbase, islib),
    }
    runs, rows = {}, []
    t1 = time.time()
    for mk in ('xgb', 'lr'):
        for cond, (XX, trn, zs) in COND.items():
            tag = f'{mk}_eg_{cond}'
            sc, _ = cv_scores(XX, y, fold, trn, zs, mk)
            runs[tag] = sc
            for name, rm, sm in comparison_sets(sub, lib, mf):
                rows.append({'run': tag, 'comparison': name, **L.compare(y, sc, grp, rm, sm)})
            print(f'  {tag} done ({time.time() - t1:.0f} s)', flush=True)
    comp = pd.DataFrame(rows)
    comp.to_csv(os.path.join(OUT, 'egemaps_comparisons.csv'), index=False)
    sc_out = pd.DataFrame({'file': np.r_[base['file'].values, libs['file'].values], 'y': y, 'source': sub, 'library': lib,
                           'mf_layer': mf, 'grp': grp, 'fold': fold,
                           'duration_s': np.r_[fb['duration_s'].values, fl['duration_s'].values]})
    for tag, sc in runs.items():
        sc_out[tag] = sc
    sc_out.to_csv(os.path.join(OUT, 'egemaps_scores.csv'), index=False, float_format='%.6f')
    t_cv = time.time() - t1

    # ---------------- top features (full T2 XGBoost, all 1550 isolated corpus files)
    spw = float(np.sum(y[isbase] == 0)) / max(1, int(np.sum(y[isbase] == 1)))
    cc = sub == 'ccmusic'
    algm = np.isin(sub, ['soundfont_single', 'otey_single'])

    def top_features(XX):
        m, _ = L.fit_predict('xgb', XX[isbase], y[isbase], [XX[:1]], spw=spw)
        gain = m.get_booster().get_score(importance_type='total_gain')
        gain = pd.Series({FEATS[int(k[1:])]: v for k, v in gain.items()}).sort_values(ascending=False)
        gain = gain / gain.sum()
        out = []
        for f in gain.index[:10]:
            j = FEATS.index(f)

            def ua(msk):
                return L.auc_fast(np.r_[np.zeros(cc.sum()), np.ones(msk.sum())], np.r_[XX[cc, j], XX[msk, j]])
            out.append([f, f'{gain[f]:.3f}', f'{ua(algm):.3f}', f'{ua(mf):.3f}', f'{np.nanmedian(XX[cc, j]):.3g}',
                        f'{np.nanmedian(XX[algm, j]):.3g}', f'{np.nanmedian(XX[mf, j]):.3g}'])
        return out
    uni = top_features(X)
    uni_c = top_features(XC)

    # ---------------- listening clips, leave-own-group-out (protocol v2 section 4)
    CLASS_OF = {'Iowa Steinway': 'Iowa', 'K18 upright': 'K18', 'Salamander (sampled)': 'Salamander',
                'soundfont': 'SoundFont (Casio CA-110)', 'Otey physical model': 'Otey'}
    REC, ALG = ['Iowa', 'K18', 'Salamander'], ['SoundFont (Casio CA-110)', 'Otey']
    st = clips[['clip', 'source', 'note', 'src']].copy()
    st['cls'] = st['source'].map(CLASS_OF)
    otey_g, casio_g = 'syn:otey', 'syn:sf2:' + L.norm_token('Casio_CA_110_Piano_FM')
    assert casio_g in set(base['grp']), casio_g
    excl = {'full': [], 'noOtey': [otey_g], 'noCasio': [casio_g], 'noOteyCasio': [otey_g, casio_g]}
    own = np.where(st['cls'] == 'Otey', 'noOtey', np.where(st['cls'] == ALG[0], 'noCasio', 'full'))
    for mk in ('xgb', 'lr'):
        preds = {}
        for name, ex in excl.items():
            tr = ~base['grp'].isin(ex).values
            spw_c = float((base['y'].values[tr] == 0).sum()) / max(1, int((base['y'].values[tr] == 1).sum()))
            _, (pc, ps) = L.fit_predict(mk, Xb[tr], base['y'].values[tr], [Xc, Xs], spw=spw_c)
            preds[name] = (pc, ps)
        st[f'eg_{mk}_clip_LOGO'] = [preds[o][0][i] for i, o in enumerate(own)]
        st[f'eg_{mk}_src_LOGO'] = [preds[o][1][i] for i, o in enumerate(own)]
        st[f'eg_{mk}_clip_noOteyCasio'] = preds['noOteyCasio'][0]
        # same, models trained on clip-format units (format of train = format of the clips)
        preds = {}
        for name, ex in excl.items():
            tr = ~base['grp'].isin(ex).values
            spw_c = float((base['y'].values[tr] == 0).sum()) / max(1, int((base['y'].values[tr] == 1).sum()))
            _, (pc, ps) = L.fit_predict(mk, Xbc[tr], base['y'].values[tr], [Xc, Xsc], spw=spw_c)
            preds[name] = (pc, ps)
        st[f'egcf_{mk}_clip_LOGO'] = [preds[o][0][i] for i, o in enumerate(own)]
        st[f'egcf_{mk}_srcproc_LOGO'] = [preds[o][1][i] for i, o in enumerate(own)]
    s47 = pd.read_csv(os.path.join(AOUT, 'stimuli_detector_scores.csv')).set_index('clip')
    for c in ('xgb_all47_clip_LOGO', 'lr_all47_clip_LOGO', 'xgb_all47_src_LOGO'):
        st[f'v2_{c}'] = st['clip'].map(s47[c]).values
    st.to_csv(os.path.join(OUT, 'egemaps_clip_scores.csv'), index=False, float_format='%.6f')
    # the clip WAVs vs the clip transform re-applied to their sources (only difference: PCM_16 write, library versions)
    clip_dp = {mk: float(np.max(np.abs(st[f'egcf_{mk}_clip_LOGO'].values - st[f'egcf_{mk}_srcproc_LOGO'].values)))
               for mk in ('xgb', 'lr')}
    rel = np.abs(Xc - Xsc) / np.maximum(np.abs(Xc), 1e-9)
    clip_feat_rel = float(np.nanmedian(rel))
    clip_feat_rel99 = float(np.nanpercentile(rel, 99))
    cls_arr = st['cls'].values
    rec = np.isin(cls_arr, REC)
    alg = ~rec
    oldreal = np.isin(cls_arr, ['Iowa', 'K18'])
    sal = cls_arr == 'Salamander'

    def clip_ci(p, neg, pos, n=L.N_BOOT):          # = protocol_v2.score_stimuli.clip_ci
        rng = np.random.default_rng(L.SEED)
        idx_by = {c: np.where(cls_arr == c)[0] for c in REC + ALG}
        v = np.empty(n)
        for b in range(n):
            ii = np.concatenate([rng.choice(ix, len(ix), replace=True) for ix in idx_by.values()])
            mm = neg[ii] | pos[ii]
            v[b] = L.auc_fast(pos[ii][mm].astype(int), p[ii][mm])
        return np.nanpercentile(v, 2.5), np.nanpercentile(v, 97.5)

    clip_rows, clip_key = [], {}
    for col, label in [('eg_xgb_clip_LOGO', 'eGeMAPS XGB, clip, LOGO (primary)'),
                       ('eg_xgb_clip_noOteyCasio', 'eGeMAPS XGB, clip, one model w/o Otey+CA-110'),
                       ('eg_lr_clip_LOGO', 'eGeMAPS LR, clip, LOGO'),
                       ('eg_xgb_src_LOGO', 'eGeMAPS XGB, unprocessed source file, LOGO'),
                       ('eg_lr_src_LOGO', 'eGeMAPS LR, unprocessed source file, LOGO'),
                       ('egcf_xgb_clip_LOGO', 'eGeMAPS XGB trained on clip-format units, clip, LOGO'),
                       ('egcf_lr_clip_LOGO', 'eGeMAPS LR trained on clip-format units, clip, LOGO'),
                       ('v2_xgb_all47_clip_LOGO', '47-feature XGB all47, clip, LOGO (RESULTS_v2 s.4)'),
                       ('v2_lr_all47_clip_LOGO', '47-feature LR all47, clip, LOGO (RESULTS_v2 s.4)'),
                       ('v2_xgb_all47_src_LOGO', '47-feature XGB all47, source file, LOGO (RESULTS_v2 s.4)')]:
        p = st[col].values.astype(float)
        cnt = {c: f"{int((p[cls_arr == c] > 0.5).sum())}/{int((cls_arr == c).sum())}" for c in REC + ALG}
        a1 = L.auc_fast(alg[rec | alg].astype(int), p[rec | alg])
        lo1, hi1 = clip_ci(p, rec, alg)
        a2 = L.auc_fast(alg[oldreal | alg].astype(int), p[oldreal | alg])
        a3 = L.auc_fast(sal[oldreal | sal].astype(int), p[oldreal | sal])
        clip_rows.append([label] + [cnt[c] for c in REC + ALG] + [f'{a1:.3f} [{lo1:.3f}, {hi1:.3f}]', f'{a2:.3f}', f'{a3:.3f}'])
        clip_key[col] = {'counts': cnt, 'auc_rec_vs_alg': a1, 'ci': [lo1, hi1], 'auc_iowak18_vs_alg': a2,
                         'auc_sal_vs_iowak18': a3, 'meanP': {c: float(p[cls_arr == c].mean()) for c in REC + ALG}}

    # ---------------- 47-feature comparators (RESULTS_v2 sections 2-3)
    pn = pd.read_csv(os.path.join(AOUT, 'pernote_comparisons.csv'))
    cmp47 = pd.read_csv(os.path.join(AOUT, 'comparisons.csv'))
    iso47 = pd.read_csv(os.path.join(AOUT, 'oof_scores_isolated_only.csv'))

    def v2row(run, name):
        if name == 'CCMusic notes vs algorithmic (SoundFont + Otey)':
            if run.endswith('_Ionly'):
                sk = iso47['subkind'].values
                return pd.Series(L.compare(iso47['y'].values, iso47[run].values, iso47['grp'].values,
                                           sk == 'ccmusic', np.isin(sk, ['soundfont_single', 'otey_single'])))
            r = cmp47[(cmp47.run == run) & (cmp47.id == 'I1')]
        else:
            r = pn[(pn.run == run) & (pn.comparison == name)]
        return r.iloc[0] if len(r) else None

    names = [n for n, _, _ in comparison_sets(sub, lib, mf)]

    def cell(r, det=True):
        if r is None:
            return 'n/a'
        return L.fci(r) + (f"; det {L.fpct(r['det_syn'])}" if det else '')

    def eg(tag, name):
        return comp[(comp.run == tag) & (comp.comparison == name)].iloc[0]

    # ---------------- determinism check
    files = ['egemaps_comparisons.csv', 'egemaps_scores.csv', 'egemaps_clip_scores.csv']
    ck = os.path.join(OUT, 'checksums.txt')
    prev = {}
    if os.path.exists(ck):
        for line in open(ck, encoding='utf-8'):
            if '  ' in line:
                h, f = line.strip().split('  ', 1)
                prev[f] = h
    now = {f: L.md5_of(os.path.join(OUT, f)) for f in files}
    with open(ck, 'w', encoding='utf-8') as fh:
        for f, h in now.items():
            fh.write(f'{h}  {f}\n')
    if prev:
        same = [f for f in now if prev.get(f) == now[f]]
        det_note = f'Determinism check vs the previous run (out/checksums.txt): {len(same)} of {len(now)} output CSVs byte-identical.'
    else:
        det_note = 'Determinism check: first run (no previous out/checksums.txt); run again to compare.'

    # ---------------- report
    ver = subprocess.run([sys.executable, '-c',
                          'import numpy, sys, os; sys.path.insert(0, os.environ["EGEMAPS_OPENSMILE_PATH"]); '
                          'import pandas, opensmile, audinterface, audiofile; '
                          'print(opensmile.__version__, audinterface.__version__, audiofile.__version__, pandas.__version__, numpy.__version__)'],
                         env={**os.environ, 'EGEMAPS_OPENSMILE_PATH': OSP}, capture_output=True, text=True).stdout.split()
    md = ['# eGeMAPS baseline under protocol v2 (format-matched, isolated notes)\n\n',
          f'Generated {time.strftime("%Y-%m-%d %H:%M")} by one command, from the project root:\n\n',
          '```\npython Stattia/revision/egemaps/run_egemaps.py\n```\n\n',
          f'Models: Python {platform.python_version()}, numpy {np.__version__}, pandas {pd.__version__}, scikit-learn '
          f'{sklearn.__version__}, scipy {scipy.__version__}, xgboost {xv} (n_jobs=1, random_state=0) -- the protocol-v2 '
          f'environment. Extraction (separate interpreter, `egemaps_extract.py`): opensmile {ver[0] if ver else "?"}, '
          f'audinterface {ver[1] if len(ver) > 1 else "?"}, audiofile {ver[2] if len(ver) > 2 else "?"}, pandas '
          f'{ver[3] if len(ver) > 3 else "?"} (needed by audinterface), numpy {ver[4] if len(ver) > 4 else "?"} (system). '
          f'opensmile was installed with `pip install --target` into a scratch folder; the system Python is unchanged. '
          f'Runtime {time.time() - t0:.0f} s (extraction {t_ex:.0f} s incl. cache hits, CV {t_cv:.0f} s). {det_note}\n\n',
          'Scripts: `Stattia/revision/egemaps/{run_egemaps, egemaps_extract}.py`. Outputs (`Stattia/revision/egemaps/out/`): '
          '`egemaps_comparisons.csv` (every AUC row, incl. CCMusic false alarms), `egemaps_scores.csv` (per-file scores, '
          'every run), `egemaps_clip_scores.csv`, `key_numbers_egemaps.json`, `cache/egemaps_{main,sr44k}.csv` '
          '(88 functionals per file and loader; delete to re-extract), `cache/lead_in.csv`.\n\n']

    # ---- section 0: key numbers vs 47 features
    md.append('## 0. Key numbers: eGeMAPS vs the 47-feature XGBoost (RESULTS_v2)\n\n')
    MFN = 'CCMusic notes vs library notes: all 4 libraries, mf layer'
    ALN = 'CCMusic notes vs library notes: all 4 libraries, all layers'
    AGN = 'CCMusic notes vs algorithmic (SoundFont + Otey)'

    def ac(r, det=False):
        s = f"{r['auc']:.3f} [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]"
        return s + (f", det {L.fpct(r['det_syn'])}" if det else '')
    lib_cells = []
    for tag, lab in (('xgb_eg_T1', 'XGB T1'), ('xgb_eg_T2', 'XGB T2'), ('lr_eg_T1', 'LR T1'), ('lr_eg_T2', 'LR T2'),
                     ('xgb_eg_T1_clipfmt', 'XGB T1 clip format'), ('xgb_eg_T2_clipfmt', 'XGB T2 clip format'),
                     ('lr_eg_T1_clipfmt', 'LR T1 clip format'), ('lr_eg_T2_clipfmt', 'LR T2 clip format')):
        for nm, ln in ((MFN, 'mf'), (ALN, 'all layers')):
            r = eg(tag, nm)
            side = 'above' if r['ci_lo'] > 0.5 else ('below' if r['ci_hi'] < 0.5 else 'includes')
            lib_cells.append((lab, ln, r, side))
    j_c = FEATS.index(uni_c[0][0])
    leads = {sk: float(fb.loc[(base.subkind == sk).values, 'lead_s'].median()) for sk in ('ccmusic', 'soundfont_single', 'otey_single')}
    leads_l = {lb: float(fl.loc[(libs.library == lb).values, 'lead_s'].median()) for lb in LIBS}
    md.append('**Summary (all numbers from the tables below).**\n\n')
    md.append(f"- CCMusic notes vs algorithmic notes: eGeMAPS XGB {ac(eg('xgb_eg_T1', AGN), True)} (T1), "
              f"{ac(eg('xgb_eg_T2', AGN), True)} (T2); 47-feature XGB {ac(v2row('xgb_all47_T1', AGN))} (T1), "
              f"{ac(v2row('xgb_all47_T2', AGN))} (T2), {ac(v2row('xgb_all47_Ionly', AGN))} (I-only). Submitted eGeMAPS "
              f"value 0.950.\n")
    md.append(f"- CCMusic notes vs library notes (pooled, mf layer): eGeMAPS XGB {ac(eg('xgb_eg_T1', MFN), True)} (T1), "
              f"{ac(eg('xgb_eg_T2', MFN), True)} (T2); eGeMAPS LR {ac(eg('lr_eg_T1', MFN))} (T1), {ac(eg('lr_eg_T2', MFN))} "
              f"(T2); 47-feature XGB {ac(v2row('xgb_all47_T1', MFN))} (T1), {ac(v2row('xgb_all47_T2', MFN))} (T2), "
              f"{ac(v2row('xgb_all47_Ionly', MFN))} (I-only). Submitted eGeMAPS 'sample-based' value 0.065 "
              f"(concatenated 14-note takes, not per-note units).\n")
    md.append('- Position of the pooled library AUC relative to chance (95% CI above / including / below 0.5): '
              + '; '.join(f'{lab} {ln}: {side} ({r["auc"]:.3f})' for lab, ln, r, side in lib_cells) + '.\n')
    md.append(f"- Native files: the three eGeMAPS features with the largest XGBoost gain are {uni[0][0]}, {uni[1][0]}, "
              f"{uni[2][0]} (together {sum(float(u[1]) for u in uni[:3]):.2f} of the total gain). Median lead-in from file "
              f"start to the attack: CCMusic {leads['ccmusic']:.3f} s, SoundFont renders {leads['soundfont_single']:.3f} s, "
              f"Otey {leads['otey_single']:.3f} s, libraries " + ', '.join(f'{k} {v:.3f} s' for k, v in leads_l.items())
              + '; median durations and levels differ too (section 1). In the clip format (section 3.2), where lead-in, '
              f"duration, level and sample rate are equalised, the top feature is {uni_c[0][0]} (gain share {uni_c[0][1]}; "
              f"median CCMusic {np.nanmedian(XC[cc, j_c]):.3g}, SoundFont {np.nanmedian(XC[sub == 'soundfont_single', j_c]):.3g}, "
              f"Otey {np.nanmedian(XC[sub == 'otey_single', j_c]):.3g}, library mf {np.nanmedian(XC[mf, j_c]):.3g})"
              + ('; the SoundFont renders (median 1.8 s) are zero-padded to 3 s in this format, so they end in digital silence'
                 if uni_c[0][0].startswith('MeanUnvoiced') else '') + '.\n\n')
    md.append('Cells: AUC [95% CI]; det = share of synthetic files with P(synthetic) > 0.5. The eGeMAPS detector is '
              'trained on isolated notes only (CCMusic vs SoundFont renders + Otey, plus library notes under T1). '
              'Its T2 is therefore the analogue of the 47-feature **I-only** detector (isolated notes only, libraries '
              'zero-shot); the 47-feature T1 / T2 columns come from the whole-corpus models (performances in training; '
              'under T1 the libraries were in training as 14-note takes, not as single notes).\n\n')
    key_names = ['CCMusic notes vs SoundFont renders', 'CCMusic notes vs Otey notes',
                 'CCMusic notes vs algorithmic (SoundFont + Otey)',
                 'CCMusic notes vs library notes: all 4 libraries, mf layer',
                 'CCMusic notes vs library notes: all 4 libraries, all layers']
    tb = []
    for name in key_names:
        tb.append([short(name), cell(eg('xgb_eg_T1', name)), cell(eg('xgb_eg_T2', name)),
                   cell(v2row('xgb_all47_T1', name)), cell(v2row('xgb_all47_T2', name)), cell(v2row('xgb_all47_Ionly', name))])
    md.append(L.md_table(['CCMusic notes vs', 'eGeMAPS XGB T1', 'eGeMAPS XGB T2', '47-feat XGB T1', '47-feat XGB T2',
                          '47-feat XGB I-only'], tb))
    md.append('\nSame, logistic regression:\n\n')
    tb = []
    for name in key_names:
        tb.append([short(name), cell(eg('lr_eg_T1', name)), cell(eg('lr_eg_T2', name)),
                   cell(v2row('lr_all47_T1', name)), cell(v2row('lr_all47_T2', name)), cell(v2row('lr_all47_Ionly', name))])
    md.append(L.md_table(['CCMusic notes vs', 'eGeMAPS LR T1', 'eGeMAPS LR T2', '47-feat LR T1', '47-feat LR T2',
                          '47-feat LR I-only'], tb))
    md.append('\nFormat-equalised sensitivity (section 3.2): every unit (CCMusic, SoundFont, Otey, library notes) passed '
              'through the listening-clip transform (44.1 kHz mono, onset-trimmed, 3 s, RMS-normalised) before eGeMAPS:\n\n')
    tb = []
    for name in key_names:
        tb.append([short(name)] + [cell(eg(f'{mk}_eg_{c}', name)) for mk in ('xgb', 'lr') for c in ('T1_clipfmt', 'T2_clipfmt')])
    md.append(L.md_table(['CCMusic notes vs', 'eGeMAPS XGB T1, clip format', 'eGeMAPS XGB T2, clip format',
                          'eGeMAPS LR T1, clip format', 'eGeMAPS LR T2, clip format'], tb))
    ck_ = clip_key['eg_xgb_clip_LOGO']
    c47 = clip_key['v2_xgb_all47_clip_LOGO']
    md.append(f"\nListening clips (LOGO, XGB): eGeMAPS detects {', '.join(f'{c} {v}' for c, v in ck_['counts'].items())}; "
              f"AUC recorded vs algorithmic {ck_['auc_rec_vs_alg']:.3f} [{ck_['ci'][0]:.3f}, {ck_['ci'][1]:.3f}] "
              f"(47-feature: {', '.join(f'{c} {v}' for c, v in c47['counts'].items())}; AUC {c47['auc_rec_vs_alg']:.3f}).\n\n")
    md.append('Submitted paper (Table 2 row 3, `quality/opensmile_baseline.py`, different units and folds): eGeMAPS '
              '0.950 [0.854, 1.000] real vs algorithmic, 0.065 [0.001, 0.194] real vs sample-based (8 concatenated '
              '14-note takes per library, in training as synthetic).\n\n')

    # ---- section 1: data + extraction
    md.append('## 1. Units, folds and audio loading\n\n')
    md.append(f'File lists = protocol v2 section 3: {int((base.subkind == "ccmusic").sum())} CCMusic note files, '
              f'{int((base.subkind == "soundfont_single").sum())} SoundFont single-note renders, '
              f'{int((base.subkind == "otey_single").sum())} Otey notes (rows of `analysis/out/folds.csv`, with their CV '
              f'group and FIXED fold), {len(libs)} library note files ({int(libs.mf_layer.sum())} in the mf-equivalent layer; '
              f'the rows of `analysis/out/pernote_library_scores.csv`, checked identical to `protocol_v2.library_note_files()`). '
              'Every unit is one struck note in its own file. Library notes take their library\'s fold from folds.csv '
              '(FreePats, Iowa: fold 1; K18, Salamander: fold 2), so under T1 each library is scored by the fold model '
              'that never saw that library. CCMusic = 6 brand groups (protocol v2), not the 8 filename-prefix groups of '
              'the submitted baseline.\n\n')
    md.append('**Feature extraction** = eGeMAPSv02 functionals (88), `opensmile.Smile(feature_set=eGeMAPSv02, '
              'feature_level=Functionals)` with default options (channels=0, mixdown=False, no resampling), i.e. the call of '
              '`quality/opensmile_baseline.py`. Audio loading is the baseline\'s: single-note corpus files and clips via '
              '`smile.process_file(path)` (baseline l.59); library audio via `detune.segment.load_mono` (mean of channels, '
              'native rate) + `smile.process_signal` (baseline l.95/101, used there for the concatenated takes).\n\n')
    md.append(L.md_table(['source', 'files', 'errors', 'sample rate', 'channels', 'loader', 'median duration (s)',
                          'median lead-in before the attack (s)', 'median peak |x|', 'files with |x| >= 1',
                          'NaN rate (88 features)'], diag))
    md.append('\nLead-in = time from the file start to the first sample above 2% of the file peak (the onset rule of '
              'the listening-clip transform); clip-format units start 10 ms before that point. Median peak of the '
              'library notes relative to CCMusic: ' + ', '.join(f'{k} {v:+.0f} dB' for k, v in lvl.items()) + '.\n')
    md.append(f'\nDifferences from the submitted baseline and checks:\n\n'
              f'1. **Units**: library notes are single-note files (format-matched to CCMusic / SoundFont / Otey), not '
              f'concatenations of 14 random notes; nothing else is concatenated.\n'
              f'2. **Loader equivalence**: all CCMusic, SoundFont, Otey and clip files are mono, so the two loading routes '
              f'coincide for them: {len(chk)} files (first 20 of each source) extracted both ways differ by max |d| = '
              f'{eq_diff:.3g} ({eq_nan} NaN mismatches). The library files are stereo: as in the baseline they are mixed '
              f'to mono (mean of channels); `process_file` would have used the left channel only. Iowa .aiff files cannot '
              f'be read by `process_file` in this environment (audiofile needs mediainfo/ffmpeg for AIFF); the baseline '
              f'never did either.\n'
              f'3. **Sample rate**: K18 and Salamander are 48 kHz, everything else 44.1 kHz; eGeMAPS runs at the native '
              f'rate (as the baseline and the 47-feature pipeline). Sensitivity: library notes resampled to 44.1 kHz with '
              f'opensmile\'s own resampler (`sampling_rate=44100, resample=True`; section 3.1). Check, max |d| between '
              f'the resampled and the native features: FreePats {sr44_same["FreePats"]:.3g}, Iowa {sr44_same["Iowa"]:.3g} '
              f'(44.1 kHz already, so unchanged as expected); K18 {sr44_same["K18"]:.3g}, Salamander '
              f'{sr44_same["Salamander"]:.3g} (resampled).\n'
              f'4. **int16 conversion**: the opensmile wrapper feeds `(x * 32768).astype(int16)` to openSMILE, which wraps '
              f'around for |x| >= 1; the column "files with |x| >= 1" counts the affected files (same behaviour as the '
              f'baseline).\n'
              f'5. **Errors** are recorded, not silently dropped (the baseline\'s `except: pass`): '
              f'{len(err_rows)} file(s) with an extraction error.\n'
              f'6. **Folds / model / CI**: fixed protocol-v2 folds instead of `GroupKFold` (tie-order dependent); XGBoost '
              f'hyper-parameters as the baseline but n_jobs=1, random_state=0; scale_pos_weight = global real/synthetic '
              f'ratio of the training population; stratified grouped bootstrap (real and synthetic groups resampled '
              f'separately, B = 2000, seed 0) instead of the baseline\'s unstratified group bootstrap.\n\n')
    if len(err_rows):
        md.append(L.md_table(['source', 'file', 'error'], [[r.source, r.file, r.error] for r in err_rows.itertuples()]))
        md.append('\n')

    # ---- section 2: full tables
    md.append('## 2. Per-note isolated regime with eGeMAPS (out-of-fold / zero-shot)\n\n')
    md.append('"det." = share of synthetic files with P(synthetic) > 0.5; "mean P" = mean P(synthetic) of the synthetic '
              'files. CCMusic false alarms (share of CCMusic files with P > 0.5) and mean P are given below each table.\n\n')
    for mk, title in (('xgb', '2.1 XGBoost, 88 eGeMAPS functionals'), ('lr', '2.2 Logistic regression, 88 eGeMAPS functionals')):
        md.append(f'### {title}\n\n')
        tb = []
        for name in names:
            a, b = eg(f'{mk}_eg_T1', name), eg(f'{mk}_eg_T2', name)
            tb.append([short(name), f"{a['n_real']}/{a['g_real']} vs {a['n_syn']}/{a['g_syn']}",
                       L.fci(a), L.fpct(a['det_syn']), f"{a['meanP_syn']:.3f}",
                       L.fci(b), L.fpct(b['det_syn']), f"{b['meanP_syn']:.3f}"])
        md.append(L.md_table(['CCMusic notes vs', 'files/groups', 'T1 AUC [95% CI]', 'T1 det.', 'T1 mean P',
                              'T2 AUC [95% CI]', 'T2 det.', 'T2 mean P'], tb))
        a, b = eg(f'{mk}_eg_T1', names[0]), eg(f'{mk}_eg_T2', names[0])
        md.append(f"\nCCMusic false alarms: T1 {L.fpct(a['fa_real'])} (mean P {a['meanP_real']:.3f}); "
                  f"T2 {L.fpct(b['fa_real'])} (mean P {b['meanP_real']:.3f}).\n\n")

    md.append('### 2.3 Per-library AUC next to the 47-feature detector (XGBoost)\n\n')
    tb = []
    for name in names[3:]:
        tb.append([short(name), L.fci(eg('xgb_eg_T1', name)), L.fci(eg('xgb_eg_T2', name)),
                   cell(v2row('xgb_all47_T1', name), False), cell(v2row('xgb_all47_T2', name), False),
                   cell(v2row('xgb_all47_Ionly', name), False)])
    md.append(L.md_table(['CCMusic notes vs', 'eGeMAPS T1', 'eGeMAPS T2', '47-feat T1', '47-feat T2', '47-feat I-only'], tb))
    md.append('\n')

    # ---- section 3: sensitivity
    md.append('## 3. Sensitivity analyses\n\n### 3.1 Which library notes are in training (T1) and the sample rate\n\n')
    md.append('**T1mf** = only the 224 mf-layer library notes in training (other layers scored by the fold model of their '
              'library, which never saw that library). **sr44k** = library notes resampled to 44.1 kHz before extraction '
              '(K18 and Salamander change; FreePats and Iowa are 44.1 kHz already). Cells: AUC [95% CI]; det.\n\n')
    for mk in ('xgb', 'lr'):
        tb = []
        for name in names:
            tb.append([short(name)] + [cell(eg(f'{mk}_eg_{c}', name)) for c in ('T1', 'T1mf', 'T1_sr44k', 'T2', 'T2_sr44k')])
        md.append(f"**{'XGBoost' if mk == 'xgb' else 'Logistic regression'}**\n\n")
        md.append(L.md_table(['CCMusic notes vs', 'T1 (all layers in training)', 'T1mf', 'T1 sr44k', 'T2', 'T2 sr44k'], tb))
        md.append('\n')
    md.append('### 3.2 Format-equalised units (listening-clip format)\n\n')
    md.append('Every unit -- CCMusic, SoundFont, Otey and library notes -- is first passed through '
              '`quality/build_listening_kit.process`, the transform that produced the 30 listening clips: librosa load at '
              '44.1 kHz mono, start 10 ms before the first sample above 2% of the peak, 3.0 s (zero-padded if shorter), '
              '10 ms fades, RMS scaled to 0.1 (-20 dBFS), peak limited to 0.97. This removes the differences in duration, '
              'lead-in before the attack, level and sample rate between the sources (section 1 table; "3.0 s" also cuts '
              'the long library decays, and pads the 1.8 s SoundFont renders with digital silence); it is not part of the '
              'submitted baseline. Same folds, models and conditions. Cells: AUC [95% CI]; det.\n\n')
    for mk in ('xgb', 'lr'):
        tb = []
        for name in names:
            tb.append([short(name)] + [cell(eg(f'{mk}_eg_{c}', name)) for c in ('T1', 'T1_clipfmt', 'T2', 'T2_clipfmt')])
        md.append(f"**{'XGBoost' if mk == 'xgb' else 'Logistic regression'}**\n\n")
        md.append(L.md_table(['CCMusic notes vs', 'T1 (native files)', 'T1, clip format', 'T2 (native files)',
                              'T2, clip format'], tb))
        a, b = eg(f'{mk}_eg_T1_clipfmt', names[0]), eg(f'{mk}_eg_T2_clipfmt', names[0])
        md.append(f"\nCCMusic false alarms, clip format: T1 {L.fpct(a['fa_real'])} (mean P {a['meanP_real']:.3f}); "
                  f"T2 {L.fpct(b['fa_real'])} (mean P {b['meanP_real']:.3f}).\n\n")

    # ---- section 4: what eGeMAPS uses
    md.append('## 4. What the eGeMAPS XGBoost uses\n\n')
    md.append('Top 10 features by share of total gain, XGBoost fitted on all 1550 isolated corpus files (T2 population); '
              'univariate AUC (feature value as score; < 0.5 = lower in the synthetic class) and medians.\n\n'
              '**Native files** (as the baseline):\n\n')
    md.append(L.md_table(['feature', 'gain share', 'univariate AUC CCMusic vs algorithmic',
                          'univariate AUC CCMusic vs library notes (mf)', 'median CCMusic', 'median algorithmic',
                          'median library (mf)'], uni))
    md.append('\n**Clip format** (section 3.2):\n\n')
    md.append(L.md_table(['feature', 'gain share', 'univariate AUC CCMusic vs algorithmic',
                          'univariate AUC CCMusic vs library notes (mf)', 'median CCMusic', 'median algorithmic',
                          'median library (mf)'], uni_c))
    md.append('\n')

    # ---- section 5: listening clips
    md.append('## 5. Listening-test clips scored out of sample (leave-own-group-out, as RESULTS_v2 section 4)\n\n')
    md.append('eGeMAPS of each of the 30 clips the listeners heard (`listening_kit/web/clip_XX.wav`, 3 s, onset-trimmed, '
              'RMS-normalised; `process_file`) and of their unprocessed source files. Each clip is scored by a model fitted '
              'on all T2 training files (CCMusic + SoundFont renders + Otey; no library) with its own group removed: Otey '
              'clips by a model without Otey, Casio CA-110 clips by a model without that SoundFont, library clips by the full '
              'model. Unlike RESULTS_v2 s.4, the training set has no performances (eGeMAPS was extracted for isolated notes '
              'only). Detection = P(synthetic) > 0.5. The 47-feature rows are recomputed from '
              '`analysis/out/stimuli_detector_scores.csv` with the same code. "Trained on clip-format units" = the same '
              'LOGO scheme with every training file in the clip format of section 3.2, so that training and test units '
              f'have the same format. Check: the clip WAVs and the clip transform re-applied to their source files give '
              f'eGeMAPS values with median relative difference {clip_feat_rel:.2g} (99th percentile {clip_feat_rel99:.2g}) '
              f'and scores differing by max |dP| = {clip_dp["xgb"]:.3f} (XGB) / {clip_dp["lr"]:.3f} (LR) (the WAVs are '
              f'PCM_16).\n\n')
    md.append(L.md_table(['scoring', 'Iowa det.', 'K18 det.', 'Salamander det.', 'SoundFont det.', 'Otey det.',
                          'AUC recorded(20) vs algorithmic(10) [95% CI, clips resampled within class]',
                          'AUC Iowa+K18(14) vs algorithmic(10)', 'AUC Salamander(6) vs Iowa+K18(14)'], clip_rows))
    md.append('\nPer-clip P(synthetic):\n\n')
    md.append(L.md_table(['clip', 'class', 'note', 'eGeMAPS XGB LOGO', 'eGeMAPS LR LOGO', 'eGeMAPS XGB src LOGO',
                          'eGeMAPS XGB LOGO, clip-format training', '47-feat XGB LOGO'],
                         [[r['clip'], r['cls'], r['note'], f"{r['eg_xgb_clip_LOGO']:.3f}", f"{r['eg_lr_clip_LOGO']:.3f}",
                           f"{r['eg_xgb_src_LOGO']:.3f}", f"{r['egcf_xgb_clip_LOGO']:.3f}",
                           f"{r['v2_xgb_all47_clip_LOGO']:.3f}"] for _, r in st.iterrows()]))

    text = ''.join(md)
    with open(os.path.join(HERE, 'RESULTS_EGEMAPS.md'), 'w', encoding='utf-8') as fh:
        fh.write(text)
    keyn = {f'{r.run}:{r.comparison}': {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v)
                                         for k, v in r._asdict().items() if k not in ('Index',)} for r in comp.itertuples()}
    json.dump({'comparisons': keyn, 'clips': clip_key, 'loader_equivalence_max_abs_diff': eq_diff,
               'sr44k_unchanged_libs_max_abs_diff': sr44_same},
              open(os.path.join(OUT, 'key_numbers_egemaps.json'), 'w'), indent=1,
              default=lambda o: o.item() if isinstance(o, (np.floating, np.integer, np.bool_)) else str(o))
    print(f'wrote {os.path.join(HERE, "RESULTS_EGEMAPS.md")} ({time.time() - t0:.0f} s)')


if __name__ == '__main__':
    main()
