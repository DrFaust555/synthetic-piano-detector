"""Protocol v3 -- controls for the isolated-note claim (CODE_AUDIT.md B2).

I-only XGBoost (CCMusic notes vs SoundFont + Otey single notes; library notes zero-shot, own-fold scoring),
all47 and med16, T2, re-extracted features under:
  orig    : the stored renders / recordings
  C1a     : every isolated note cut to 1.2 s after its onset + 0.1-s raised-cosine fade (before any
            SoundFont note-off at 1.4 s)
  C1b     : SoundFont notes re-rendered with the key held 5.0 s and Otey notes rendered for 5.0 s
            (v3_render.py), every source cut at 4.0 s after onset + 0.1-s fade; real notes (CCMusic,
            library notes) kept only if they still sound at 4.0 s (frame level >= -60 dB re peak)
  C3      : every note convolved with a measured room IR (MIT IR Survey, one IR drawn per file) + white
            noise at 40 dB SNR re the note RMS; real notes too
  C1a+C3, C1b+C3 : C3 applied to the full note, then the cut
C2: SoundFont synthesis-method categories from the SF2 data (v3_sf2.py), per analysed window.
C4: phaseogram drift irregularity of the figure analysis' 190 notes after C3.
C5: loop / exact-repetition index on the C1b notes.
"""
from __future__ import annotations

import os
import re
import sys
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

import pv2_lib as L
import v3_lib as V
import v3_sf2 as S2
import v3_render as R
from pv2_lib import FEATURE_COLS, ROOT

CONDS = ['orig', 'c1a', 'c1b', 'c3', 'c1a+c3', 'c1b+c3']
WINDOW = {'orig': 1.8, 'c1a': 1.3, 'c1b': 4.1, 'c3': 1.8, 'c1a+c3': 1.3, 'c1b+c3': 4.1}
CLABEL = {'orig': 'original', 'c1a': 'C1a 1.2 s', 'c1b': 'C1b 4.0 s', 'c3': 'C3 room+noise',
          'c1a+c3': 'C1a+C3', 'c1b+c3': 'C1b+C3'}


def iso_sources(df, libv):
    """one table of all isolated-note files: source, group, fold, original path, C1b path."""
    rows = []
    for _, r in df[df.subkind == 'ccmusic'].iterrows():
        rows.append(dict(src='CCMusic', y=0, grp=r.grp, fold=r.fold, file=r.file, path=V.audio_path(r.file),
                         path_c1b=V.audio_path(r.file), inst=r.file.split('/')[-1].split('_')[0]))
    for _, r in df[(df.subkind == 'soundfont_single') & ~df.sf_dup].iterrows():
        inst = r.file.split('/')[-2]
        rows.append(dict(src='SoundFont', y=1, grp=r.grp, fold=r.fold, file=r.file, path=V.audio_path(r.file),
                         path_c1b=os.path.join(R.DST, 'sf', inst, os.path.basename(r.file)), inst=inst))
    for _, r in df[df.subkind == 'otey_single'].iterrows():
        rows.append(dict(src='Otey', y=1, grp=r.grp, fold=r.fold, file=r.file, path=V.audio_path(r.file),
                         path_c1b=os.path.join(R.DST, 'otey', os.path.basename(r.file)), inst='Otey'))
    lib_fold = {g: int(df.loc[df.grp == g, 'fold'].iloc[0]) for g in libv['grp'].unique()}
    for _, r in libv.iterrows():
        f = os.path.relpath(r['path'], ROOT).replace('\\', '/')
        rows.append(dict(src='library', y=1, grp=r['grp'], fold=lib_fold[r['grp']], file=f, path=r['path'],
                         path_c1b=r['path'], inst=r['library'], mf=bool(r['mf_layer'])))
    t = pd.DataFrame(rows)
    t['mf'] = t['mf'].fillna(False).astype(bool)
    return t


def sf2_categories(src):
    """C2 categories per SoundFont note for each analysed window."""
    pdts = {R.inst_name(p): S2.read_pdta(p) for p in R.sf2_list()}
    out = []
    for _, r in src[src.src == 'SoundFont'].iterrows():
        m = re.match(r'([A-G])(s)?(-?\d)_v(\d+)\.wav', os.path.basename(r.file))
        pc = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}[m.group(1)] + (1 if m.group(2) else 0)
        key = 12 * (int(m.group(3)) + 1) + pc
        info = S2.note_info(pdts[r.inst], key, int(m.group(4)))
        rec = {'file': r.file, 'inst': r.inst, 'key': key, 'vel': int(m.group(4))}
        if info:
            rec.update({k: info[k] for k in ('sample', 'root', 'transpose', 'abs_transpose', 'looped', 't_loop',
                                              't_end', 'loop_len', 'n_layers')})
        for c, w in WINDOW.items():
            rec[f'cat_{c}'] = S2.category(info, w)
        out.append(rec)
    return pd.DataFrame(out)


def features(src, cond, log=print):
    if cond == 'orig':
        return None
    use = src.copy()
    use['p'] = use['path_c1b'] if cond.startswith('c1b') else use['path']
    # the note's ORIGINAL relative path keys the cache and seeds the C3 RNG, so a note gets the same IR and
    # noise in C3, C1a+C3 and C1b+C3 (and in C4)
    use['rel'] = use['file'].where(~use['file'].str.startswith('ccmusic_pianos/'), 'audio/' + use['file'])
    jobs = list(zip(use['p'], [cond] * len(use), use['rel']))
    return V.extract_cached3(jobs, 'iso_controls', log=log)


def orig_features(df, src, libv):
    """original features: corpus rows (CCMusic, SoundFont, Otey) and the cached library-note extraction."""
    f = df.set_index('file')
    lf = libv.copy()
    lf['file'] = [os.path.relpath(p, ROOT).replace('\\', '/') for p in lf['path']]
    lf = lf.set_index('file')
    rows = []
    for _, r in src.iterrows():
        rows.append((f if r.src != 'library' else lf).loc[r.file, FEATURE_COLS].values.astype(float))
    out = pd.DataFrame(np.array(rows), columns=FEATURE_COLS)
    out['level_4s_db'] = np.nan
    return out


def run_iso(src, feats, fs):
    """I-only, T2, own-fold: train on CCMusic + SoundFont + Otey rows present, score library notes."""
    d = src.copy().reset_index(drop=True)
    d[FEATURE_COLS] = feats[FEATURE_COLS].values
    d['kind'] = np.where(d.src == 'library', 'sampled_library', np.where(d.y == 0, 'real_isolated', 'algo_single'))
    ok = d[FEATURE_COLS].notna().sum(axis=1) >= 30
    d = d[ok | (d.src == 'library')].reset_index(drop=True)
    train = d[d.src != 'library'].reset_index(drop=True)
    lib = d[d.src == 'library'].reset_index(drop=True)
    res = V.run_cv3(train, V.FS3[fs], 'T1', ext={'lib': lib}, ext_fold={'lib': lib['fold'].values})
    sc = pd.Series(np.r_[res['own'], res['ext_own']['lib']], index=np.r_[train.index.values, len(train) + lib.index.values])
    m5 = np.r_[res['own'], res['ext_mean5']['lib']]
    allr = pd.concat([train, lib], ignore_index=True)
    allr['P'] = sc.values
    allr['P_mean5'] = m5
    return allr


def iso_table(allr, cats, cond):
    y = allr['y'].values
    p = allr['P'].values
    grp = allr['grp'].values
    rm = (allr.src == 'CCMusic').values
    cat = allr['file'].map(cats.set_index('file')[f'cat_{cond}']).fillna('').values if cats is not None else None
    defs = [('SoundFont', allr.src == 'SoundFont'),
            ('SoundFont, unaltered sample', (allr.src == 'SoundFont').values & (cat == 'unaltered sample')),
            ('Otey', allr.src == 'Otey'),
            ('library notes, all layers', allr.src == 'library'),
            ('library notes, mf layer', (allr.src == 'library').values & allr['mf'].fillna(False).values.astype(bool))]
    out = {}
    for name, sm in defs:
        sm = np.asarray(sm)
        out[name] = V.cmp3(y, p, grp, rm, sm) if sm.any() else {'auc': np.nan}
        out[name]['n'] = int(sm.sum())
    out['CCMusic FA'] = float(np.mean(p[rm] > 0.5))
    out['n_ccmusic'] = int(rm.sum())
    return out


# ----------------------------------------------------------------------------- C5 loop index
def loop_index(args):
    path, = args
    import soundfile as sf
    from scipy.signal import fftconvolve
    try:
        x, sr = sf.read(path, always_2d=True)
        x = x.mean(axis=1)
        y = V.cut(x, sr, 4.0)
        o = V.onset_index(y, sr)
        w = y[o + int(0.5 * sr): o + int(4.0 * sr)]
        Ls = int(0.05 * sr)
        lmin, lmax = int(0.02 * sr), int(1.5 * sr)
        best = (0.0, np.nan, np.nan)
        for r0 in np.arange(0.0, 1.95 + 1e-9, 0.1):
            a0 = int(r0 * sr)
            a = w[a0:a0 + Ls]
            if len(a) < Ls:
                break
            a = a - a.mean()
            na = np.sqrt(np.sum(a * a))
            if na < 1e-9:
                continue
            seg = w[a0 + lmin: a0 + lmax + Ls]
            if len(seg) < Ls + 1:
                continue
            dot = fftconvolve(seg, a[::-1], mode='valid')
            cs = np.concatenate([[0.0], np.cumsum(seg)])
            cs2 = np.concatenate([[0.0], np.cumsum(seg * seg)])
            n = len(dot)
            s1 = cs[Ls:Ls + n] - cs[:n]
            s2 = cs2[Ls:Ls + n] - cs2[:n]
            var = s2 - s1 * s1 / Ls
            ok = var > (1e-12 * Ls)
            r = np.where(ok, dot / (na * np.sqrt(np.maximum(var, 1e-30))), 0.0)
            j = int(np.argmax(r))
            if r[j] > best[0]:
                best = (float(r[j]), (lmin + j) / sr, float(r0 + 0.5))
        return {'r_max': best[0], 'lag_s': best[1], 'ref_s': best[2], 'len_s': len(w) / sr, 'error': ''}
    except Exception as e:
        return {'r_max': np.nan, 'lag_s': np.nan, 'ref_s': np.nan, 'len_s': np.nan, 'error': repr(e)[:120]}


def c5(src, keep_real, log=print):
    cp = os.path.join(V.CACHE3, 'c5_loop.csv')
    use = src[(src.src.isin(['SoundFont', 'Otey'])) | keep_real].copy()
    paths = list(use['path_c1b'])
    sigs = [L._sig(p) for p in paths]
    cache = pd.read_csv(cp, dtype={'sig': str}) if os.path.exists(cp) else pd.DataFrame(columns=['path', 'sig'])
    have = set(zip(cache['path'], cache['sig']))
    todo = [(p, s) for p, s in zip(paths, sigs) if (p, s) not in have]
    if todo:
        log(f'  [C5] {len(todo)} new loop-index computations')
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=8) as ex:
            res = list(ex.map(loop_index, [(p,) for p, _ in todo], chunksize=8))
        new = pd.DataFrame([{'path': p, 'sig': s, **r} for (p, s), r in zip(todo, res)])
        cache = pd.concat([cache, new], ignore_index=True).drop_duplicates(['path', 'sig'], keep='last')
        cache.to_csv(cp, index=False)
    ci = cache.set_index(['path', 'sig'])
    t = ci.loc[list(zip(paths, sigs))].reset_index()
    use = use.reset_index(drop=True)
    for c in ('r_max', 'lag_s', 'ref_s', 'len_s'):
        use[c] = pd.to_numeric(t[c], errors='coerce').values
    return use


# ----------------------------------------------------------------------------- C4 phaseogram
def phaseo_one(args):
    path, rel, do_c3 = args
    sys.path.insert(0, os.path.join(ROOT, 'Stattia', 'revision', 'figures'))
    import phaseo_lib as PL
    try:
        x, sr = PL.load_mono(path)
        if do_c3:
            x, _ = V.c3(x, sr, rel)
        d = PL.phaseogram_from_audio(x, sr)
        if not d['ok']:
            return {'ok': 0, 'irreg': np.nan}
        return {'ok': 1, 'irreg': float(PL.drift_summary(d['img'])['irreg'])}
    except Exception as e:
        return {'ok': 0, 'irreg': np.nan, 'err': repr(e)[:100]}


def c4(log=print):
    figd = os.path.join(ROOT, 'Stattia', 'revision', 'figures')
    sys.path.insert(0, figd)
    import drift_regularity as DR
    dr = pd.read_csv(os.path.join(figd, 'drift_notes_all.csv'))
    recs = dr.to_dict('records')
    ok = [r for r in recs if r.get('ok') == 1 and np.isfinite(r.get('irreg', np.nan))]
    for r in ok:
        r['pitch_ok'] = int(r['pitch_ok']) if np.isfinite(r.get('pitch_ok', np.nan)) else 0
    pool_in = DR.dedupe([r for r in ok if DR.MIDI_LO <= r['midi_nom'] <= DR.MIDI_HI and r['pitch_ok']])
    primary = []
    for cls in ('real', 'samplelib', 'soundfont', 'otey'):
        rows = [r for r in pool_in if r['cls'] == cls]
        primary += rows if cls == 'otey' else DR.stratified(rows, DR.N_PER_CLASS, DR.SEED)
    sel = pd.DataFrame(primary)[['cls', 'source', 'file', 'midi_nom', 'irreg']]
    cp = os.path.join(V.CACHE3, 'c4_phaseo.csv')
    if os.path.exists(cp) and len(pd.read_csv(cp)) == 2 * len(sel):
        res = pd.read_csv(cp)
    else:
        log(f'  [C4] phaseograms for {len(sel)} notes x 2')
        jobs = [(os.path.join(ROOT, f), f, c) for c in (False, True) for f in sel['file']]
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=8) as ex:
            out = list(ex.map(phaseo_one, jobs, chunksize=4))
        res = pd.DataFrame([{'file': j[1], 'c3': j[2], **o} for j, o in zip(jobs, out)])
        res.to_csv(cp, index=False)
    base = res[res.c3 == False].set_index('file')['irreg']        # noqa: E712
    aft = res[res.c3 == True].set_index('file')['irreg']          # noqa: E712
    sel['irreg_recomputed'] = sel['file'].map(base).values
    sel['irreg_c3'] = sel['file'].map(aft).values
    return sel


def _mw(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan
    u, p = mannwhitneyu(b, a, alternative='two-sided')
    return float(p), float(u / (len(a) * len(b)))


# ----------------------------------------------------------------------------- main
def main(df, log=print):
    import followup_v2 as F
    key = {}
    libv = F._libv(df)
    src = iso_sources(df, libv)
    cats = sf2_categories(src)
    cats.to_csv(os.path.join(V.OUT3, 'c2_soundfont_categories.csv'), index=False)
    feats = {'orig': orig_features(df, src, libv)}
    for c in CONDS[1:]:
        feats[c] = features(src, c, log)
        log(f'  controls: features {c} ready')
    # C1b real-note filter (still sounding at 4.0 s, measured on the un-convolved original audio)
    lev = feats['c1b']['level_4s_db'].values
    keep_real = (src.src.isin(['CCMusic', 'library']).values) & np.isfinite(lev) & (lev >= -60.0)
    keep = {c: np.ones(len(src), bool) for c in CONDS}
    for c in ('c1b', 'c1b+c3'):
        keep[c] = src.src.isin(['SoundFont', 'Otey']).values | keep_real
    # synthetic notes that are silent at 4.0 s (sample ended) -- reported, kept
    syn_silent = src.src.isin(['SoundFont', 'Otey']).values & ~(np.isfinite(lev) & (lev >= -60.0))
    res, scores = {}, []
    for c in CONDS:
        for fs in ('all47', 'med16'):
            m = keep[c]
            allr = run_iso(src[m].reset_index(drop=True), feats[c][m].reset_index(drop=True), fs)
            res[(c, fs)] = iso_table(allr, cats, c)
            t = allr[['file', 'src', 'inst', 'grp', 'fold', 'P', 'P_mean5']].copy()
            t['cond'], t['featset'] = c, fs
            scores.append(t)
        log(f'  controls: I-only {c} done')
    sc = pd.concat(scores, ignore_index=True)
    sc.to_csv(os.path.join(V.OUT3, 'controls_scores.csv'), index=False, float_format='%.6f')

    md = ['\n## 3. Controls for the isolated-note claim (I-only XGBoost, T2, own-fold scoring)\n\n']
    md.append(__doc__.split('\n\n', 1)[1].split('C2:')[0].replace('\n', ' ') + '\n\n')
    ir_n = len(V._ir_list())
    md.append(f'Room IRs: {ir_n} measured IRs of the MIT McDermott IR Survey (`audio/_ir/mit_survey/Audio/`, '
              'Audio.zip from mcdermottlab.mit.edu/Reverb/IR_Survey.html; the archive holds 270 wav files, the survey '
              'page lists 271), 32 kHz, resampled to each file\'s rate; the convolved note is rescaled to the original '
              'peak; RNG per file = crc32(relative path) (IR choice and noise). '
              f'C1b real-note filter: CCMusic {int(keep_real[src.src.values == "CCMusic"].sum())} of '
              f'{int((src.src == "CCMusic").sum())} and library notes {int(keep_real[src.src.values == "library"].sum())} of '
              f'{int((src.src == "library").sum())} still sound at 4.0 s (' +
              ', '.join(f'{lb} {int(keep_real[(src.inst == lb).values].sum())}/{int((src.inst == lb).sum())}'
                        for lb in ('Iowa', 'K18', 'Salamander', 'FreePats')) +
              f'). Synthetic notes below -60 dB at 4.0 s (sample ended or decayed; kept): SoundFont '
              f'{int(syn_silent[src.src.values == "SoundFont"].sum())} of {int((src.src == "SoundFont").sum())}, Otey '
              f'{int(syn_silent[src.src.values == "Otey"].sum())} of {int((src.src == "Otey").sum())}. SoundFont = '
              f'{int((src.src == "SoundFont").sum())} renders after removing velocity copies.\n\n')
    for fs in ('all47', 'med16'):
        rows = []
        for name in ('SoundFont', 'SoundFont, unaltered sample', 'Otey', 'library notes, all layers', 'library notes, mf layer'):
            cells = [name]
            for c in CONDS:
                r = res[(c, fs)][name]
                cells.append(V.fmt3(r) + f" (n={r.get('n', 0)})" if np.isfinite(r.get('auc', np.nan)) else 'n/a')
                key[f'ctrl:{fs}:{c}:{name}'] = {k: r.get(k) for k in ('auc', 'ci_lo', 'ci_hi', 'det_syn', 'n', 'one_syn_group')}
            rows.append(cells)
        rows.append(['CCMusic false alarms'] + [f"{L.fpct(res[(c, fs)]['CCMusic FA'])} (n={res[(c, fs)]['n_ccmusic']})"
                                                for c in CONDS])
        for c in CONDS:
            key[f'ctrl:{fs}:{c}:CCMusic FA'] = res[(c, fs)]['CCMusic FA']
        md.append(f'**Control table B, {"all 47 features" if fs == "all47" else "16 Group-A medians"}** '
                  '(AUC vs CCMusic notes [95% CI]; det = share detected at P > 0.5):\n\n')
        md.append(L.md_table(['CCMusic notes vs'] + [CLABEL[c] for c in CONDS], rows))
        md.append('\n')

    # per-instrument detection
    md.append('**Detection rate per source and condition** (all 47, share with P > 0.5; CCMusic = false alarms):\n\n')
    s47 = sc[sc.featset == 'all47']
    insts = ['CCMusic'] + sorted(src.loc[src.src == 'SoundFont', 'inst'].unique()) + ['Otey', 'Iowa', 'K18', 'Salamander', 'FreePats']
    rows = []
    for ins in insts:
        cells = [ins]
        for c in CONDS:
            t = s47[(s47.cond == c) & ((s47.inst == ins) | ((ins == 'CCMusic') & (s47.src == 'CCMusic')))]
            cells.append(f"{L.fpct(np.mean(t['P'] > 0.5))} ({len(t)})" if len(t) else 'n/a')
        rows.append(cells)
        key[f'ctrl_det:{ins}'] = cells[1:]
    md.append(L.md_table(['source'] + [CLABEL[c] for c in CONDS], rows))

    # C2 categories
    md.append('\n**C2 -- SoundFont notes by synthesis method** (from the SF2 data: sample root key, active loop mode '
              '(generator 54) and loop start; "transposed" = |key - root| >= 0.5 semitone, "looped" = the loop '
              'start is reached within the analysed window: 1.8 s original / 1.3 s C1a / 4.1 s C1b). All 47, '
              'I-only T2; AUC vs all CCMusic notes; each cell: AUC; det (n notes).\n\n')
    rows = []
    for catn in ('unaltered sample', 'transposed only', 'looped only', 'transposed + looped'):
        cells = [catn]
        for c in ('orig', 'c1a', 'c1b', 'c3', 'c1a+c3', 'c1b+c3'):
            t = s47[s47.cond == c]
            cat = t['file'].map(cats.set_index('file')[f'cat_{c}']).values
            rm = (t.src == 'CCMusic').values
            sm = (t.src == 'SoundFont').values & (cat == catn)
            if sm.sum() == 0:
                cells.append('n/a (0)')
                continue
            r = V.cmp3(np.r_[np.zeros(rm.sum()), np.ones(sm.sum())], np.r_[t['P'].values[rm], t['P'].values[sm]],
                       np.r_[t['grp'].values[rm], t['grp'].values[sm]],
                       np.r_[np.ones(rm.sum(), bool), np.zeros(sm.sum(), bool)],
                       np.r_[np.zeros(rm.sum(), bool), np.ones(sm.sum(), bool)])
            cells.append(f"{r['auc']:.3f}; det {L.fpct(r['det_syn'])} ({int(sm.sum())}, {r['g_syn']} SF)")
            key[f'c2:{c}:{catn}'] = {'auc': r['auc'], 'det': r['det_syn'], 'n': int(sm.sum()), 'n_sf': r['g_syn']}
        rows.append(cells)
    md.append(L.md_table(['category'] + [CLABEL[c] for c in ('orig', 'c1a', 'c1b', 'c3', 'c1a+c3', 'c1b+c3')], rows))
    ct = cats.groupby('inst').agg(n=('file', 'size'), abs_transpose_med=('abs_transpose', 'median'),
                                  looped_mode=('looped', 'mean'), t_loop_med=('t_loop', 'median'),
                                  t_end_med=('t_end', 'median'))
    md.append('\nPer SoundFont (SF2 data for the rendered notes): ' + '; '.join(
        f"{i}: |transp.| median {r.abs_transpose_med:.1f} st, loop mode on {100 * r.looped_mode:.0f}%, loop start median "
        f"{r.t_loop_med:.2f} s, sample end median {r.t_end_med:.2f} s" for i, r in ct.iterrows()) + '.\n')

    # C4
    sel = c4(log)
    sel.to_csv(os.path.join(V.OUT3, 'c4_phaseogram_c3.csv'), index=False, float_format='%.5f')
    md.append('\n**C4 -- phaseogram drift irregularity after C3** (the figure analysis\' primary 190 notes: MIDI 48-76, '
              'pitch check passed, 50 per class stratified by source, all 40 Otey; `figures/phaseo_lib.py` '
              'phaseogram + drift_summary; samples per cycle):\n\n')
    rows = []
    real = sel[sel.cls == 'real']
    for cls in ('real', 'samplelib', 'soundfont', 'otey'):
        g = sel[sel.cls == cls]
        p0, u0 = _mw(real['irreg_recomputed'], g['irreg_recomputed']) if cls != 'real' else (np.nan, np.nan)
        p1, u1 = _mw(real['irreg_c3'], g['irreg_c3']) if cls != 'real' else (np.nan, np.nan)
        rows.append([cls, len(g), f"{g['irreg_recomputed'].median():.3f}", f"{g['irreg_c3'].median():.3f}",
                     '' if cls == 'real' else f'{u0:.2f} (p={p0:.2g})', '' if cls == 'real' else f'{u1:.2f} (p={p1:.2g})'])
        key[f'c4:{cls}'] = {'median_orig': float(g['irreg_recomputed'].median()), 'median_c3': float(g['irreg_c3'].median()),
                            'P_gt_real_orig': u0, 'P_gt_real_c3': u1, 'p_c3': p1}
    md.append(L.md_table(['class', 'notes', 'median irregularity (original)', 'median irregularity (C3)',
                          'P(class > real), original', 'P(class > real), C3'], rows))
    md.append(f"\nRecomputed original irregularity vs `drift_notes_all.csv`: max |difference| = "
              f"{np.nanmax(np.abs(sel['irreg_recomputed'] - sel['irreg'])):.2e}.\n")

    # C5
    lp = c5(src, keep_real, log)
    lp = lp.merge(cats[['file', 'looped', 't_loop', 'loop_len', 't_end', 'abs_transpose']], on='file', how='left')
    lp.to_csv(os.path.join(V.OUT3, 'c5_loop_index.csv'), index=False, float_format='%.6f')
    md.append('\n**C5 -- loop / exact-repetition index on the C1b notes** (window 0.5-4.0 s after onset; max over '
              '50-ms reference segments every 0.1 s of the normalised cross-correlation with the signal at lags '
              '20 ms-1.5 s; each segment mean-removed and normalised, so the slow envelope is removed):\n\n')
    rows = []
    for name, m in [('CCMusic (sounding at 4 s)', lp.src == 'CCMusic')] + \
                   [(f'library {lb} (sounding at 4 s)', lp.inst == lb) for lb in ('Iowa', 'K18', 'Salamander', 'FreePats')] + \
                   [('Otey', lp.src == 'Otey')] + \
                   [(f'SF {i}', (lp.src == 'SoundFont') & (lp.inst == i)) for i in sorted(lp.loc[lp.src == 'SoundFont', 'inst'].unique())]:
        g = lp[m]
        if not len(g):
            continue
        r = g['r_max']
        pred = ''
        if name.startswith('SF'):
            lpred = (g['looped'].fillna(False).astype(bool) & (g['t_loop'] < 4.1)).mean()
            pred = f"{100 * lpred:.0f}%"
            ended = (g['t_end'] < 4.1).mean()
            pred += f" / ended {100 * ended:.0f}%"
        rows.append([name, len(g), f'{r.median():.4f}', f'{r.quantile(0.9):.4f}', f'{100 * (r > 0.999).mean():.0f}%',
                     f'{100 * (r > 0.9999).mean():.0f}%', f"{g.loc[r > 0.999, 'lag_s'].median():.3f}" if (r > 0.999).any() else '-',
                     pred, f"{g['loop_len'].median():.3f}" if name.startswith('SF') and g['looped'].fillna(False).any() else '-'])
        key[f'c5:{name}'] = {'median': float(r.median()), 'share_0999': float((r > 0.999).mean()),
                             'share_09999': float((r > 0.9999).mean())}
    md.append(L.md_table(['source', 'notes', 'median r_max', '90th pct', 'r > 0.999', 'r > 0.9999',
                          'median lag of r > 0.999 (s)', 'SF2: loop reached by 4.1 s / sample ended',
                          'SF2 loop length median (s)'], rows))
    return ''.join(md), key, res
