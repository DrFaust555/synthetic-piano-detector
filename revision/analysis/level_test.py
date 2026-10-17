"""Level-invariance test of the 47-feature pipeline (section 8 of RESULTS_v2.md).

    python Stattia/revision/analysis/level_test.py      (also called by run_all.py; per-file results cached)

Stratified random sample (numpy default_rng(0)): 50 CCMusic notes, 50 real performances (25 manifest +
25 YouTube, <= 150 s), 40 SoundFont single notes (4 per SoundFont), 30 Otey single notes, 20 SoundFont
performance renders (2 per SoundFont) and 10 Otey performance renders.  Every file is re-extracted with the
unchanged pipeline (quality.pipeline.note_raw_descriptors + aggregate_file) after transforming the loaded
mono signal in memory (detune.segment.load_mono is wrapped, so segmentation, per-note descriptors and the
detune analysis all see the transformed audio):
    orig   : as stored (float64 from the file)
    x0.5   : x * 0.5   (float, no requantisation)
    x0.1   : x * 0.1   (float, no requantisation)
    pcm16  : peak-normalised to -1 dBFS and rounded to 16-bit (the real corpus' storage convention)
Scores: the unchanged XGBoost fold models (T2, protocol v2 folds); every file is scored by the fold model
that never saw it.
"""
from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import pv2_lib as L                       # noqa: E402
from pv2_lib import FEATURE_COLS, OUT, ROOT, MED16   # noqa: E402

VARIANTS = ['orig', 'x0.5', 'x0.1', 'pcm16']
CACHE = os.path.join(L.FEAT_CACHE, 'level_test.csv')


def audio_path(f):
    if f.startswith('ccmusic_pianos/') or '_extra/' in f or '/' not in f:
        return os.path.join(ROOT, 'audio', f)
    return os.path.join(ROOT, f)


def transform(x, variant):
    if variant == 'orig':
        return x
    if variant == 'x0.5':
        return x * 0.5
    if variant == 'x0.1':
        return x * 0.1
    if variant == 'pcm16':
        y = x / (np.max(np.abs(x)) + 1e-30) * 10 ** (-1 / 20)
        return np.clip(np.round(y * 32767.0), -32768, 32767) / 32768.0
    raise ValueError(variant)


def level_one(args):
    """features of one file under one level variant (+ segmentation counts)."""
    path, variant = args
    import detune.segment as S
    import quality.pipeline as QP
    orig = S.__dict__.get('_pv2_orig_load') or S.load_mono
    S._pv2_orig_load = orig

    def patched(p):
        x, sr = orig(p)
        return transform(x, variant), sr
    S.load_mono = patched
    QP.load_mono = patched
    out = {'error': ''}
    try:
        x, sr = patched(path)
        out['peak_dbfs'] = float(20 * np.log10(np.max(np.abs(x)) + 1e-30))
        out['n_onsets'] = int(len(S.detect_onsets(x, sr)))
        meta, notes, tuning = QP.note_raw_descriptors(path)
        feats = QP.aggregate_file(notes, tuning, L._trends())
        out.update({'n_notes': meta['n_notes'], 'mode': meta['mode'], **feats})
    except Exception as e:
        out['error'] = repr(e)[:200]
    finally:
        S.load_mono = orig
        QP.load_mono = orig
    return out


def micro_one(path):
    """Per-note descriptor invariance on ONE fixed window: segment the original file, take the loudest
    segment, and recompute calib.features.extract + pipeline._note_desc on that window after scaling it
    (x0.5, x0.1) or requantising it (file-level -1 dBFS PCM-16 gain).  Isolates the per-note maths from the
    segmentation."""
    from detune.segment import load_mono, build_segments
    from calib.features import extract
    from quality.pipeline import _note_desc
    out = {}
    try:
        x, sr = load_mono(path)
        segs = [s for s in build_segments(x, sr)[0] if 24.0 < s.f0 < 4500.0]
        if not segs:
            return {'error': 'no segment'}
        s = max(segs, key=lambda s: s.energy)
        base = _note_desc(extract(s.x, sr, s.f0), s.f0)
        g = 10 ** (-1 / 20) / (np.max(np.abs(x)) + 1e-30)
        for v, w in (('x0.5', s.x * 0.5), ('x0.1', s.x * 0.1),
                     ('pcm16', np.clip(np.round(s.x * g * 32767.0), -32768, 32767) / 32768.0)):
            d = _note_desc(extract(w, sr, s.f0), s.f0)
            ch = {k: abs(d[k] - base[k]) for k in base if np.isfinite(base[k]) and np.isfinite(d[k])}
            nanm = [k for k in base if np.isfinite(base[k]) != np.isfinite(d[k])]
            out[f'micro_maxdiff_{v}'] = float(max(ch.values())) if ch else np.nan
            out[f'micro_nanmismatch_{v}'] = int(len(nanm))
            out[f'micro_changed_{v}'] = ';'.join(sorted([k for k, x in ch.items() if x > 1e-6] + [k + '(NaN)' for k in nanm]))
        out['error'] = ''
    except Exception as e:
        out['error'] = repr(e)[:200]
    return out


def sample(df):
    rng = np.random.default_rng(L.SEED)
    import soundfile as sf
    picks = []

    def take(mask, n, label):
        idx = np.where(mask)[0]
        sel = rng.choice(idx, min(n, len(idx)), replace=False)
        picks.extend((int(i), label) for i in sorted(sel))
    sk = df['subkind'].values
    take(sk == 'ccmusic', 50, 'real: CCMusic note')
    for sub, n in (('manifest', 25), ('youtube_extra', 25)):
        idx = np.where(sk == sub)[0]
        ok = np.array([sf.info(audio_path(df['file'].iloc[i])).duration <= 150 for i in idx])
        take(np.isin(np.arange(len(df)), idx[ok]), n, 'real: performance')
    for g in sorted(set(df.loc[sk == 'soundfont_single', 'grp'])):
        take((sk == 'soundfont_single') & (df['grp'].values == g), 4, 'synth: SoundFont note')
    take(sk == 'otey_single', 30, 'synth: Otey note')
    for g in sorted(set(df.loc[sk == 'sf2_perf', 'grp'])):
        take((sk == 'sf2_perf') & (df['grp'].values == g), 2, 'synth: performance render')
    take(sk == 'otey_perf', 10, 'synth: performance render')
    return pd.DataFrame(picks, columns=['row', 'stratum'])


def run_cached(jobs, fn, keycols, workers=8):
    """jobs: list of dicts with 'file', 'sig', 'variant', 'path' -> cached rows (CSV)."""
    cache = pd.read_csv(CACHE, float_precision='round_trip') if os.path.exists(CACHE) else pd.DataFrame()
    have = set()
    if len(cache):
        cache['sig'] = cache['sig'].astype(str)
        have = set(zip(cache['file'], cache['sig'], cache['variant']))
    todo = [j for j in jobs if (j['file'], j['sig'], j['variant']) not in have]
    if todo:
        print(f'  [level test] {len(todo)} new extraction(s) (cached {len(jobs) - len(todo)})', flush=True)
        from concurrent.futures import ProcessPoolExecutor
        args = [(j['path'], j['variant']) if j['variant'] != 'micro2' else j['path'] for j in todo]
        with ProcessPoolExecutor(max_workers=workers) as ex:
            res = []
            for j, a in zip(todo, args):
                res.append(ex.submit(micro_one if j['variant'] == 'micro2' else level_one, a))
            res = [r.result() for r in res]
        new = pd.DataFrame([{'file': j['file'], 'sig': j['sig'], 'variant': j['variant'], **r} for j, r in zip(todo, res)])
        cache = pd.concat([cache, new], ignore_index=True) if len(cache) else new
        cache = cache.drop_duplicates(['file', 'sig', 'variant'], keep='last')
        os.makedirs(L.FEAT_CACHE, exist_ok=True)
        cache.to_csv(CACHE, index=False)
        cache = pd.read_csv(CACHE, float_precision='round_trip')
        cache['sig'] = cache['sig'].astype(str)
    return cache


def main(log=print):
    import protocol_v2 as P
    import warnings
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    df, _ = P.build_corpus(log=log)
    smp = sample(df)
    smp['file'] = df['file'].values[smp['row']]
    smp['path'] = [audio_path(f) for f in smp['file']]
    smp['sig'] = [L._sig(p) for p in smp['path']]
    jobs = [{'file': f, 'sig': s, 'variant': v, 'path': p} for f, s, p in zip(smp['file'], smp['sig'], smp['path'])
            for v in VARIANTS]
    iso = smp['stratum'].str.contains('note').values
    jobs += [{'file': f, 'sig': s, 'variant': 'micro2', 'path': p}
             for f, s, p in zip(smp.loc[iso, 'file'], smp.loc[iso, 'sig'], smp.loc[iso, 'path'])]
    cache = run_cached(jobs, None, None)
    ci = cache.set_index(['file', 'sig', 'variant'])
    feats = {}
    for v in VARIANTS:
        t = ci.loc[[(f, s, v) for f, s in zip(smp['file'], smp['sig'])]].reset_index()
        t[FEATURE_COLS] = t[FEATURE_COLS].apply(pd.to_numeric, errors='coerce')
        feats[v] = t
    errs = {v: int((feats[v]['error'].fillna('') != '').sum()) for v in VARIANTS}

    # ---- unchanged fold models (T2) score every variant; each file by the model that never saw it
    fold = df['fold'].values[smp['row']]
    scores = {}
    for fs in ('all47', 'med16'):
        _, ext = P.run_cv(df, 'xgb', fs, 'T2', ext={v: feats[v] for v in VARIANTS})
        for v in VARIANTS:
            scores[(fs, v)] = ext[v][fold, np.arange(len(smp))]
    oof = P.run_cv(df, 'xgb', 'all47', 'T2')[0][smp['row'].values]
    repro = float(np.nanmax(np.abs(scores[('all47', 'orig')] - oof)))
    stored = df[FEATURE_COLS].values[smp['row'].values]
    a, b = stored, feats['orig'][FEATURE_COLS].values
    both = np.isfinite(a) & np.isfinite(b)
    feat_repro = float(np.max(np.abs(a[both] - b[both])))

    # ---- per-feature change relative to the inter-file IQR of the corpus
    q = df[FEATURE_COLS].quantile([0.25, 0.75])
    iqr = (q.loc[0.75] - q.loc[0.25])
    scale = iqr.where(iqr > 0, df[FEATURE_COLS].std())
    rows, csv = [], []
    real = smp['stratum'].str.startswith('real').values
    for c in FEATURE_COLS:
        cells = [c]
        for v in VARIANTS[1:]:
            d = np.abs(feats[v][c].values - feats['orig'][c].values) / scale[c]
            nan_flip = np.isfinite(feats[v][c].values) != np.isfinite(feats['orig'][c].values)
            md_all = np.nanmedian(d)
            p90 = np.nanpercentile(d, 90)
            cells.append(f'{md_all:.3f} / {p90:.2f}' + (f' ({int(nan_flip.sum())} NaN flips)' if nan_flip.any() else ''))
            csv.append({'feature': c, 'variant': v, 'median_abs_change_over_iqr': md_all, 'p90': p90,
                        'median_real': np.nanmedian(d[real]), 'median_synth': np.nanmedian(d[~real]),
                        'nan_flips': int(nan_flip.sum())})
        rows.append(cells)
    pd.DataFrame(csv).to_csv(os.path.join(OUT, 'level_test_features.csv'), index=False, float_format='%.5f')
    cdf = pd.DataFrame(csv)

    md = ['\n## 8. Level-invariance test of the 47 features\n\n']
    md.append(__doc__.split('\n\n', 1)[1].replace('\n', ' ').strip() + '\n\n')
    md.append(f'Sample: ' + ', '.join(f'{k} {v}' for k, v in smp['stratum'].value_counts().sort_index().items()) +
              f'. Extraction errors per variant: {errs}. Re-extraction of the unscaled files reproduces the stored '
              f'corpus rows to {feat_repro:.1e} and their out-of-fold scores to {repro:.1e}.\n\n')
    pk = feats['orig'].groupby(smp['stratum'].values)['peak_dbfs'].median()
    md.append('Native peak level of the sampled files (median dBFS): ' + ', '.join(f'{k} {v:.1f}' for k, v in pk.items()) +
              '. (All 120 Otey notes: median -15.9 dBFS, range -32 to -6; real files are stored at -1 dBFS PCM-16.)\n\n')

    # segmentation changes
    seg = []
    for v in VARIANTS[1:]:
        cells = [v]
        for lab in sorted(smp['stratum'].unique()):
            m = (smp['stratum'] == lab).values
            on = feats[v]['n_onsets'].values[m] != feats['orig']['n_onsets'].values[m]
            nn = feats[v]['n_notes'].values[m] != feats['orig']['n_notes'].values[m]
            mo = feats[v]['mode'].values[m] != feats['orig']['mode'].values[m]
            cells.append(f'{int(on.sum())} / {int(nn.sum())} / {int(mo.sum())} of {int(m.sum())}')
        seg.append(cells)
    md.append('**Segmentation changes** (files whose number of detected onsets / number of analysed note segments / '
              'single-vs-piece mode differs from the unscaled file):\n\n')
    md.append(L.md_table(['variant'] + sorted(smp['stratum'].unique()), seg))

    # per-note maths in isolation
    mc = ci.loc[[(f, s, 'micro2') for f, s in zip(smp.loc[iso, 'file'], smp.loc[iso, 'sig'])]].reset_index()
    md.append('\n**Per-note descriptors on a fixed window** (segmentation held fixed; isolated-note files): max |change| '
              'of the 21 per-note descriptors over the files -- ' + ', '.join(
                  f"{v}: {pd.to_numeric(mc[f'micro_maxdiff_{v}'], errors='coerce').max():.2e}"
                  f" ({int(pd.to_numeric(mc[f'micro_nanmismatch_{v}'], errors='coerce').sum())} NaN flips)"
                  for v in ('x0.5', 'x0.1', 'pcm16')) + '. Descriptors changed by requantisation (number of files): ' +
              ', '.join(f'{k_} {n_}' for k_, n_ in pd.Series(
                  [t for x in mc['micro_changed_pcm16'].fillna('') for t in str(x).split(';') if t]).value_counts().items())
              + f' (of {len(mc)} isolated-note files).\n\n')

    # feature table: top changes + summary
    piv = cdf.pivot(index='feature', columns='variant', values='median_abs_change_over_iqr')
    order = piv['x0.1'].sort_values(ascending=False).index
    md.append('**Per feature**: median over files of |change| / inter-file IQR of the feature in the corpus '
              '(std where the IQR is 0), and its 90th percentile; full list in `out/level_test_features.csv`. '
              'Summary over the 47 features: ' + ', '.join(
                  f"{v}: median {piv[v].median():.3f}, max {piv[v].max():.3f}, features with median > 0.1: "
                  f"{int((piv[v] > 0.1).sum())}" for v in VARIANTS[1:]) + '.\n\n')
    rmap = {r[0]: r for r in rows}
    md.append(L.md_table(['feature (sorted by x0.1)', 'x0.5: median / p90', 'x0.1: median / p90', 'pcm16: median / p90'],
                         [rmap[c] for c in order[:20]]))
    grpmed = {}
    from quality.pipeline import BASS_FEATURE_COLS, TUNING
    gdef = {'A medians': MED16, 'A IQRs': [c for c in FEATURE_COLS if c.endswith('_iqr')],
            'B': list(BASS_FEATURE_COLS), 'C': list(TUNING)}
    for g, cols in gdef.items():
        grpmed[g] = {v: float(piv.loc[cols, v].median()) for v in VARIANTS[1:]}
    md.append('\nBy group (median of the per-feature medians): ' + '; '.join(
        f"{g} " + ', '.join(f'{v} {x:.3f}' for v, x in d.items()) for g, d in grpmed.items()) + '.\n\n')

    # score changes
    sc_rows = []
    for fs in ('all47', 'med16'):
        for v in VARIANTS[1:]:
            d = scores[(fs, v)] - scores[(fs, 'orig')]
            flips = (scores[(fs, v)] > 0.5) != (scores[(fs, 'orig')] > 0.5)
            cells = [fs, v, f'{np.median(np.abs(d)):.3f}', f'{int(flips.sum())}/{len(d)}']
            for lab in sorted(smp['stratum'].unique()):
                m = (smp['stratum'] == lab).values
                cells.append(f'{np.mean(d[m]):+.3f} ({int(flips[m].sum())})')
            sc_rows.append(cells)
    md.append('**P(synthetic) under the unchanged T2 fold models**: median |dP|, decision flips at 0.5, and mean dP '
              '(flips) per stratum:\n\n')
    md.append(L.md_table(['features', 'variant', 'median |dP|', 'flips'] + [f'{s}: mean dP (flips)' for s in sorted(smp['stratum'].unique())],
                         sc_rows))

    # AUCs within the sample
    st = smp['stratum'].values
    pairs = [('CCMusic vs SoundFont notes', st == 'real: CCMusic note', st == 'synth: SoundFont note'),
             ('CCMusic vs Otey notes', st == 'real: CCMusic note', st == 'synth: Otey note'),
             ('real performances vs performance renders', st == 'real: performance', st == 'synth: performance render')]
    auc_rows = []
    for fs in ('all47', 'med16'):
        for name, rm, sm in pairs:
            yy = np.r_[np.zeros(rm.sum()), np.ones(sm.sum())]

            def A(vr, vs):
                return L.auc_fast(yy, np.r_[scores[(fs, vr)][rm], scores[(fs, vs)][sm]])
            auc_rows.append([fs, name, f"{A('orig', 'orig'):.3f}"] +
                            [f"{A(v, v):.3f}" for v in VARIANTS[1:]] +
                            [f"{A('orig', v):.3f}" for v in VARIANTS[1:]])
    md.append('\n**Isolated-regime (and performance) AUCs inside the sample**, unchanged T2 fold models. "both v" = both '
              'classes transformed; "real orig / synth v" = only the synthetic files transformed (a level mismatch '
              'between the classes; pcm16 brings the synthetic files to the real corpus\' convention).\n\n')
    md.append(L.md_table(['features', 'comparison', 'orig'] + [f'both {v}' for v in VARIANTS[1:]] +
                         [f'real orig / synth {v}' for v in VARIANTS[1:]], auc_rows))
    md.append('\n' + WHY)
    key = {'feature_change': grpmed, 'auc': auc_rows, 'score': sc_rows, 'repro': (feat_repro, repro)}
    out = smp[['file', 'stratum']].copy()
    for (fs, v), s in scores.items():
        out[f'P_{fs}_{v}'] = s
    for v in VARIANTS:
        out[f'n_onsets_{v}'] = feats[v]['n_onsets'].values
        out[f'n_notes_{v}'] = feats[v]['n_notes'].values
    out.to_csv(os.path.join(OUT, 'level_test_files.csv'), index=False, float_format='%.6f')
    return ''.join(md), key


WHY = """**Why the features change with level (code references).** The per-note descriptors are computed on a
peak-normalised note (`calib/features.py` l.303-304 `x_n = x_t / peak`) with relative thresholds (`_trim_onset`
l.37-50, -50/-60 dB re the note's own peak; envelope dB re peak l.100-101; T60 fit on 0..-40 dB re peak
l.113-117), so they are exactly invariant to a pure gain on a FIXED window (fixed-window test above). The level
dependence enters before that, in the segmentation, which works on un-normalised magnitudes through `log1p`:

* `detune/segment.py` l.73 `m = np.log1p(np.abs(rfft(seg)))` -- the spectral-flux onset detector. log1p(g*|X|) is
  ~linear for |X| << 1 and ~log for |X| >> 1, so a gain changes the relative flux of quiet and loud partials;
  after `flux / flux.max()` the fixed thresholds `height = max(0.08, local + 0.06)` and `prominence=0.04`
  (l.86-88) then admit a different set of onsets.  This decides single-note vs performance mode
  (`n_strikes`, l.354-358) and which note windows exist (l.360-398).
* `detune/segment.py` l.171 `logmag = np.log1p(mag)` and `_harmonic_salience` (l.98-126, mean of log1p peaks over
  6 harmonics) -- the pitch salience; the relative salience of candidates and the chord-tone cut
  `s >= 0.55 * top_sal` (l.297) change with level, so the number and pitch of segments per note changes
  (and with it every IQR feature and the register-trend residuals).
* `quality/pipeline.py` l.139 (`_partial_stats`) `present = prof > -55.0` and `calib/features.py` l.229 `np.max(e) < -55.0`:
  absolute dB thresholds on the un-normalised FFT magnitude of the peak-normalised note (|X| ~ nfft/4 x amplitude,
  so the threshold sits ~110 dB below the note peak).  Invariant to a pure gain, but after 16-bit requantisation
  the quantisation floor (fixed re full scale) lands above it for quiet notes / late frames -- one reason the
  pcm16 variant differs from the float gains.
* The detune analysis (`detune/analyze.py`, `unison.py`, `tuning.py`) uses relative thresholds (0.12 x max,
  0.03 x max band energy) but runs on the same level-dependent segments.
"""


if __name__ == '__main__':
    t, k = main()
    print(t)
