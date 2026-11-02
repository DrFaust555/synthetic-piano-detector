"""Quantitative test of the Sec. 3.1 claim: "real and sampled recordings drift unevenly;
the physical model drifts regularly".

    python figures/drift_regularity.py

Every phaseogram is built with the detector's own construction (phaseo_lib.py:
build_segments onset + -45 dB trim + 30 ms skip + stiff-string f0, compute_phaseogram
P=256, C=48).  For each row pair (c, c+1) the circular cross-correlation lag
(sub-sample) and the zero-lag row correlation are measured; per note:
  (i)   drift     = mean lag per cycle (samples of P=256; negative = earlier phase)
  (ii)  irreg     = std of the lag increments after removing a linear trend in c
  (iii) r_row     = mean zero-lag row-to-row correlation
Classes: real (CCMusic isolated notes), samplelib (Iowa, Salamander, K18, FreePats),
soundfont (10 SoundFont single-note renders), otey (out_full).

Outputs (results/figures/): drift_notes_all_iso1.csv (every analysed file), drift_summary_iso1.json,
drift_summary_iso1.txt, Fig_drift_irregularity_by_class_iso1.png.  Needs the audio.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import random
import re
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..')))   # repository root
from detector.paths import DATA, RESULTS, AUDIO_ROOT, results_dir   # noqa: E402
OUT = results_dir('figures')
import figures.phaseo_lib as L  # noqa: E402

ROOT = L.ROOT
AUD = os.path.join(ROOT, 'audio')
MIDI_LO, MIDI_HI = 48, 76
N_PER_CLASS = 50
SEED = 0
OUT_CSV = os.path.join(OUT, f'drift_notes_all{L.SUFFIX}.csv')
OUT_JSON = os.path.join(OUT, f'drift_summary{L.SUFFIX}.json')
OUT_TXT = os.path.join(OUT, f'drift_summary{L.SUFFIX}.txt')


# --------------------------------------------------------------------------- candidates
def candidates():
    """(cls, source, path, midi_nominal, dyn) for every isolated-note file."""
    out = []
    # real: CCMusic piano sound-quality dataset (8 pianos), Helmholtz note names
    d = os.path.join(AUD, 'ccmusic_pianos', 'wav')
    names = {'PearlRiver': 'Pearl River upright', 'YoungChang': 'Young Chang upright',
             'Steinway-T': 'Steinway (Grand Theatre)', 'Hsinghai': 'Hsinghai upright',
             'Kawai': 'Kawai upright', 'Steinway': 'Steinway grand',
             'Kawai-G': 'Kawai grand', 'Yamaha': 'Yamaha upright'}
    for fn in sorted(os.listdir(d)):
        m = re.fullmatch(r'(.+?)_([A-Ga-g]\d?s?(?:-[A-Ga-g]\d?b)?)_(train|validation|test)_\d+\.wav', fn)
        if not m:
            continue
        midi = L.helmholtz_to_midi(m.group(2))
        if midi is None:
            continue
        out.append(('real', 'CCMusic ' + names.get(m.group(1), m.group(1)),
                    os.path.join(d, fn), midi, 'na'))
    # sample libraries (raw, as in extract_samplelib_features.LIBS)
    d = os.path.join(AUD, 'ref_top_grand_iowa_steinway')
    for fn in sorted(os.listdir(d)):
        m = re.fullmatch(r'Piano\.(pp|mf|ff)\.([A-G]b?\d)\.aiff', fn)
        if m:
            out.append(('samplelib', 'Iowa Steinway B', os.path.join(d, fn),
                        L.sci_to_midi(m.group(2)), m.group(1)))
    d = os.path.join(AUD, 'ref_bright_yamaha_salamander', 'SalamanderGrandPianoV3_OggVorbis', 'ogg')
    for fn in sorted(os.listdir(d)):
        m = re.fullmatch(r'([A-G]#?\d)v(\d+)\.ogg', fn)
        if m:
            out.append(('samplelib', 'Salamander (Yamaha C5)', os.path.join(d, fn),
                        L.sci_to_midi(m.group(1)), 'v' + m.group(2)))
    d = os.path.join(AUD, 'ref_dull_upright', 'K18')
    for fn in sorted(os.listdir(d)):
        m = re.fullmatch(r'([A-G]#?\d)\.(pp|p|mf|f|ff|sf)\.wav', fn)
        if m:
            out.append(('samplelib', 'K18 upright', os.path.join(d, fn),
                        L.sci_to_midi(m.group(1)), m.group(2)))
    d = os.path.join(AUD, 'ref_bad_old_piano', 'PianoFB SFZ+WAV-20200401', 'samples')
    for fn in sorted(os.listdir(d)):
        m = re.fullmatch(r'([A-G]#?\d)\.wav', fn)
        if m:
            out.append(('samplelib', 'FreePats old piano', os.path.join(d, fn),
                        L.sci_to_midi(m.group(1)), 'single'))
    # SoundFont single-note renders
    for d in sorted(glob.glob(os.path.join(AUD, 'SynthPianoSamples', 'rendered', '*'))):
        inst = os.path.basename(d)
        if inst == '_demos' or not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            m = re.fullmatch(r'([A-G]s?\d)_v(\d+)\.wav', fn)
            if m:
                out.append(('soundfont', inst, os.path.join(d, fn),
                            L.sci_to_midi(m.group(1)), 'v' + m.group(2)))
    # Otey physical model
    d = os.path.join(ROOT, 'otey', 'out_full')
    for fn in sorted(os.listdir(d)):
        m = re.fullmatch(r'([A-G]\d)_v(\d+)\.wav', fn)
        if m:
            out.append(('otey', 'Otey', os.path.join(d, fn),
                        L.sci_to_midi(m.group(1)), 'v' + m.group(2)))
    return out


# --------------------------------------------------------------------------- worker
def _analyse(job):
    cls, src, path, midi, dyn, noise_snr_db, seed = job
    rec = {'cls': cls, 'source': src, 'file': os.path.relpath(path, ROOT).replace('\\', '/'),
           'midi_nom': midi, 'dyn': dyn, 'noise_snr_db': noise_snr_db}
    try:
        x, sr = L.load_mono(path)
        if noise_snr_db is not None:
            clean = L.phaseogram_from_audio(x, sr, midi)
            rng = np.random.default_rng(seed)
            sig = clean['win_rms']
            x = x + rng.standard_normal(len(x)) * sig / (10.0 ** (noise_snr_db / 20.0))
        d = L.phaseogram_from_audio(x, sr, midi)
        if not d['ok']:
            rec['ok'] = 0
            return rec
        nf = L.noise_floor_rms(x, sr)
        rec.update(ok=1, f0=d['f0'], B=d['B'], midi_est=d['midi_est'], mode=d['mode'],
                   n_segs=d['n_segs'], row0_s=d['row0_s'], win_rms=d['win_rms'],
                   snr_db=float(20 * np.log10(d['win_rms'] / nf)) if nf and nf > 0 else np.nan,
                   img_hash=hashlib.md5(np.round(d['img'], 6).tobytes()).hexdigest()[:12])
        s = L.drift_summary(d['img'])                    # primary: lag search |lag| <= 32
        rec.update({k: s[k] for k in ('n_pairs', 'drift', 'drift_slope', 'irreg', 'irreg_mad',
                                      'wander', 'r_row', 'r_aligned', 'jump_frac', 'jitter',
                                      'slow', 'irreg_early', 'irreg_late')})
        sf = L.drift_summary(d['img'], max_lag=None)     # sensitivity: whole circle
        rec.update(drift_full=sf['drift'], irreg_full=sf['irreg'])
    except Exception as e:                               # pragma: no cover
        rec['ok'] = 0
        rec['err'] = repr(e)[:120]
    return rec


# --------------------------------------------------------------------------- statistics
def q(v):
    v = np.asarray([a for a in v if np.isfinite(a)], float)
    if len(v) == 0:
        return {'n': 0}
    return {'n': int(len(v)), 'median': float(np.median(v)),
            'q1': float(np.percentile(v, 25)), 'q3': float(np.percentile(v, 75))}


def mwu(a, b):
    from scipy.stats import mannwhitneyu
    a = np.asarray([v for v in a if np.isfinite(v)], float)
    b = np.asarray([v for v in b if np.isfinite(v)], float)
    r = mannwhitneyu(a, b, alternative='two-sided', method='auto')
    return {'n1': int(len(a)), 'n2': int(len(b)), 'U': float(r.statistic),
            'p': float(r.pvalue), 'AUC_1gt2': float(r.statistic / (len(a) * len(b)))}


def stratified(rows, n, seed):
    """Round-robin over sources (random order within each source), n notes in total."""
    rng = random.Random(seed)
    by = {}
    for r in rows:
        by.setdefault(r['source'], []).append(r)
    for v in by.values():
        rng.shuffle(v)
    keys = sorted(by)
    out, i = [], 0
    while len(out) < n and any(by.values()):
        k = keys[i % len(keys)]
        if by[k]:
            out.append(by[k].pop())
        i += 1
    return out


def dedupe(rows):
    """Drop renders whose phaseogram is identical to another of the same source
    (several SoundFonts ignore velocity, so v64/v100/v127 are the same waveform)."""
    seen, out = set(), []
    for r in rows:
        key = (r['source'], r['img_hash'])
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


METRICS = ['drift', 'abs_drift', 'irreg', 'r_row', 'r_aligned', 'irreg_mad', 'wander',
           'irreg_full', 'drift_slope', 'snr_db', 'jitter', 'slow', 'slow_frac',
           'irreg_early', 'irreg_late', 'abs_slope']


def summarise(sel):
    res = {}
    for cls in ('real', 'samplelib', 'soundfont', 'otey'):
        rr = [r for r in sel if r['cls'] == cls]
        res[cls] = {m: q([r[m] for r in rr]) for m in METRICS}
        res[cls]['n_sources'] = len({r['source'] for r in rr})
    tests = {}
    for m in ('irreg', 'r_row', 'abs_drift', 'irreg_mad', 'irreg_full', 'slow', 'abs_slope'):
        g = {c: [r[m] for r in sel if r['cls'] == c] for c in ('real', 'samplelib', 'soundfont', 'otey')}
        tests[m] = {'real_vs_otey': mwu(g['real'], g['otey']),
                    'real_vs_soundfont': mwu(g['real'], g['soundfont']),
                    'real_vs_samplelib': mwu(g['real'], g['samplelib']),
                    'samplelib_vs_otey': mwu(g['samplelib'], g['otey']),
                    'soundfont_vs_otey': mwu(g['soundfont'], g['otey']),
                    'samplelib_vs_soundfont': mwu(g['samplelib'], g['soundfont'])}
    return {'classes': res, 'mannwhitney': tests}


def per_source(sel, metric='irreg'):
    out = {}
    for r in sel:
        out.setdefault((r['cls'], r['source']), []).append(r[metric])
    return {f'{c}|{s}': {'n': len(v), 'median': float(np.nanmedian(v))}
            for (c, s), v in sorted(out.items())}


# --------------------------------------------------------------------------- main
def main():
    cands = candidates()
    print(f'{len(cands)} candidate files', flush=True)
    jobs = [(c, s, p, m, d, None, 0) for (c, s, p, m, d) in cands]
    with Pool(12) as pool:
        recs = pool.map(_analyse, jobs, chunksize=4)
    ok = [r for r in recs if r.get('ok') == 1 and np.isfinite(r.get('irreg', np.nan))]
    for r in ok:
        r['abs_drift'] = abs(r['drift'])
        r['slow_frac'] = r['slow'] / r['irreg'] if r['irreg'] > 0 else np.nan
        r['abs_slope'] = abs(r['drift_slope'])         # curvature: change of lag per cycle
        r['pitch_ok'] = int(abs(r['midi_est'] - r['midi_nom']) < 1.0)
    print(f'analysed OK: {len(ok)} / {len(recs)}', flush=True)

    # ---- noise control: Otey in-range notes + white noise at the real notes' SNR ----
    real_inrange = [r for r in ok if r['cls'] == 'real' and MIDI_LO <= r['midi_nom'] <= MIDI_HI
                    and r['pitch_ok']]
    snrs = np.array([r['snr_db'] for r in real_inrange if np.isfinite(r['snr_db'])])
    snr_levels = {'median_real': float(np.median(snrs)), 'p10_real': float(np.percentile(snrs, 10))}
    otey_in = [(c, s, p, m, d) for (c, s, p, m, d) in cands
               if c == 'otey' and MIDI_LO <= m <= MIDI_HI]
    njobs = []
    for lab, snr in snr_levels.items():
        for k, (c, s, p, m, d) in enumerate(otey_in):
            njobs.append(('otey+noise:' + lab, s, p, m, d, round(snr, 2), 1000 + k))
    with Pool(12) as pool:
        nrecs = pool.map(_analyse, njobs, chunksize=2)
    noise_ctrl = {}
    for lab in snr_levels:
        v = [r['irreg'] for r in nrecs if r.get('ok') == 1 and r['cls'] == 'otey+noise:' + lab]
        rr = [r['r_row'] for r in nrecs if r.get('ok') == 1 and r['cls'] == 'otey+noise:' + lab]
        noise_ctrl[lab] = {'snr_db': snr_levels[lab], 'irreg': q(v), 'r_row': q(rr)}

    # ---- write every analysed file ----
    cols = ['cls', 'source', 'file', 'midi_nom', 'dyn', 'ok', 'pitch_ok', 'f0', 'midi_est', 'B',
            'mode', 'n_segs', 'row0_s', 'snr_db', 'n_pairs', 'drift', 'abs_drift', 'drift_slope',
            'irreg', 'irreg_mad', 'wander', 'r_row', 'r_aligned', 'jump_frac', 'drift_full',
            'irreg_full', 'jitter', 'slow', 'irreg_early', 'irreg_late', 'img_hash',
            'noise_snr_db']
    import csv
    with open(OUT_CSV, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        for r in recs + nrecs:
            w.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()})

    # ---- designs ----
    def inrange(r):
        return MIDI_LO <= r['midi_nom'] <= MIDI_HI

    pool_in = dedupe([r for r in ok if inrange(r) and r['pitch_ok']])
    primary = []
    for cls in ('real', 'samplelib', 'soundfont', 'otey'):
        rows = [r for r in pool_in if r['cls'] == cls]
        primary += rows if cls == 'otey' else stratified(rows, N_PER_CLASS, SEED)
    designs = {
        'primary_midi48-76_strat50': primary,
        'S1_all_inrange': pool_in,
        'S2_all_registers': dedupe([r for r in ok if r['pitch_ok']]),
        'S3_inrange_incl_pitch_failures': dedupe([r for r in ok if inrange(r)]),
    }
    summary = {'n_candidates': len(cands), 'n_ok': len(ok),
               'n_pitch_fail_all': int(sum(1 - r['pitch_ok'] for r in ok)),
               'n_pitch_fail_by_class': {c: int(sum(1 - r['pitch_ok'] for r in ok if r['cls'] == c))
                                         for c in ('real', 'samplelib', 'soundfont', 'otey')},
               'max_lag_samples': L.MAX_LAG, 'midi_range': [MIDI_LO, MIDI_HI],
               'designs': {}, 'noise_control_otey': noise_ctrl,
               'jump_frac_pairs_full_circle': {}}
    for name, sel in designs.items():
        s = summarise(sel)
        s['per_source_irreg'] = per_source(sel, 'irreg')
        s['files'] = [r['file'] for r in sel] if name.startswith('primary') else None
        # instrument-level test (clustering): source medians, real pianos vs SoundFonts
        ps = s['per_source_irreg']
        s['source_level_real_vs_soundfont'] = mwu(
            [v['median'] for k, v in ps.items() if k.startswith('real|')],
            [v['median'] for k, v in ps.items() if k.startswith('soundfont|')])
        s['source_level_real_vs_samplelib'] = mwu(
            [v['median'] for k, v in ps.items() if k.startswith('real|')],
            [v['median'] for k, v in ps.items() if k.startswith('samplelib|')])
        from scipy.stats import spearmanr
        s['spearman_irreg_vs_snr'] = {}
        for cls in ('real', 'samplelib', 'soundfont', 'otey'):
            rr = [(r['irreg'], r['snr_db']) for r in sel if r['cls'] == cls
                  and np.isfinite(r['snr_db'])]
            if len(rr) > 5:
                rho, pv = spearmanr([a for a, _ in rr], [b for _, b in rr])
                s['spearman_irreg_vs_snr'][cls] = {'rho': float(rho), 'p': float(pv), 'n': len(rr)}
        summary['designs'][name] = s
    # how often a full-circle search jumps away from the +/-32 window
    for cls in ('real', 'samplelib', 'soundfont', 'otey'):
        rr = [r for r in designs['S1_all_inrange'] if r['cls'] == cls]
        summary['jump_frac_pairs_full_circle'][cls] = float(np.mean(
            [abs(r['irreg_full'] - r['irreg']) > 1e-9 for r in rr])) if rr else None
    json.dump(summary, open(OUT_JSON, 'w', encoding='utf-8'), indent=1)

    # ---- readable text ----
    lines = []
    for name, s in summary['designs'].items():
        lines.append(f'=== {name} ===')
        for cls, d in s['classes'].items():
            def f(m):
                x = d[m]
                return (f"{x['median']:.3f} [{x['q1']:.3f}, {x['q3']:.3f}]" if x.get('n') else 'n/a')
            lines.append(f"{cls:10s} n={d['irreg']['n']:4d} sources={d['n_sources']:2d}  "
                         f"drift {f('drift')}  |drift| {f('abs_drift')}  irreg {f('irreg')}  "
                         f"r_row {f('r_row')}  r_al {f('r_aligned')}  irregMAD {f('irreg_mad')}  "
                         f"wander {f('wander')}  SNR {f('snr_db')}  jitter {f('jitter')}  "
                         f"slow {f('slow')}  slow/irreg {f('slow_frac')}  early {f('irreg_early')}  "
                         f"late {f('irreg_late')}  |slope| {f('abs_slope')}")
        for m, t in s['mannwhitney'].items():
            for k, v in t.items():
                lines.append(f"  MWU {m:10s} {k:18s} n={v['n1']}/{v['n2']} U={v['U']:.0f} "
                             f"p={v['p']:.3g} AUC(1>2)={v['AUC_1gt2']:.3f}")
        for key in ('source_level_real_vs_soundfont', 'source_level_real_vs_samplelib'):
            v = s[key]
            lines.append(f"  {key} (irreg source medians): n={v['n1']}/{v['n2']} "
                         f"U={v['U']:.0f} p={v['p']:.3g} AUC={v['AUC_1gt2']:.3f}")
        lines.append('  Spearman irreg vs SNR: ' + '; '.join(
            f"{k}: rho={v['rho']:.2f} p={v['p']:.2g} n={v['n']}"
            for k, v in s['spearman_irreg_vs_snr'].items()))
        lines.append('  per-source median irreg: ' + '; '.join(
            f"{k} ({v['n']}): {v['median']:.3f}" for k, v in s['per_source_irreg'].items()))
    lines.append('=== noise control (Otey MIDI 48-76 + white noise at real-note SNR) ===')
    for lab, d in noise_ctrl.items():
        lines.append(f"{lab}: SNR {d['snr_db']:.1f} dB -> irreg {d['irreg']['median']:.3f} "
                     f"[{d['irreg']['q1']:.3f}, {d['irreg']['q3']:.3f}] n={d['irreg']['n']}; "
                     f"r_row {d['r_row']['median']:.4f}")
    lines.append(f"pitch failures (|midi_est-midi_nom|>=1): {summary['n_pitch_fail_by_class']}")
    lines.append(f"full-circle search changed irreg (fraction of notes): "
                 f"{summary['jump_frac_pairs_full_circle']}")
    open(OUT_TXT, 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    plot_classes(designs['primary_midi48-76_strat50'], noise_ctrl)


def plot_classes(sel, noise_ctrl):
    """Supplementary figure: per-note irregularity and row correlation by class."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 7, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5})
    classes = [('real', 'Real\n(CCMusic)'), ('samplelib', 'Sample\nlibraries'),
               ('soundfont', 'SoundFont\nrenders'), ('otey', 'Physical\nmodel (Otey)')]
    cols = ['#1b6ca8', '#2a9d8f', '#e9a03b', '#c0392b']
    rng = np.random.default_rng(0)
    fig, axs = plt.subplots(1, 2, figsize=(6.9, 2.4), constrained_layout=True)
    for ax, m, lab in [(axs[0], 'irreg', 'irregularity (samples per cycle, of 256)'),
                       (axs[1], 'r_row', '1 - mean row-to-row correlation')]:
        data = []
        for cls, _ in classes:
            v = np.array([r[m] for r in sel if r['cls'] == cls], float)
            data.append(np.clip(1 - v, 1e-5, None) if m == 'r_row' else v)
        ax.boxplot(data, widths=0.5, showfliers=False, medianprops={'color': 'k'})
        for i, v in enumerate(data):
            ax.plot(i + 1 + rng.uniform(-0.18, 0.18, len(v)), v, 'o', ms=2.2, alpha=0.55,
                    color=cols[i], mec='none')
        if m == 'irreg':
            for d in noise_ctrl.values():
                ax.axhline(d['irreg']['median'], color='0.45', lw=0.6, ls='--')
                ax.text(4.42, d['irreg']['median'], f"Otey + noise\nat {d['snr_db']:.0f} dB SNR",
                        fontsize=5, va='center', ha='left', color='0.3')
        ax.set_xticks(range(1, 5))
        ax.set_xticklabels([c[1] for c in classes])
        ax.set_yscale('log')
        ax.set_ylabel(lab)
        ax.set_xlim(0.5, 5.15)
    fig.savefig(os.path.join(OUT, f'Fig_drift_irregularity_by_class{L.SUFFIX}.png'), dpi=300)
    plt.close(fig)


if __name__ == '__main__':
    main()
