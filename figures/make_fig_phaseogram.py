"""Revised Fig. 2: pitch-synchronous phaseograms of C4 from four sources, built exactly
as the detector builds them (see phaseo_lib.py), with one shared symmetric colour scale.

    python Stattia/revision/figures/make_fig_phaseogram.py            (from the repo root)

Writes into Stattia/revision/figures/ only:
    Fig2_phaseogram_revised.png      300 dpi, JAES single-column width (3.35 in)
    Fig2_phaseogram_revised.pdf      vector copy
    Fig2_lag_traces.png              supplementary: per-cycle lag of the four panels
    panel_metrics.json               f0, B, onset, drift metrics of each panel (+ the
                                     submitted figure's construction, for comparison)
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                       # noqa: E402
from matplotlib.gridspec import GridSpec             # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import phaseo_lib as L                               # noqa: E402

ROOT = L.ROOT
SKIP_MS = 30.0                                       # detector's attack skip (segment.py)
COL_W_IN = 3.35                                      # JAES single column (~85 mm)

# Panel choice rule (fixed before plotting): for each class, the C4 note whose irregularity
# is closest to that class's median in the primary sample of drift_regularity.py
# (real 1.355, sample libraries 0.769, SoundFonts 0.288, Otey 0.054 samples/cycle).
# Candidates: the 8 CCMusic C4 notes; Iowa C4 mf/ff (as requested); the 10 SoundFont C4
# renders; the 4 Otey C4 renders.  Iowa ff won, so SoundFont and Otey use v127 ("ff");
# the CTK-230 render is identical at v64/v100/v127.  ff is also the Iowa layer the
# detector corpus used (extract_samplelib_features._dedup keeps the first sorted file).
PANELS = [
    ('(a) Real piano', 'CCMusic Yamaha upright, C4',
     'audio/ccmusic_pianos/wav/Yamaha_c1_train_190.wav'),
    ('(b) Sample library', 'Iowa Steinway B, C4 ff',
     'audio/ref_top_grand_iowa_steinway/Piano.ff.C4.aiff'),
    ('(c) SoundFont render', 'Casio CTK-230 SF2, C4 v127',
     'audio/SynthPianoSamples/rendered/CTK_230_SoundFont/C4_v127.wav'),
    ('(d) Physical model', 'Otey, C4 v127',
     'otey/out_full/C4_v127.wav'),
]

# Submitted figure (Stattia/make_figures.py::fig1_phaseogram), re-measured for comparison.
SUBMITTED = [
    ('Real acoustic (submitted)', 'oggSamples/C4_v127_p0.ogg'),
    ('Physical model (submitted)', 'otey/out_full/C4_v127.wav'),
    ('"Sample-based digital" (submitted)', 'modelSamples/Iowa_Steinway/C4_v127_p0.wav'),
]
C4_NOMINAL = 261.626


def submitted_construction(path):
    """The submitted figure's construction: 2%-of-peak onset, no 30-ms skip, nominal C4."""
    x, sr = L.load_mono(path)
    pk = np.argmax(np.abs(x) > 0.02 * (np.max(np.abs(x)) + 1e-9))
    return L.compute_phaseogram(np.asarray(x[pk:], float), sr, C4_NOMINAL)


def main():
    plt.rcParams.update({'font.size': 7, 'axes.titlesize': 7, 'axes.labelsize': 7,
                         'xtick.labelsize': 6, 'ytick.labelsize': 6,
                         'font.family': 'DejaVu Sans'})
    data, metrics = [], {}
    for title, sub, rel in PANELS:
        d = L.detector_phaseogram(os.path.join(ROOT, rel))
        assert d['ok'], rel
        s = L.drift_summary(d['img'])
        lags, r0, _ = L.row_pair_lags(d['img'])
        data.append((title, sub, rel, d, lags))
        metrics[rel] = {'panel': title + ' - ' + sub,
                        'f0_Hz': d['f0'], 'cents_vs_C4': 1200 * np.log2(d['f0'] / C4_NOMINAL),
                        'B': d['B'], 'mode': d['mode'], 'n_segments': d['n_segs'],
                        'flux_onset_s': d['onset_flux_s'], 'row0_s': d['row0_s'],
                        'span_ms': 1000 * L.C / d['f0'],
                        'abs_max_value': float(np.max(np.abs(d['img']))),
                        'p99_abs_value': float(np.percentile(np.abs(d['img']), 99)),
                        **s, 'drift_cents_equiv': L.drift_to_cents(s['drift'])}
    # submitted figure's construction, for the report: 2 x 2 of {onset handling} x {f0}
    sub_metrics = {}
    for name, rel in SUBMITTED:
        x, sr = L.load_mono(os.path.join(ROOT, rel))
        pk = int(np.argmax(np.abs(x) > 0.02 * (np.max(np.abs(x)) + 1e-9)))
        d = L.detector_phaseogram(os.path.join(ROOT, rel))
        segs, _ = L.build_segments(x, sr)
        seg = max([q for q in segs if 24.0 < q.f0 < 4500.0], key=lambda q: q.energy)
        combos = {'submitted_onset+nominal_C4 (as submitted)': (x[pk:], C4_NOMINAL),
                  'submitted_onset+estimated_f0': (x[pk:], seg.f0),
                  'detector_onset+nominal_C4': (seg.x, C4_NOMINAL),
                  'detector_onset+estimated_f0 (detector)': (seg.x, seg.f0)}
        res = {}
        for lab, (xx, f0) in combos.items():
            img = L.compute_phaseogram(np.asarray(xx, float), sr, f0)
            sm = L.drift_summary(img)
            res[lab] = {'f0_used_Hz': f0, 'drift': sm['drift'], 'irreg': sm['irreg'],
                        'r_row': sm['r_row'],
                        'harmonic_shift_per_cycle': {k: v['shift_per_cycle'] for k, v in
                                                     L.harmonic_drift(img, 6).items()},
                        'harmonic_power_share': {k: v['power_share'] for k, v in
                                                 L.harmonic_drift(img, 6).items()}}
        sub_metrics[rel] = {'panel': name, 'f0_est_Hz': d['f0'], 'B': d['B'],
                            'cents_vs_C4': 1200 * np.log2(d['f0'] / C4_NOMINAL),
                            'predicted_uniform_shift_from_offset':
                                -L.P * (d['f0'] * np.sqrt(1 + d['B']) / C4_NOMINAL - 1),
                            'submitted_onset_s': pk / sr, 'detector_row0_s': d['row0_s'],
                            'combos': res}

    # ---------------- figure ----------------
    vmax = 1.0                       # every phaseogram is peak-normalised to max|v| = 1
    fig = plt.figure(figsize=(COL_W_IN, 3.9))
    gs = GridSpec(2, 2, figure=fig, left=0.13, right=0.86, top=0.93, bottom=0.205,
                  wspace=0.17, hspace=0.30)
    im = None
    panel_axes = []
    for k, (title, sub, rel, d, _) in enumerate(data):
        r, c = divmod(k, 2)
        ax = fig.add_subplot(gs[r, c])
        panel_axes.append(ax)
        im = ax.imshow(d['img'], cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto',
                       interpolation='nearest', origin='upper',
                       extent=[0, 360, L.C - 0.5, -0.5])
        ax.set_title(f'{title}\n{sub}', fontsize=6.5, pad=2.5, linespacing=1.1)
        ax.set_xticks([0, 90, 180, 270, 360])
        ax.set_yticks([0, 12, 24, 36, 47])
        ax.tick_params(length=2, pad=1.5)
        if c == 0:
            ax.set_ylabel('cycle index', labelpad=1)
        else:
            ax.set_yticklabels([])
        if r == 0:
            ax.set_xticklabels([])
        if c == 1:
            T = 1000.0 / d['f0']     # ms per cycle (own f0); row c starts at 30 + c*T ms

            def fwd(y, T=T):
                return SKIP_MS + (np.asarray(y) + 0.5) * T

            def inv(t, T=T):
                return (np.asarray(t) - SKIP_MS) / T - 0.5
            sec = ax.secondary_yaxis('right', functions=(fwd, inv))
            sec.set_yticks([50, 100, 150, 200])
            sec.tick_params(length=2, pad=1.5, labelsize=6)
            sec.set_ylabel('time after onset (ms)', labelpad=2)
    bl = panel_axes[2].get_position()
    br = panel_axes[3].get_position()
    fig.text(0.5 * (bl.x0 + br.x1), bl.y0 - 0.036, 'phase within the period (deg)',
             ha='center', va='top', fontsize=7)
    cax = fig.add_axes([bl.x0, bl.y0 - 0.112, br.x1 - bl.x0, 0.022])
    cb = fig.colorbar(im, cax=cax, orientation='horizontal', ticks=[-1, -0.5, 0, 0.5, 1])
    cb.set_label('normalized amplitude', labelpad=1)
    cb.ax.tick_params(length=2, pad=1.5, labelsize=6)
    out_png = os.path.join(HERE, 'Fig2_phaseogram_revised.png')
    fig.savefig(out_png, dpi=300)
    fig.savefig(os.path.join(HERE, 'Fig2_phaseogram_revised.pdf'))
    plt.close(fig)

    # ---------------- supplementary: lag traces ----------------
    fig, axs = plt.subplots(1, 2, figsize=(COL_W_IN * 2, 2.0), constrained_layout=True)
    for (title, sub, rel, d, lags), col in zip(data, ['#1b6ca8', '#2a9d8f', '#e9a03b', '#c0392b']):
        cyc = np.arange(1, len(lags) + 1)
        axs[0].plot(cyc, lags, '-o', ms=2, lw=0.8, color=col, label=f'{title[:3]} {sub}')
        X = np.concatenate([[0.0], np.nancumsum(lags)])
        axs[1].plot(np.arange(len(X)), X, '-', lw=1.0, color=col, label=f'{title[:3]} {sub}')
    axs[0].set_xlabel('row pair (c, c+1)')
    axs[0].set_ylabel('lag per cycle (samples of 256)')
    axs[0].axhline(0, color='0.6', lw=0.5)
    axs[1].legend(fontsize=6, frameon=False, loc='lower left')
    axs[1].set_xlabel('cycle index')
    axs[1].set_ylabel('cumulative shift (samples)')
    axs[1].axhline(0, color='0.6', lw=0.5)
    fig.savefig(os.path.join(HERE, 'Fig2_lag_traces.png'), dpi=300)
    plt.close(fig)

    json.dump({'panels': metrics, 'submitted_figure': sub_metrics,
               'colour_scale': {'cmap': 'RdBu_r', 'vmin': -vmax, 'vmax': vmax,
                                'shared': True}},
              open(os.path.join(HERE, 'panel_metrics.json'), 'w', encoding='utf-8'),
              indent=1, default=float)
    from PIL import Image
    w, h = Image.open(out_png).size
    print(f'{out_png}  ({w}x{h} px @300 dpi)')
    for rel, m in metrics.items():
        print(f"{m['panel']:55s} f0={m['f0_Hz']:.2f} ({m['cents_vs_C4']:+.1f} c) B={m['B']:.2e} "
              f"mode={m['mode']} row0={m['row0_s']:.3f}s drift={m['drift']:+.3f} "
              f"irreg={m['irreg']:.3f} r_row={m['r_row']:.4f} max|v|={m['abs_max_value']:.2f} "
              f"p99={m['p99_abs_value']:.2f}")
    for rel, m in sub_metrics.items():
        print(f"{m['panel']}  f0={m['f0_est_Hz']:.3f} ({m['cents_vs_C4']:+.2f} c), predicted "
              f"uniform shift from offset {m['predicted_uniform_shift_from_offset']:+.3f} smp/cycle")
        for lab, v in m['combos'].items():
            hs = ' '.join(f"k{k}:{v['harmonic_shift_per_cycle'][k]:+.2f}"
                          f"({100 * v['harmonic_power_share'][k]:.0f}%)" for k in range(1, 7))
            print(f"    {lab:45s} drift={v['drift']:+.3f} irreg={v['irreg']:.3f} "
                  f"r_row={v['r_row']:.4f}  {hs}")


if __name__ == '__main__':
    main()
