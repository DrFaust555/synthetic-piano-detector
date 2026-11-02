"""Fig. 2: pitch-synchronous phaseograms of C4 from four sources, built exactly as the detector builds them
(see figures/phaseo_lib.py), with one shared symmetric colour scale.  Needs the audio.

    python figures/drift_regularity.py     (first: the class medians that choose the panels)
    python figures/fig_phaseogram.py

Panel rule: for each class, the C4 note whose drift irregularity is closest to that class's median in the primary
sample of drift_regularity.py.  Writes into results/figures/:
    Fig2_phaseogram_revised_iso1.png   300 dpi, single-column width (3.35 in)
    Fig2_phaseogram_revised_iso1.pdf   vector copy
    Fig2_lag_traces_iso1.png           supplementary: per-cycle lag of the four panels
    panel_metrics_iso1.json            f0, B, onset, drift metrics of each panel
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
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..')))   # repository root
from detector.paths import DATA, RESULTS, AUDIO_ROOT, results_dir   # noqa: E402
OUT = results_dir('figures')
import figures.phaseo_lib as L  # noqa: E402

ROOT = L.ROOT
SKIP_MS = 30.0                                       # detector's attack skip (segment.py)
COL_W_IN = 3.35                                      # JAES single column (~85 mm)

C4_NOMINAL = 261.626


def select_panels():
    """Apply the panel rule to the drift results (drift_regularity.py run first):
    per class the C4 candidate whose irregularity is closest to the class median of the primary sample."""
    import csv
    summ = json.load(open(os.path.join(OUT, f'drift_summary{L.SUFFIX}.json'), encoding='utf-8'))
    med = {c: v['irreg']['median'] for c, v in summ['designs']['primary_midi48-76_strat50']['classes'].items()}
    rows = [r for r in csv.DictReader(open(os.path.join(OUT, f'drift_notes_all{L.SUFFIX}.csv'), encoding='utf-8'))
            if r['ok'] == '1' and r['noise_snr_db'] in ('', 'None') and r['irreg'] not in ('', 'nan')
            and int(float(r['midi_nom'])) == 60]
    rule = {'real': lambda r: True,          # libraries: Iowa mf/ff (the drift is visible best there)
            'samplelib': lambda r: r['source'] == 'Iowa Steinway B' and r['dyn'] in ('mf', 'ff'),
            'soundfont': lambda r: r['dyn'] == 'v127',
            'otey': lambda r: True}
    titles = {'real': '(a) Real piano', 'samplelib': '(b) Sample library', 'soundfont': '(c) SoundFont render',
              'otey': '(d) Physical model'}
    out = []
    for cls in ('real', 'samplelib', 'soundfont', 'otey'):
        c = [r for r in rows if r['cls'] == cls and rule[cls](r)]
        r = min(c, key=lambda r: abs(float(r['irreg']) - med[cls]))
        src = r['source'].replace('CCMusic ', 'CCMusic ')
        sub = 'C4'                     # instrument names are not shown in the figure (panel_metrics.json keeps the files)
        out.append((titles[cls], sub, r['file']))
        print(f"{cls}: {r['file']} irreg {float(r['irreg']):.3f} (class median {med[cls]:.3f}, {len(c)} candidates)")
    return out


def main():
    plt.rcParams.update({'font.size': 7, 'axes.titlesize': 7, 'axes.labelsize': 7,
                         'xtick.labelsize': 6, 'ytick.labelsize': 6,
                         'font.family': 'DejaVu Sans'})
    data, metrics = [], {}
    panels = select_panels()
    for title, sub, rel in panels:
        d = L.detector_phaseogram(os.path.join(ROOT, rel), 60)
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
                       interpolation='antialiased', origin='upper',
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
    out_png = os.path.join(OUT, f'Fig2_phaseogram_revised{L.SUFFIX}.png')
    fig.savefig(out_png, dpi=300)
    fig.savefig(os.path.join(OUT, f'Fig2_phaseogram_revised{L.SUFFIX}.pdf'))
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
    fig.savefig(os.path.join(OUT, f'Fig2_lag_traces{L.SUFFIX}.png'), dpi=300)
    plt.close(fig)

    json.dump({'panels': metrics,
               'colour_scale': {'cmap': 'RdBu_r', 'vmin': -vmax, 'vmax': vmax,
                                'shared': True}},
              open(os.path.join(OUT, f'panel_metrics{L.SUFFIX}.json'), 'w', encoding='utf-8'),
              indent=1, default=float)
    from PIL import Image
    w, h = Image.open(out_png).size
    print(f'{out_png}  ({w}x{h} px @300 dpi)')
    for rel, m in metrics.items():
        print(f"{m['panel']:55s} f0={m['f0_Hz']:.2f} ({m['cents_vs_C4']:+.1f} c) B={m['B']:.2e} "
              f"mode={m['mode']} row0={m['row0_s']:.3f}s drift={m['drift']:+.3f} "
              f"irreg={m['irreg']:.3f} r_row={m['r_row']:.4f} max|v|={m['abs_max_value']:.2f} "
              f"p99={m['p99_abs_value']:.2f}")


if __name__ == '__main__':
    main()
