"""Drawing helpers of Fig. 3 (figures/fig_scores.py): one panel of per-source P(synthetic) on a logit scale --
dots per file, box (quartiles) and median per source, the threshold giving 5 % false alarms on the real class of
the panel (dashed line), class brackets on the right.
"""
import numpy as np
import pandas as pd

BLUE, GREEN, ORANGE, PURPLE = '#2f6fd1', '#2aa876', '#e8693a', '#8e5bc4'
CLASSES = [('real', BLUE), ('sample-based', GREEN), ('algorithmic: sample-bank', PURPLE), ('algorithmic: physical model', ORANGE)]
LO, HI = 1e-6, 1 - 1e-4


def panel(ax, rows, title, rng, stats=True):
    from sklearn.metrics import roc_auc_score
    real = rows[0][1]
    t5 = float(np.quantile(real[np.isfinite(real)], 0.95))     # 5 % false alarms on the real class of the panel
    n = len(rows)
    for i, (lab, p, col, kind) in enumerate(rows):
        yv = n - 1 - i
        p = np.clip(p[np.isfinite(p)], LO, HI)
        ax.scatter(p, yv + rng.uniform(-0.28, 0.28, len(p)), s=9, color=col, alpha=0.35 if len(p) > 300 else 0.6,
                   linewidths=0, zorder=2)
        q1, md, q3 = np.quantile(p, [0.25, 0.5, 0.75])
        ax.add_patch(__import__('matplotlib').patches.Rectangle((q1, yv - 0.33), q3 - q1, 0.66, fill=False, lw=1.6,
                                                                ec='#222', zorder=3))
        ax.plot([md, md], [yv - 0.33, yv + 0.33], color='#111', lw=3, zorder=4)
        if not stats:
            continue
        if kind == 'real':
            txt = f'n={len(p)}  FA 5%'
        else:
            auc = roc_auc_score(np.r_[np.zeros(len(real)), np.ones(len(p))], np.r_[real, p])
            txt = f'n={len(p)}  det {np.mean(p > t5):.0%}  AUC {auc:.2f}'
        ax.text(1.02, yv, txt, transform=ax.get_yaxis_transform(), va='center', fontsize=10.5, color='#555')
    ax.set_yticks(range(n))
    ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=11.5)
    ax.set_xscale('logit')
    ax.set_xlim(LO * 0.7, 1 - 1e-4 * 0.7)
    ticks = [1e-6, 1e-5, 1e-4, 1e-3, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999, 1 - 1e-4]
    ax.set_xticks(ticks)
    ax.set_xticklabels([r'$10^{-6}$', r'$10^{-5}$', r'$10^{-4}$', '0.001', '0.01', '0.1', '0.5', '0.9', '0.99', '0.999',
                        r'1-$10^{-4}$'], fontsize=10.5)
    ax.xaxis.set_minor_locator(__import__('matplotlib').ticker.NullLocator())
    ax.grid(axis='x', color='#ddd', lw=0.8, zorder=0)
    ax.axvline(t5, color='#c0392b', ls='--', lw=1.2, zorder=1)
    ax.text(t5, n - 0.42, f' 5 % FA: P > {t5:.3g}', color='#c0392b', fontsize=10, va='bottom', ha='left',
            bbox=dict(fc='white', ec='none', pad=1.0), zorder=5)
    ax.set_ylim(-0.6, n - 0.4)
    if not stats:                       # class brackets on the right (readable in black and white)
        name = {GREEN: 'sample-based', PURPLE: 'sample-bank', ORANGE: 'physical\nmodel', BLUE: 'real'}
        i = 0
        while i < n:
            j = i
            while j + 1 < n and rows[j + 1][2] == rows[i][2]:
                j += 1
            top, bot = n - 1 - i + 0.38, n - 1 - j - 0.38
            tr = ax.get_yaxis_transform()
            ax.plot([1.015, 1.03, 1.03, 1.015], [top, top, bot, bot], color='#333', lw=1.0, transform=tr, clip_on=False)
            ax.text(1.045, (top + bot) / 2, name[rows[i][2]], transform=tr, va='center', ha='left', fontsize=11)
            i = j + 1
    ax.set_xlabel('P(synthetic), logit scale', fontsize=11.5)
    ax.set_title(title, loc='left', fontsize=12.5)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)

