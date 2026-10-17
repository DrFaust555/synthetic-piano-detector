"""Revised Fig. 1: detector architecture, corrected against the code
(see Stattia/revision/facts_B_model_features.md, sections A and B.0).

    python Stattia/revision/figures/make_fig1_v2.py            (from the repo root)

Writes into Stattia/revision/figures/ only:
    Fig1_detector_architecture_v2.png    300 dpi, white background, 7.0 in wide (JAES two columns)
    Fig1_detector_architecture_v2.pdf    vector copy

The editable twin of this figure is detector_architecture_v2.drawio (same layout, draw.io
coordinates = 3 x the point coordinates used here).

Changes against the submitted Fig. 1 (Stattia/detector_architecture.drawio):
  * shared front end: onset detection and note windows, first 30 ms skipped (segment.py:285-286);
  * summary branch: no partial tracking and no stiff-string LSQ box; per-note short-time
    spectra at 7 anchors (0.02-1.6 s), levels of partials k = 1..12, f0, B, envelope;
    only the 16 per-note descriptors are pitch-detrended and median/IQR-aggregated (32);
    the register (8) and tuning (7) descriptors are per-file values (15); total x = 47;
  * the summary-feature detector reported in the paper is XGBoost on x (train_realness.py);
  * fusion head is two-layer: linear 128, ReLU, dropout 0.4, linear 1, logistic
    (realness_net.py:100-101), not sigma(w.[e;x]+b); the CNN-only model uses the same head on e;
  * "channel-robust" dropped from the branch title.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                   # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PNG = os.path.join(HERE, 'Fig1_detector_architecture_v2.png')
OUT_PDF = os.path.join(HERE, 'Fig1_detector_architecture_v2.pdf')

# ---------------------------------------------------------------- geometry (points, y down)
W, H = 504.0, 272.0              # 7.0 in x 3.78 in
FS, FS_TITLE, FS_LABEL = 7.0, 8.0, 6.5
LW = 0.7

# columns
C_AUD = (4, 38)
C_ONS = (50, 120)
C_A1 = (50, 140)
C_A2 = (152, 250)
C_B1, C_B2, C_B3 = (50, 108), (121, 179), (192, 250)
C_X = (264, 338)                 # x / concatenate / pooling
C_H = (352, 428)                 # XGBoost / head / CNN-only note
C_O = (442, 500)                 # outputs
# rows
R_TOP = (22, 101)
R_A2A = (22, 62)
R_A2B = (70, 101)
R_MID = (117, 162)
R_BOT = (178, 222)
R_CI = (238, 268)

Y_XGB = 0.5 * (R_TOP[0] + R_TOP[1])
Y_MID = 0.5 * (R_MID[0] + R_MID[1])
Y_BOT = 0.5 * (R_BOT[0] + R_BOT[1])


def mid(c):
    return 0.5 * (c[0] + c[1])


# id: (x0, x1, y0, y1, text, kind). kind: box | round | round_bold | dashed
BOXES = {
    'in':  (*C_AUD, Y_MID - 14, Y_MID + 14, 'Audio', 'box'),
    'on':  (*C_ONS, *R_MID, 'Onset detection,\nnote windows\n(first 30 ms\nskipped)', 'box'),
    'A1':  (*C_A1, *R_TOP, 'Short-time spectra at\n7 times (0.02–1.6 s):\npartial levels\n'
                           '$k$ = 1…12, $f_0$, $B$,\nenvelope', 'box'),
    'A2a': (*C_A2, *R_A2A, '16 per-note descriptors\n→ pitch detrend\n→ median + IQR\n'
                           'over notes (32)', 'box'),
    'A2b': (*C_A2, *R_A2B, 'Register (8) and\ntuning (7) descriptors\nper file', 'box'),
    'x':   (*C_X, *R_TOP, '$x$\n(47 features)', 'box'),
    'xgb': (*C_H, Y_XGB - 11, Y_XGB + 11, 'XGBoost', 'box'),
    'pf':  (*C_O, Y_XGB - 11, Y_XGB + 11, 'P(synthetic)', 'round'),
    'B1':  (*C_B1, *R_BOT, 'Pitch-sync\nresample:\n1 period =\n256 samples', 'box'),
    'B2':  (*C_B2, *R_BOT, '48 periods\n→ phaseogram\n48×256', 'box'),
    'B3':  (*C_B3, *R_BOT, '2-D CNN\n(4 conv blocks,\n114k)\n→ $z_n$ (128)', 'box'),
    'B4':  (*C_X, *R_BOT, 'Pool over notes:\n$e$ = [ mean ; std ]', 'box'),
    'cc':  (*C_X, Y_MID - 17, Y_MID + 17, 'Concatenate\n[ $e$ ; $x$ ]', 'box'),
    'hd':  (*C_H, *R_MID, 'Two-layer head:\nlinear 128, ReLU,\ndropout 0.4,\nlinear 1, logistic', 'box'),
    'out': (*C_O, Y_MID - 17, Y_MID + 17, 'P(synthetic)\n∈ [0, 1]', 'round_bold'),
    'ci':  (175, 295, *R_CI, 'Training: two augmented\nviews + consistency loss', 'dashed'),
}

# free text: (x, y, text, size, weight, style, ha)
TEXTS = {
    'tA':  (4, 10, 'Summary-feature branch', FS_TITLE, 'bold', 'normal', 'left'),
    'tB':  (4, 236, 'Phaseogram branch (pitch-synchronous)', FS_TITLE, 'bold', 'normal', 'left'),
    'lfd': (mid((C_H[0], C_O[1])), Y_XGB - 19, 'feature detector', FS_LABEL, 'normal', 'italic', 'center'),
    'lcnn': (C_H[0], Y_BOT, 'CNN only: same head on $e$', FS_LABEL, 'normal', 'italic', 'left'),
}

# arrows: (id, (x0, y0), (x1, y1), dashed)
X_ON = mid(C_ONS)
ARROWS = [
    ('e_in_on', (C_AUD[1], Y_MID), (C_ONS[0], Y_MID), False),
    ('e_on_A1', (X_ON, R_MID[0]), (X_ON, R_TOP[1]), False),
    ('e_on_B1', (X_ON, R_MID[1]), (X_ON, R_BOT[0]), False),
    ('e_A1_A2a', (C_A1[1], mid(R_A2A)), (C_A2[0], mid(R_A2A)), False),
    ('e_A1_A2b', (C_A1[1], mid(R_A2B)), (C_A2[0], mid(R_A2B)), False),
    ('e_A2a_x', (C_A2[1], mid(R_A2A)), (C_X[0], mid(R_A2A)), False),
    ('e_A2b_x', (C_A2[1], mid(R_A2B)), (C_X[0], mid(R_A2B)), False),
    ('e_x_xgb', (C_X[1], Y_XGB), (C_H[0], Y_XGB), False),
    ('e_xgb_pf', (C_H[1], Y_XGB), (C_O[0], Y_XGB), False),
    ('e_x_cc', (mid(C_X), R_TOP[1]), (mid(C_X), Y_MID - 17), False),
    ('e_B4_cc', (mid(C_X), R_BOT[0]), (mid(C_X), Y_MID + 17), False),
    ('e_cc_hd', (C_X[1], Y_MID), (C_H[0], Y_MID), False),
    ('e_hd_out', (C_H[1], Y_MID), (C_O[0], Y_MID), False),
    ('e_B12', (C_B1[1], Y_BOT), (C_B2[0], Y_BOT), False),
    ('e_B23', (C_B2[1], Y_BOT), (C_B3[0], Y_BOT), False),
    ('e_B34', (C_B3[1], Y_BOT), (C_X[0], Y_BOT), False),
    ('e_ci_B3', (mid(C_B3), R_CI[0]), (mid(C_B3), R_BOT[1]), True),
]

MIN_PAD_X, MIN_PAD_Y = 2.0, 1.5   # pt between text and box edge


def setup_fonts():
    plt.rcParams.update({
        'font.family': ['Arial', 'DejaVu Sans'],
        'mathtext.fontset': 'custom',
        'mathtext.rm': 'Arial',
        'mathtext.it': 'Arial:italic',
        'mathtext.bf': 'Arial:bold',
        'pdf.fonttype': 42,
        'savefig.facecolor': 'white',
    })


def draw():
    setup_fonts()
    fig = plt.figure(figsize=(W / 72.0, H / 72.0), facecolor='white')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_axis_off()

    box_texts = []
    for bid, (x0, x1, y0, y1, txt, kind) in BOXES.items():
        if kind == 'box':
            p = Rectangle((x0, y0), x1 - x0, y1 - y0, fc='white', ec='black', lw=LW)
        elif kind == 'dashed':
            p = Rectangle((x0, y0), x1 - x0, y1 - y0, fc='white', ec='black', lw=LW,
                          ls=(0, (3, 2)))
        else:
            p = FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, fc='white', ec='black', lw=LW,
                               boxstyle='round,pad=0,rounding_size=5')
        ax.add_patch(p)
        t = ax.text(0.5 * (x0 + x1), 0.5 * (y0 + y1), txt, ha='center', va='center',
                    fontsize=FS, linespacing=1.22,
                    fontweight='bold' if kind == 'round_bold' else 'normal',
                    fontstyle='italic' if kind == 'dashed' else 'normal')
        box_texts.append((bid, t, (x0, x1, y0, y1)))

    for _, (x, y, txt, fs, fw, st, ha) in TEXTS.items():
        ax.text(x, y, txt, ha=ha, va='center', fontsize=fs, fontweight=fw, fontstyle=st)

    for _, (xa, ya), (xb, yb), dashed in ARROWS:
        ax.annotate('', xy=(xb, yb), xytext=(xa, ya),
                    arrowprops=dict(arrowstyle='->' if dashed else '-|>', lw=LW, color='black',
                                    ls=(0, (3, 2)) if dashed else '-',
                                    mutation_scale=7, shrinkA=0, shrinkB=0))
    return fig, ax, box_texts


def check_fit(fig, ax, box_texts):
    """Report the smallest gap (pt) between each box's text and its border."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    to_pt = 72.0 / fig.dpi
    ok = True
    for bid, t, (x0, x1, y0, y1) in box_texts:
        bb = t.get_window_extent(r)
        (bx0, by0), (bx1, by1) = ax.transData.transform([(x0, y1), (x1, y0)])
        px = min(bb.x0 - bx0, bx1 - bb.x1) * to_pt
        py = min(bb.y0 - by0, by1 - bb.y1) * to_pt
        flag = '' if (px >= MIN_PAD_X and py >= MIN_PAD_Y) else '   <-- TOO TIGHT'
        ok &= not flag
        print(f'{bid:4s} pad_x {px:5.1f} pt   pad_y {py:5.1f} pt{flag}')
    return ok


def main():
    fig, ax, box_texts = draw()
    ok = check_fit(fig, ax, box_texts)
    fig.savefig(OUT_PNG, dpi=300, facecolor='white')
    fig.savefig(OUT_PDF, facecolor='white')
    plt.close(fig)
    try:                                   # flatten to opaque RGB, exact 300 dpi tag
        from PIL import Image
        Image.open(OUT_PNG).convert('RGB').save(OUT_PNG, dpi=(300, 300))
    except ImportError:
        pass
    print('wrote', OUT_PNG)
    print('wrote', OUT_PDF)
    if not ok:
        print('WARNING: some text is too close to its box border')


if __name__ == '__main__':
    main()
