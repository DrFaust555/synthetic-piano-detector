"""Fig. 3: P(synthetic) of every file under the fusion (argument fus) or the phaseogram network (cnn), logit
scale, from the shipped out-of-fold predictions (runs from data/ alone).
(a) isolated notes (libraries: mf layer only, 224 notes): data/predictions/notes_oof.csv (training/notes_cnn.py,
    training/notes_cnn_predictions.py; same notes and folds as the feature detector, room copies in training).
(b) performances: digital instruments in training, the note selection of training/perf_phaseograms.py, no
    augmentation (data/predictions/perf/*.npz, training/perf_cnn.py; SC-55 / GP-3 left out of the rows); Pianoteq
    scored by the model of its fold (data/predictions/pianoteq_cnn_fusion.csv, training/pianoteq_perf_cnn.py).
Threshold line = 5 % false alarms on the real class of the panel; the thresholds are also written to
Fig_scores_<model>_thresholds.txt.
    python figures/fig_scores.py [cnn|fus]    -> results/figures/Fig_scores_<model>.{png,pdf}
"""
import os, sys
import numpy as np
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..')))   # repository root
from detector.paths import DATA, RESULTS, AUDIO_ROOT, results_dir   # noqa: E402
OUT = results_dir('figures')
import figures.scores_lib as M  # panel(), colours
NON_PIANO = r'(?i)toy|vox|talk|phase|magic|res|stick'


def scores(model):
    o = pd.read_csv(os.path.join(DATA, 'predictions', 'notes_oof.csv'))
    col = {'cnn': 'cnn_p', 'fus': 'fus27_p'}[model]
    src = o.source.values
    m = pd.read_csv(os.path.join(DATA, 'predictions', 'notes_meta.csv'))
    mf = o.file.map(dict(zip(m.file, m.mf_layer.fillna(False).astype(bool)))).fillna(False).values.astype(bool)
    a = [('Real notes (CCMusic)', o.loc[(src == 'ccmusic') & o.corpus, col].values, M.BLUE, 'real')]
    for lib in ('Salamander', 'Iowa', 'K18', 'FreePats'):
        a.append((f'{lib}*', o.loc[(src == 'library') & o.usable & mf & (o['sub'] == lib), col].values, M.GREEN, 'syn'))
    a += [('SoundFont renders', o.loc[(src == 'soundfont') & o.corpus, col].values, M.PURPLE, 'syn'),
          ('Otey physical model', o.loc[(src == 'otey') & o.corpus, col].values, M.ORANGE, 'syn')]
    meta = pd.concat([pd.read_csv(os.path.join(DATA, 'predictions', 'perf_meta.csv')),
                      pd.read_csv(os.path.join(DATA, 'predictions', 'perf_meta_sfz.csv'))],
                     ignore_index=True)
    fold = meta.fold.values.astype(int)
    mname = {'cnn': 'cnn_perf', 'fus': 'fusion_perf'}[model]
    p = np.full(len(meta), np.nan)
    for k in range(5):
        p[fold == k] = np.load(os.path.join(DATA, 'predictions', 'perf', f'{mname}_f{k}.npz'))['p'][fold == k]
    k_, sk = meta.kind.values, meta.subkind.values
    toy = meta.file.str.replace('m1_patches/m1_', '', regex=False).str.contains(NON_PIANO).values
    pt = pd.read_csv(os.path.join(DATA, 'predictions', 'pianoteq_cnn_fusion.csv'))[f'p_{mname}'].values
    b = [('Real performances', p[k_ == 'real_performance'], M.BLUE, 'real'),
         ('Digital pianos, sampled', p[k_ == 'sampled_digital_performance'], M.GREEN, 'syn'),
         ('SFZ sampled play*', p[k_ == 'sfz_performance'], M.GREEN, 'syn'),
         ('SoundFont renders', p[sk == 'sf2_perf'], M.PURPLE, 'syn'),
         ('Korg M1, piano presets', p[(k_ == 'm1') & ~toy], M.PURPLE, 'syn'),
         ('Korg M1, non-piano', p[(k_ == 'm1') & toy], M.PURPLE, 'syn'),
         ('Otey renders', p[sk == 'otey_perf'], M.ORANGE, 'syn'),
         ('Digital pianos, modeled', p[k_ == 'modelled_digital'], M.ORANGE, 'syn'),
         ('Pianoteq*', pt, M.ORANGE, 'syn')]
    return a, b


def main(model):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    a, b = scores(model)
    rng = np.random.default_rng(0)
    H = 6.66                                  # figure height, inches; margins fixed in inches, fonts unchanged
    fig, axes = plt.subplots(2, 1, figsize=(10, H), gridspec_kw={'height_ratios': [len(a), len(b)]})
    name = {'cnn': 'network', 'fus': 'fusion'}[model]
    M.panel(axes[0], a, '(a) Isolated notes', rng, stats=False)
    M.panel(axes[1], b, '(b) Performances', rng, stats=False)
    classes = {'real', 'sample-based', 'sample-bank', 'physical' + chr(10) + 'model'}
    for ax in axes:
        ax.set_title(ax.get_title(loc='left'), loc='left', fontsize=10)
        for t in ax.texts:
            if t.get_text() in classes:
                t.set_fontsize(10)
        ax.tick_params(axis='both', labelsize=10)
        ax.xaxis.label.set_fontsize(10)
    top, bottom, gap = 1 - 0.45 / H, 0.6 / H, 1.0
    fig.subplots_adjust(left=0.215, right=0.84, top=top, bottom=bottom,
                        hspace=gap / ((H * (top - bottom) - gap) / 2))
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(OUT, f'Fig_scores_{model}.{ext}'), dpi=200, bbox_inches='tight', pad_inches=0.03)
    t = [float(np.quantile(r[0][1][np.isfinite(r[0][1])], 0.95)) for r in (a, b)]
    open(os.path.join(OUT, f'Fig_scores_{model}_thresholds.txt'), 'w').write(
        f'{name}: 5 % FA thresholds  notes P > {t[0]:.3f}  performances P > {t[1]:.3f}\n')
    print(f'{name}: notes P > {t[0]:.3f}, performances P > {t[1]:.3f}; saved Fig_scores_{model}.png')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'cnn')
