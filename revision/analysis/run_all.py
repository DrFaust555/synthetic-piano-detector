"""Protocol v2 -- ONE command for every number in out/RESULTS_v2.md.

    cd <project root>
    python Stattia/revision/analysis/run_all.py

Needs xgboost 3.2.0 importable (pip install xgboost==3.2.0, or PYTHONPATH=<folder containing it>),
numpy/scikit-learn/scipy/pandas/soundfile (system Python 3.10 here). Reads only; writes only into
Stattia/revision/analysis/out/. Feature extraction of library notes / stimuli is cached in
out/cache/ (delete it to re-extract from audio). data_v2/*.csv are picked up automatically.
"""
from __future__ import annotations

import os
import sys
import time
import json
import platform

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import numpy as np                      # noqa: E402
import pv2_lib as L                     # noqa: E402


def _summary(key, lkey):
    def g(k):
        return key.get(k, {})

    def a(k):
        r = g(k)
        return L.fci(r) if r else 'n/a'

    def d(k):
        r = g(k)
        return L.fpct(r.get('det_syn', np.nan)) if r else 'n/a'
    s = ['## 0. Key numbers (auto-generated from the tables below)\n\n']
    rows = []
    for cid, lbl in (('I1a', 'isolated: CCMusic vs SoundFont renders'), ('I1b', 'isolated: CCMusic vs Otey'),
                     ('I1', 'isolated: CCMusic vs algorithmic (both)'), ('I2', 'isolated: CCMusic vs library takes'),
                     ('P1a', 'performance: real vs SoundFont performance renders'),
                     ('P1b', 'performance: real vs Otey performance renders'),
                     ('P2', 'performance: real vs sampled digital pianos (old + v2)'),
                     ('P2a', 'performance: real vs old digital demos'),
                     ('P2d', 'performance: real vs digital_v2 sample-based'),
                     ('P2d-line', 'performance: real vs digital_v2 sampled, line capture'),
                     ('P3', 'performance: real vs modelled digital'), ('P4', 'performance: real vs Korg M1'),
                     ('C1', 'format-crossed: real performances vs single-note algorithmic'),
                     ('X1', 'pooled: all real vs algorithmic'), ('X2', 'pooled: all real vs sample-based')):
        rows.append([lbl] + [f"{a(f'{r}:{cid}')}; det {d(f'{r}:{cid}')}" for r in
                             ('xgb_all47_T1', 'xgb_all47_T2', 'xgb_med16_T1', 'xgb_med16_T2')])
    for name in ('CCMusic notes vs library notes: all 4 libraries, mf layer',):
        rows.append(['per-note: CCMusic notes vs library notes (mf layer)'] +
                    [f"{a(f'pernote:{r}:{name}')}; det {d(f'pernote:{r}:{name}')}" for r in
                     ('xgb_all47_T1', 'xgb_all47_T2', 'xgb_med16_T1', 'xgb_med16_T2')])
    s.append(L.md_table(['comparison', 'XGB all47 T1', 'XGB all47 T2', 'XGB med16 T1', 'XGB med16 T2'], rows))
    fa = key.get('f0audit:CCMusic (real)')
    if fa:
        s.append(f"\nf0 audit (CCMusic, primary segment / all segments off by >= 1 semitone): MIDI<48 {fa[1]}; "
                 f"48-76 {fa[2]}; >76 {fa[3]}; all {fa[4]}.\n")
        rr = []
        for rn in ('baseline', 'f0-valid & 48-76', 'f0-clean features (retrained)', 'f0-clean features & 48-76'):
            rr.append([rn] + [L.fci(key.get(f'f0sens:xgb_all47_T2:{rn}:{c}', {})) if key.get(f'f0sens:xgb_all47_T2:{rn}:{c}') else 'n/a'
                              for c in ('SoundFont renders', 'Otey notes', 'library notes, mf layer')])
        s.append('\nIsolated regime under f0 / register restrictions (XGB all47, T2; CCMusic vs ...):\n\n')
        s.append(L.md_table(['restriction', 'SoundFont renders', 'Otey notes', 'library notes (mf)'], rr))
    st = key.get('stim:xgb_all47_clip_LOGO')
    if st:
        s.append(f"\nListening stimuli, out-of-sample XGB all47 (LOGO, clips): detected "
                 + ', '.join(f'{c} {v}' for c, v in st['counts'].items())
                 + f"; AUC recorded vs algorithmic {st['auc_rec_vs_alg']:.3f}. ")
    old = key.get('stim:prehard_src_stored')
    if old:
        s.append('Old in-sample model (paper): ' + ', '.join(f'{c} {v}' for c, v in old['counts'].items())
                 + f"; AUC {old['auc_rec_vs_alg']:.3f}.\n")
    lc = {k.split(':', 1)[1]: v for k, v in lkey.items() if k.startswith('cls:')}
    if lc:
        s.append('\nListeners, judged synthetic: ' + ', '.join(
            f"{k} {100 * v['rate']:.0f}% (Wilcoxon p>0.5 = {v['wilcoxon_p_greater']:.2g})" for k, v in lc.items()) + '.\n')
    au = lkey.get('auc:recorded (20) vs algorithmic (10)')
    if au:
        s.append(f"Listener AUC recorded vs algorithmic: mean {au['mean_listener_auc']:.3f} "
                 f"(CI {au['ci'][0]:.3f}-{au['ci'][1]:.3f}), panel {au['panel_auc']:.3f}; mean d' {au['mean_dprime']:.2f}.\n")
    return ''.join(s) + '\n'


def main():
    t0 = time.time()
    os.makedirs(L.OUT, exist_ok=True)
    xv = L.check_xgboost()
    import sklearn
    import scipy
    import pandas
    import protocol_v2
    import listening_v2
    import followup_v2
    import level_test
    import formats_v2
    print('protocol v2: detector analyses ...', flush=True)
    det_md, key, info = protocol_v2.main()
    print('protocol v2: listening analyses ...', flush=True)
    lis_md, lkey = listening_v2.main()
    print('protocol v2: follow-up (ablation, attribution, tuning / channel tests) ...', flush=True)
    fu_md, fkey = followup_v2.main()
    print('protocol v2: level-invariance test ...', flush=True)
    lv_md, lvkey = level_test.main()
    print('protocol v2: format-specific detectors ...', flush=True)
    fm_md, fmkey = formats_v2.main()
    json.dump({'detector': key, 'listening': lkey, 'followup': fkey, 'level': lvkey, 'formats': fmkey}, open(os.path.join(L.OUT, 'key_numbers.json'), 'w'),
              indent=1, default=lambda o: o.item() if isinstance(o, (np.floating, np.integer, np.bool_)) else str(o))

    # checksums (determinism check against the previous run)
    files = ['folds.csv', 'oof_scores.csv', 'comparisons.csv', 'pernote_comparisons.csv', 'pernote_library_scores.csv',
             'oof_scores_isolated_only.csv', 'stimuli_detector_scores.csv', 'listening_per_class.csv',
             'listening_per_listener.csv', 'listening_per_clip.csv', 'f0_audit.csv', 'f0_sensitivity.csv',
             'fold_seed_sensitivity.csv', 'followup_ablation.csv', 'followup_shap.csv',
             'followup_tuning_channel.csv', 'digital_pianos.csv', 'level_test_features.csv', 'level_test_files.csv', 'format_specific.csv']
    ck = os.path.join(L.OUT, 'checksums.txt')
    prev = {}
    if os.path.exists(ck):
        for line in open(ck, encoding='utf-8'):
            if '  ' in line:
                h, f = line.strip().split('  ', 1)
                prev[f] = h
    now = {f: L.md5_of(os.path.join(L.OUT, f)) for f in files if os.path.exists(os.path.join(L.OUT, f))}
    with open(ck, 'w', encoding='utf-8') as fh:
        for f, h in now.items():
            fh.write(f'{h}  {f}\n')
    if prev:
        same = [f for f in now if prev.get(f) == now[f]]
        diff = [f for f in now if f in prev and prev[f] != now[f]]
        det_note = (f'Determinism check vs the previous run (out/checksums.txt): {len(same)} of {len(now)} output CSVs '
                    f'byte-identical' + (f'; changed: {", ".join(diff)}' if diff else '') + '.')
    else:
        det_note = 'Determinism check: no previous out/checksums.txt (first run); run again to compare.'

    head = ['# RESULTS v2 -- clean re-analysis for the JAES revision\n\n',
            f'Generated {time.strftime("%Y-%m-%d %H:%M")} by one command, from the project root:\n\n',
            '```\npython Stattia/revision/analysis/run_all.py\n```\n\n',
            f'Environment: Python {platform.python_version()}, numpy {np.__version__}, pandas {pandas.__version__}, '
            f'scikit-learn {sklearn.__version__}, scipy {scipy.__version__}, xgboost {xv} (n_jobs=1, random_state=0). '
            f'Scripts: `Stattia/revision/analysis/{{run_all, protocol_v2, listening_v2, followup_v2, level_test, formats_v2, pv2_lib}}.py`. '
            f'Total runtime {time.time() - t0:.0f} s. {det_note}\n\n',
            'Outputs (all in `Stattia/revision/analysis/out/`): `folds.csv` (fixed fold per file), `oof_scores.csv` '
            '(per-file scores for every model x feature set x condition), `comparisons.csv` (every AUC row incl. real '
            'false alarms), `pernote_*.csv`, `oof_scores_isolated_only.csv`, `stimuli_detector_scores.csv`, '
            '`listening_*.csv`, `excluded_non_acoustic.csv`, `old_demos_regroup_used.csv`, `f0_audit.csv`, '
            '`f0_sensitivity.csv`, `fold_seed_sensitivity.csv`, `followup_*.csv`, `digital_pianos.csv`, `level_test_*.csv`, `format_specific.csv`, `key_numbers.json`, '
            '`cache/` (extracted features of library notes, corpus single-note files and stimuli).\n\n']
    lv_line = ''
    if lvkey.get('feature_change'):
        lv_line = ('Level test (section 8), median |feature change| / inter-file IQR by group: ' + '; '.join(
            f"{g} " + ', '.join(f'{v} {x:.3f}' for v, x in d.items()) for g, d in lvkey['feature_change'].items()) + '.\n\n')
    text = ''.join(head) + _summary(key, lkey) + lv_line + det_md + lis_md + fu_md + lv_md + fm_md
    with open(os.path.join(L.OUT, 'RESULTS_v2.md'), 'w', encoding='utf-8') as fh:
        fh.write(text)
    print(f'wrote {os.path.join(L.OUT, "RESULTS_v2.md")}  ({time.time() - t0:.0f} s)')


if __name__ == '__main__':
    main()
