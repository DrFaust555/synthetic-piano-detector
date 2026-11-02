"""Exercise retained neural training and prediction entry points on tiny synthetic-image fixtures.

Uses real metadata and cached features, but random images and one epoch: this
checks integration, never estimates detector accuracy or replaces paper scores.
Outputs are isolated in results/validation/training_smoke via SPD_RESULTS.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / 'results/validation/training_smoke'
    os.environ['SPD_RESULTS'] = str(output)
    os.environ['CNN_ISO_THREADS'] = '1'
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[key] = '1'
    sys.path.insert(0, str(root))
    import numpy as np
    import pandas as pd
    import protocol.cnn as C

    rng = np.random.default_rng(149)
    u = C.universe()
    selected = pd.concat([u[u.source == 'ccmusic'].groupby('grp', sort=False).head(2),
                          u[u.source == 'soundfont'].groupby('grp', sort=False).head(2),
                          u[u.source == 'otey'].head(8),
                          u[u.source == 'library'].groupby('grp', sort=False).head(1),
                          u[u.source == 'clip']]).drop_duplicates('file').reset_index(drop=True)
    selected['ok'] = True
    iso = output / 'cnn/cache_iso'
    iso.mkdir(parents=True, exist_ok=True)
    selected.to_csv(iso / 'meta_iso.csv', index=False)
    features = C.features(selected)
    n = len(selected)
    np.save(iso / 'phaseo_iso.npy', rng.normal(size=(n, 48, 256)).astype('float32'))
    np.save(iso / 'phaseo_iso_room_train.npy', rng.normal(size=(n, 3, 48, 256)).astype('float32'))
    np.save(iso / 'phaseo_iso_room_test.npy', rng.normal(size=(n, 48, 256)).astype('float32'))
    np.save(iso / 'feat_room_train.npy', np.repeat(features[:, None], 3, axis=1))
    np.save(iso / 'feat_room_test.npy', features)

    # Assert that the listening split removes every engine note at clip pitches.
    import training.notes_cnn as T
    fold, corpus, usable = T.basis(selected)
    for split, group, sub in [('known_Otey', C.OTEY_G, 'Otey'),
                              ('known_Casio', C.CASIO_G, 'SoundFont (Casio CA-110)')]:
        mask = T.train_mask(selected, fold, corpus, split)
        pitches = selected.loc[(selected.source == 'clip') & (selected['sub'] == sub), 'nominal_midi']
        forbidden = (selected.grp == group) & selected.nominal_midi.isin(pitches)
        assert not mask[forbidden].any()
        assert mask[(selected.source == 'ccmusic')].all()

    perf = output / 'cnn/cache_perf'
    perf.mkdir(parents=True, exist_ok=True)
    original = pd.read_csv(root / 'data/predictions/perf_meta.csv')
    meta = original.groupby(['fold', 'y'], sort=False).head(2).reset_index(drop=True)
    sfz = pd.read_csv(root / 'data/predictions/perf_meta_sfz.csv').groupby('grp', sort=False).head(1)
    meta.to_csv(perf / 'meta_perf.csv', index=False)
    sfz.to_csv(perf / 'meta_sfz.csv', index=False)
    for frame, suffix in ((meta, ''), (sfz, '_sfz')):
        n = len(frame)
        np.save(perf / f'feat47{suffix}.npy', rng.normal(size=(n, 47)).astype('float32'))
        np.save(perf / f'phaseograms{suffix}.npy', rng.normal(size=(n, 3, 48, 256)).astype('float32'))
        np.save(perf / f'phaseogram_mask{suffix}.npy', np.ones((n, 3), dtype='float32'))

    jobs = [('training/notes_cnn.py', ['--epochs', '1', '--workers', '1', '--threads', '1', '--device', 'cpu']),
            ('training/notes_cnn_predictions.py', []),
            ('training/perf_cnn.py', ['--epochs', '1', '--device', 'cpu']),
            ('training/perf_cnn.py', ['--epochs', '1', '--device', 'cpu', '--models', 'fusion_perf', '--drop-iqr',
                                     '--jobs-dir', str(output / 'cnn/jobs_perf_without_iqr')])]
    for rel, args in jobs:
        print('START', rel, ' '.join(args), flush=True)
        with (output / (Path(rel).stem + ('_without_iqr' if '--drop-iqr' in args else '') + '.log')).open('w', encoding='utf-8') as log:
            result = subprocess.run([sys.executable, '-X', 'utf8', '-u', rel, *args], cwd=root,
                                    stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy(), timeout=600)
        if result.returncode:
            raise RuntimeError(f'{rel} failed; see {output}')
        print('PASS', rel, flush=True)
    clips = pd.read_csv(output / 'training/listening_clips.csv')
    assert len(clips) == 30 and np.isfinite(clips.p_fusion).all()
    for jobs_dir in ('cnn/jobs_perf', 'cnn/jobs_perf_without_iqr'):
        for path in (output / jobs_dir).glob('*.npz'):
            assert np.isfinite(np.load(path)['p']).all(), path
    (output / 'summary.json').write_text(json.dumps(dict(passed=True, isolated_files=len(selected),
                                                       performance_files=len(meta), listening_clips=len(clips),
                                                       epochs=1, images='synthetic fixtures'), indent=2), encoding='utf-8')
    print('All retained neural entry points passed (one-epoch integration check)', flush=True)


if __name__ == '__main__':
    main()
