"""Stage 3+4: clean raw excerpts with the project's cleaner, build manifest, extract features.

    python corpus/digital_finalize.py clean      # corpus/clean_demos.py on audio/digital_pianos_v2/raw
                                  # -> raw/clean/*.wav, then moved to audio/digital_pianos_v2/
    python corpus/digital_finalize.py features   # 47 features, as data/corpus/synth_features.csv
    (the manifest is written by corpus/digital_manifest.py)

Features: detector.pipeline.note_raw_descriptors(path) -> aggregate_file(notes, tuning,
trends) with data/corpus/register_trends_cal.json, rounding round(v, 6), NaN -> '' --
exactly as the rows of data/corpus/synth_features.csv (corpus/digital_repro_check.py
recomputes its 14 'yt' demo rows).
"""
from __future__ import annotations
import os, sys, csv, json, glob, shutil, subprocess
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))   # repository root
import corpus.digital_common as common
common.setup()

RAW_OUT = os.path.join(common.AUDIO_OUT, 'raw')
ATTEMPTS = os.path.join(common.SCRIPTS, 'dpv2_attempts.csv')
META_DIR = os.path.join(common.SCRIPTS, 'dpv2_meta')
ENGINES = os.path.join(common.COLLECTION, 'sound_engines.csv')
QC = os.path.join(common.SCRIPTS, 'dpv2_feature_qc.csv')


def rows(p):
    with open(p, encoding='utf-8') as f:
        return list(csv.DictReader(f))


def models():
    return {m['key']: m for m in json.load(open(common.MODELS_JSON, encoding='utf-8'))}


def engines():
    if not os.path.exists(ENGINES):
        return {}
    return {r['model_key']: r for r in rows(ENGINES)}


def accepted(need_raw=True):
    """Latest accepted attempt per (model_key, video_id) whose raw file exists (need_raw=False: the
    features only read the cleaned file, so a mirror without the raw downloads still works)."""
    out = {}
    for r in rows(ATTEMPTS):
        if r['status'] == 'accepted' and r.get('raw_file') and \
                (not need_raw or os.path.exists(os.path.join(common.ROOT, r['raw_file']))):
            out[(r['model_key'], r['video_id'])] = r
    return out


def do_clean():
    subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'digital_run_clean.py'), RAW_OUT],
                   check=True)
    for f in glob.glob(os.path.join(RAW_OUT, 'clean', '*.wav')):
        dst = os.path.join(common.AUDIO_OUT, os.path.basename(f))
        shutil.move(f, dst)
    print('clean files now in', common.AUDIO_OUT)


def sound_method(key):
    e = engines().get(key)
    if e and e.get('sound_method'):
        return e['sound_method']
    return models()[key].get('group', 'sample') + '(unverified)'


def do_features():
    import numpy as np
    from detector.pipeline import note_raw_descriptors, aggregate_file, FEATURE_COLS
    trends = json.load(open(common.TRENDS, encoding='utf-8'))
    M = models()
    acc = accepted(need_raw=False)
    cols = ['file', 'synth_type', 'instrument'] + FEATURE_COLS + ['brand', 'sound_method']
    qc = []
    n = 0
    with open(common.FEATURES, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for (key, vid), r in sorted(acc.items()):
            path = os.path.join(common.AUDIO_OUT, f'{key}_{vid}.wav')
            if not os.path.exists(path):
                qc.append({'model_key': key, 'video_id': vid, 'status': 'no_clean_file'})
                continue
            try:
                meta, notes, tuning = note_raw_descriptors(path)
                feats = aggregate_file(notes, tuning, trends)
            except Exception as e:
                qc.append({'model_key': key, 'video_id': vid, 'status': f'feature_error:{e}'[:200]})
                continue
            rel = os.path.relpath(path, common.ROOT).replace('\\', '/')
            rec = {'file': rel, 'synth_type': 'yt', 'instrument': key, **feats,
                   'brand': M[key]['brand'], 'sound_method': sound_method(key)}
            row = []
            for c in cols:
                v = rec.get(c, '')
                if isinstance(v, float):
                    v = '' if v != v else round(v, 6)
                row.append(v)
            w.writerow(row)
            n += 1
            n_nan = sum(1 for c in FEATURE_COLS if not np.isfinite(feats.get(c, np.nan)))
            qc.append({'model_key': key, 'video_id': vid, 'status': 'ok',
                       'n_notes': meta.get('n_notes'), 'mode': meta.get('mode'),
                       'median_midi': meta.get('median_midi'),
                       'duration_s': round(meta.get('duration_s', 0), 2), 'n_nan_feats': n_nan})
            print(f'  {n:3d} {key} {vid} notes={meta.get("n_notes")} nan={n_nan}', flush=True)
    with open(QC, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['model_key', 'video_id', 'status', 'n_notes', 'mode',
                                          'median_midi', 'duration_s', 'n_nan_feats'])
        w.writeheader()
        for q in qc:
            w.writerow(q)
    print(f'wrote {n} rows -> {common.FEATURES}')


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'clean':
        do_clean()
    elif cmd == 'features':
        do_features()
    else:
        print('the manifest is written by corpus/digital_manifest.py')
