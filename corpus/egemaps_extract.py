"""eGeMAPSv02 functionals (88) for a list of audio files (opensmile 2.6.0); produces data/features/egemaps.csv.

opensmile 2.6 / audinterface 1.4 need pandas >= 2.  If the models run on an older pandas, install opensmile into a
separate folder (`pip install --target <folder> opensmile==2.6.0`) and set EGEMAPS_OPENSMILE_PATH=<folder>: numpy is
then imported first from the system site-packages, and only afterwards is the folder put in front of sys.path.

Audio loading:
  loader 'file' -> smile.process_file(path)                         (corpus notes: CCMusic, SoundFont, Otey)
  loader 'mono' -> x, sr = detector.segment.load_mono(path); smile.process_signal(x, sr)   (sample-library notes)
  loader 'clip' -> the listening-clip transform (44.1 kHz mono, trimmed to 10 ms before the first sample above 2 %
                   of the peak, 3 s, 10-ms fades, RMS 0.1, peak limit 0.97) + smile.process_signal
Smile(feature_set=eGeMAPSv02, feature_level=Functionals), defaults otherwise (channels=0, mixdown=False, no
resampling).  Variant 'sr44k' (sampling_rate=44100, resample=True) is a sensitivity check only.

Cache: results/egemaps/egemaps_<variant>.csv, one row per (file, size:mtime, loader); only missing rows are
extracted.  Errors are recorded in the row, never dropped.

    SPD_AUDIO=<audio root> python corpus/egemaps_extract.py --from-table          (the files of data/features/egemaps.csv)
    SPD_AUDIO=<audio root> python corpus/egemaps_extract.py --jobs <jobs.csv>    (columns path (absolute), loader)
"""
from __future__ import annotations

import os
import sys
import argparse

_OSP = os.environ.get('EGEMAPS_OPENSMILE_PATH', '')
# multiprocessing (spawn) children inherit the parent's sys.path, which already holds _OSP: take it out,
# import the SYSTEM numpy, then put the opensmile folder in front (for pandas >= 2, audinterface, opensmile)
if _OSP:
    _n = os.path.normcase(os.path.abspath(_OSP))
    sys.path[:] = [p for p in sys.path if os.path.normcase(os.path.abspath(p or '.')) != _n]
import numpy as np                      # noqa: E402  system numpy, before the opensmile folder enters sys.path
if _OSP:
    sys.path.insert(0, _OSP)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..'))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
from detector.paths import DATA, AUDIO_ROOT, results_dir   # noqa: E402
ROOT = AUDIO_ROOT              # file names in the cache are relative to the audio root
CACHE = results_dir('egemaps')

_SMILE = {}


def smile_for(variant):
    import opensmile
    if variant not in _SMILE:
        kw = {}
        if variant == 'sr44k':
            kw = dict(sampling_rate=44100, resample=True)
        elif variant != 'main':
            raise ValueError(variant)
        _SMILE[variant] = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02,
                                          feature_level=opensmile.FeatureLevel.Functionals, **kw)
    return _SMILE[variant]


def feature_names():
    return list(smile_for('main').feature_names)


CLIP_SR = 44100


def clip_process(path, dur=3.0):
    """The listening-clip transform: 44.1 kHz mono, onset-trimmed, 3 s, 10-ms fades, RMS-normalised."""
    import librosa
    y, _ = librosa.load(path, sr=CLIP_SR, mono=True)
    pk = np.max(np.abs(y)) + 1e-9
    on = np.argmax(np.abs(y) > 0.02 * pk)        # trim leading silence
    on = max(0, on - int(0.01 * CLIP_SR))
    seg = y[on:on + int(dur * CLIP_SR)]
    if len(seg) < int(dur * CLIP_SR):
        seg = np.pad(seg, (0, int(dur * CLIP_SR) - len(seg)))
    fade = int(0.01 * CLIP_SR)
    seg[:fade] *= np.linspace(0, 1, fade)
    seg[-fade:] *= np.linspace(1, 0, fade)
    rms = np.sqrt(np.mean(seg ** 2)) + 1e-9       # loudness-normalise
    seg = seg * (0.1 / rms)
    p = np.max(np.abs(seg))
    if p > 0.97:
        seg *= 0.97 / p
    return seg.astype(np.float32)


def extract_one(args):
    path, loader, variant = args
    import soundfile as sf
    from detector.segment import load_mono
    out = {'loader': loader, 'variant': variant, 'sr': np.nan, 'channels': np.nan, 'duration_s': np.nan,
           'peak': np.nan, 'n_wrap': np.nan, 'error': ''}
    try:
        smile = smile_for(variant)
        info = sf.info(path)
        out['channels'] = int(info.channels)
        if loader == 'file':
            # what process_file feeds to openSMILE: channel 0 (channels=0, mixdown=False)
            x = sf.read(path, always_2d=True)[0][:, 0]
            sr = int(info.samplerate)
            v = smile.process_file(path).values[0]
        elif loader == 'mono':
            x, sr = load_mono(path)
            if variant == 'sr44k':                  # audresample accepts float32 only
                x = x.astype(np.float32)
            v = smile.process_signal(x, sr).values[0]
        elif loader == 'clip':
            # sensitivity "clip format": the transform that made the listening clips
            # (quality/build_listening_kit.process: librosa load at 44.1 kHz mono, trim to 10 ms before the
            # first sample > 2% of peak, 3.0 s, 10 ms fades, RMS -> 0.1, peak limit 0.97), without the PCM_16 write
            x = clip_process(path)
            sr = CLIP_SR
            v = smile.process_signal(x, sr).values[0]
        else:
            raise ValueError(loader)
        out['sr'] = sr
        out['duration_s'] = len(x) / sr
        out['peak'] = float(np.max(np.abs(x))) if len(x) else np.nan
        # opensmile's wrapper does (x * 32768).astype(int16): |x| >= 1 wraps around (except x == -1)
        out['n_wrap'] = int(np.sum((x >= 1.0) | (x < -1.0)))
        names = feature_names()
        out.update({n: float(val) for n, val in zip(names, v)})
    except Exception as e:                                          # recorded, never silently dropped
        out['error'] = repr(e)[:300]
    return out


def _sig(path):
    st = os.stat(path)
    return f'{st.st_size}:{int(st.st_mtime)}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs', help='csv with columns path (absolute) and loader')
    ap.add_argument('--from-table', action='store_true',
                    help='extract every (file, loader) of data/features/egemaps.csv from the audio root')
    ap.add_argument('--variant', default='main')
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    import pandas as pd
    import opensmile
    if a.from_table:
        t = pd.read_csv(os.path.join(DATA, 'features', 'egemaps.csv'), usecols=['file', 'loader'])
        t = t.drop_duplicates().reset_index(drop=True)
        jobs = pd.DataFrame({'path': [os.path.join(ROOT, f) for f in t['file']], 'loader': t['loader']})
    elif a.jobs:
        jobs = pd.read_csv(a.jobs)
    else:
        raise SystemExit('give --jobs <csv> or --from-table')
    os.makedirs(CACHE, exist_ok=True)
    cpath = os.path.join(CACHE, f'egemaps_{a.variant}.csv')
    rel = [os.path.relpath(p, ROOT).replace('\\', '/') for p in jobs['path']]
    sig = [_sig(p) for p in jobs['path']]
    cache = pd.read_csv(cpath, float_precision='round_trip') if os.path.exists(cpath) else pd.DataFrame()
    have = set()
    if len(cache):
        cache['sig'] = cache['sig'].astype(str)
        ok = cache['error'].fillna('').astype(str) == ''          # rows with an error are retried
        have = set(zip(cache.loc[ok, 'file'], cache.loc[ok, 'sig'], cache.loc[ok, 'loader']))
    todo = [i for i, k in enumerate(zip(rel, sig, jobs['loader'])) if k not in have]
    print(f'  [egemaps {a.variant}] opensmile {opensmile.__version__}, pandas {pd.__version__}, numpy {np.__version__}: '
          f'{len(todo)} new file(s), {len(jobs) - len(todo)} cached', flush=True)
    if not todo:
        return
    args = [(jobs['path'].iloc[i], jobs['loader'].iloc[i], a.variant) for i in todo]
    if a.workers > 1 and len(todo) > 4:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            res = list(ex.map(extract_one, args, chunksize=4))
    else:
        res = [extract_one(x) for x in args]
    names = feature_names()
    cols = ['file', 'sig', 'loader', 'variant', 'sr', 'channels', 'duration_s', 'peak', 'n_wrap', 'error'] + names
    new = pd.DataFrame([{'file': rel[i], 'sig': sig[i], **r} for i, r in zip(todo, res)])
    for c in cols:
        if c not in new:
            new[c] = np.nan
    allrows = pd.concat([cache, new[cols]], ignore_index=True) if len(cache) else new[cols]
    allrows = allrows.drop_duplicates(['file', 'sig', 'loader'], keep='last')
    allrows.to_csv(cpath, index=False)
    nerr = int((new['error'].fillna('') != '').sum())
    print(f'  [egemaps {a.variant}] wrote {len(new)} row(s), {nerr} error(s) -> {cpath}', flush=True)


if __name__ == '__main__':
    main()
