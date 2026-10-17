"""eGeMAPSv02 functionals (88) for a list of audio files -- worker script of run_egemaps.py.

Runs in its OWN interpreter (started by run_egemaps.py) because opensmile 2.6 / audinterface 1.4
need pandas >= 2, while the protocol-v2 models run on the system pandas 1.5.2.  numpy is imported
FIRST from the system site-packages (1.26.4); only then is the opensmile folder put in front of
sys.path, so a numpy 2.x that `pip install --target` may have placed there is never used.

Audio loading = quality/opensmile_baseline.py:
  loader 'file' -> smile.process_file(path)           (baseline l.59: CCMusic, renders, Otey)
  loader 'mono' -> x, sr = detune.segment.load_mono(path); smile.process_signal(x, sr)
                                                       (baseline l.95/101: sample-library audio)
Smile(feature_set=eGeMAPSv02, feature_level=Functionals), defaults otherwise (channels=0,
mixdown=False, no resampling) -- as in the baseline.  Sensitivities only:
  variant 'sr44k'  -> sampling_rate=44100, resample=True (opensmile's own audresample; float32 input);
  loader  'clip'   -> quality.build_listening_kit.process(path) (the listening-clip transform: 44.1 kHz
                      mono, onset-trimmed, 3 s, RMS-normalised) + smile.process_signal.

Cache: <out>/cache/egemaps_<variant>.csv, one row per (file, size:mtime); only missing rows are
extracted.  Errors are recorded in the row, never dropped.

    python egemaps_extract.py --jobs <jobs.csv> --variant main --workers 8
(jobs.csv: columns path (absolute), loader)
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
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))      # project root
CACHE = os.path.join(HERE, 'out', 'cache')
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

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


def extract_one(args):
    path, loader, variant = args
    import soundfile as sf
    from detune.segment import load_mono
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
            from quality.build_listening_kit import process, SR as CLIP_SR
            x = process(path)
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
    ap.add_argument('--jobs', required=True)
    ap.add_argument('--variant', default='main')
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    import pandas as pd
    import opensmile
    jobs = pd.read_csv(a.jobs)
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
