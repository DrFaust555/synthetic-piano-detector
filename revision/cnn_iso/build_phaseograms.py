"""Phaseogram cache for the format-matched CNN study (one phaseogram per file = one struck note).

Two constructions, both with the detector's own code (detune.segment.build_segments: spectral-flux onset,
-45 dB trim, 30-ms attack skip, inharmonicity-corrected f0 from refine_f0_multipartial; primary segment =
largest energy among segments with 24 < f0 < 4500 Hz, the rule of quality/extract_samplelib_features.py;
quality.phasefeat.compute_phaseogram, P=256, C=48):

  det : the detector unchanged (its pitch search spans MIDI 21-96 with a sub-octave guard).
  con : identical, except that the pitch SEARCH is constrained to +-1 semitone of the file's nominal pitch:
        detune.segment.estimate_pitches is replaced (in this process only, by attribute assignment; no file
        is modified) by the same spectrum + harmonic-sum salience evaluated only at MIDI nominal-1, nominal,
        nominal+1, followed by the same parabolic peak refinement restricted to [f(nom-1), f(nom+1)]; the
        sub-octave guard is dropped (it can only leave the window).  The detector's B estimate and
        refine_f0_multipartial then run unchanged; if the refined f0 leaves [f(nom-1), f(nom+1)] the
        window-constrained peak is kept instead (counted as 'fallback').  Applied identically to every class.

    python Stattia/revision/cnn_iso/build_phaseograms.py [--workers 12] [--force]

Writes cnn_iso/cache/phaseo_{det,con}.npy (N, 48, 256 float32, universe row order) and
cnn_iso/cache/phaseo_meta.csv.  Never touches quality/cache.
"""
from __future__ import annotations

import os
import sys
import time
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import numpy as np                     # noqa: E402
import pandas as pd                    # noqa: E402

import cnn_iso_lib as CL               # noqa: E402

C, P = 48, 256


# --------------------------------------------------------------------------- constrained pitch search
def _constrained_estimate_pitches(nom: int, counter: dict):
    import detune.segment as S
    from numpy.fft import rfft
    f_lo, f_hi = S.midi_to_freq(nom - 1), S.midi_to_freq(nom + 1)

    def est(x_win, sr, max_pitches=1):
        if len(x_win) < 256:
            return []
        nfft = int(np.clip(S._next_pow2(len(x_win)), 1 << 14, 1 << 18))     # as estimate_pitches
        win = np.hanning(len(x_win))
        mag = np.abs(rfft(x_win * win, nfft))
        if mag.max() <= 0:
            return []
        df = sr / nfft
        logmag = np.log1p(mag)
        midis, sal = S._harmonic_salience(logmag, df, sr, midi_lo=nom - 1, midi_hi=nom + 1)
        i = int(np.argmax(sal))
        if sal[i] <= 0:
            return []
        fg = S.midi_to_freq(int(midis[i]))
        tol = 0.03                                                         # as _refine_f0
        lo = max(1, int(max(fg * (1 - tol), f_lo) / df))
        hi = min(len(mag) - 1, int(min(fg * (1 + tol), f_hi) / df) + 1)
        if hi <= lo:
            f0r = fg
        else:
            loc = lo + int(np.argmax(mag[lo:hi]))
            f0r = loc * df
            if 1 <= loc < len(mag) - 1 and mag[loc] > 0:
                a, b, c = mag[loc - 1], mag[loc], mag[loc + 1]
                den = a - 2 * b + c
                f0r = (loc + (0.5 * (a - c) / den if abs(den) > 1e-12 else 0.0)) * df
        f0r = float(min(max(f0r, f_lo), f_hi))
        counter['calls'] += 1
        return [(f0r, float(sal[i]))]
    return est


def _clamped_refine(orig, nom: int, counter: dict):
    import detune.segment as S
    f_lo, f_hi = S.midi_to_freq(nom - 1), S.midi_to_freq(nom + 1)

    def ref(x_win, sr, f0, B, n_partials=8):
        f = orig(x_win, sr, f0, B, n_partials)
        if not (f_lo <= f <= f_hi):
            counter['fallback'] += 1
            return f0
        return f
    return ref


def _primary(x, sr):
    from detune.segment import build_segments
    from quality.phasefeat import compute_phaseogram
    segs, info = build_segments(x, sr)
    ok = [s for s in segs if 24.0 < s.f0 < 4500.0]
    if not ok:
        return None, {'n_segs': len(segs), 'mode': info.get('mode', '')}
    s = max(ok, key=lambda s: s.energy)
    img = compute_phaseogram(s.x, sr, s.f0)
    need = C * sr / s.f0                                   # source samples under the image
    zero_rows = int(np.sum(np.all(img == 0, axis=1)))
    return img, {'n_segs': len(segs), 'mode': info.get('mode', ''), 'f0': float(s.f0), 'B': float(s.B),
                 'seg_len_s': len(s.x) / sr, 'img_span_s': need / sr, 'zero_rows': zero_rows,
                 'onset_s': s.onset / sr}


def one(args):
    path, nom = args
    import detune.segment as S
    out = {'error': ''}
    imgs = {}
    try:
        x, sr = S.load_mono(path)
        out['sr'] = sr
        out['dur_s'] = len(x) / sr
        img, m = _primary(x, sr)                                           # det: detector unchanged
        imgs['det'] = img
        out.update({f'det_{k}': v for k, v in m.items()})
        cnt = {'calls': 0, 'fallback': 0}
        orig_est, orig_ref = S.estimate_pitches, S.refine_f0_multipartial
        try:
            S.estimate_pitches = _constrained_estimate_pitches(int(nom), cnt)
            S.refine_f0_multipartial = _clamped_refine(orig_ref, int(nom), cnt)
            img, m = _primary(x, sr)                                       # con: constrained search
        finally:
            S.estimate_pitches, S.refine_f0_multipartial = orig_est, orig_ref
        imgs['con'] = img
        out.update({f'con_{k}': v for k, v in m.items()})
        out['con_fallback'] = cnt['fallback']
    except Exception as e:                                                 # recorded, never dropped silently
        out['error'] = repr(e)[:200]
    return out, imgs


def row0_one(args):
    """Where the constrained phaseogram's row 0 falls relative to the attack: ms from the first sample with
    |x| > 0.02 * peak (the listening kit's attack rule) to row 0 (located by exact match of the segment's first
    samples in the file, as figures/phaseo_lib.py does)."""
    path, nom = args
    import detune.segment as S
    try:
        x, sr = S.load_mono(path)
        cnt = {'calls': 0, 'fallback': 0}
        orig_est, orig_ref = S.estimate_pitches, S.refine_f0_multipartial
        try:
            S.estimate_pitches = _constrained_estimate_pitches(int(nom), cnt)
            S.refine_f0_multipartial = _clamped_refine(orig_ref, int(nom), cnt)
            segs, _ = S.build_segments(x, sr)
        finally:
            S.estimate_pitches, S.refine_f0_multipartial = orig_est, orig_ref
        ok = [s for s in segs if 24.0 < s.f0 < 4500.0]
        if not ok:
            return np.nan, np.nan
        s = max(ok, key=lambda s: s.energy)
        o, m = int(s.onset), min(64, len(s.x))
        start = None
        for h in np.flatnonzero(x[o:o + int(1.0 * sr)] == s.x[0]) + o:
            if h + m <= len(x) and np.array_equal(x[h:h + m], s.x[:m]):
                start = int(h)
                break
        att = int(np.argmax(np.abs(x) > 0.02 * np.max(np.abs(x))))
        return (1000.0 * (start - att) / sr if start is not None else np.nan), 1000.0 * (o - att) / sr
    except Exception:
        return np.nan, np.nan


def row0_table(meta, workers=4):
    """Cached in cache/row0.csv: row0_ms (row 0 minus attack) and onset_ms (spectral-flux onset minus attack)."""
    p = os.path.join(CL.CACHE, 'row0.csv')
    if os.path.exists(p):
        d = pd.read_csv(p)
        if len(d) == len(meta) and (d['file'].values == meta['file'].values).all():
            return d
    args = list(zip(meta['path'], meta['nominal_midi']))
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(row0_one, args, chunksize=8))
    d = pd.DataFrame({'file': meta['file'], 'row0_ms': [r[0] for r in res], 'onset_ms': [r[1] for r in res]})
    d.to_csv(p, index=False)
    return d


def build(workers=12, force=False, log=print):
    os.makedirs(CL.CACHE, exist_ok=True)
    meta_p = os.path.join(CL.CACHE, 'phaseo_meta.csv')
    u = CL.universe()
    if not force and os.path.exists(meta_p) and all(os.path.exists(os.path.join(CL.CACHE, f'phaseo_{v}.npy'))
                                                    for v in ('det', 'con')):
        old = pd.read_csv(meta_p)
        if len(old) == len(u) and (old['file'].values == u['file'].values).all() and \
                (old['sig'].astype(str).values == u['sig'].astype(str).values).all():
            log(f'phaseogram cache up to date ({len(u)} files) -> {CL.CACHE}')
            return
        log('phaseogram cache stale -> rebuilding')
    t0 = time.time()
    args = list(zip(u['path'], u['nominal_midi']))
    log(f'building phaseograms for {len(u)} files with {workers} workers ...')
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers) as ex:
            res = list(ex.map(one, args, chunksize=4))
    else:
        res = [one(a) for a in args]
    arr = {v: np.zeros((len(u), C, P), np.float32) for v in ('det', 'con')}
    rows = []
    for i, (m, im) in enumerate(res):
        r = {}
        for v in ('det', 'con'):
            ok = im.get(v) is not None
            r[f'{v}_ok'] = ok
            if ok:
                arr[v][i] = im[v]
        rows.append({**m, **r})
    meta = pd.concat([u.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    from detune.segment import freq_to_midi
    for v in ('det', 'con'):
        meta[f'{v}_midi_est'] = [freq_to_midi(f) if np.isfinite(f) else np.nan
                                 for f in meta.get(f'{v}_f0', pd.Series(np.nan, index=meta.index)).astype(float)]
        meta[f'{v}_err_st'] = meta[f'{v}_midi_est'] - meta['nominal_midi']
    for v in ('det', 'con'):
        np.save(os.path.join(CL.CACHE, f'phaseo_{v}.npy'), arr[v])
    meta.to_csv(meta_p, index=False)
    log(f'done in {time.time() - t0:.0f} s; errors: {int((meta["error"].fillna("") != "").sum())}; '
        f'no phaseogram det/con: {int((~meta.det_ok).sum())}/{int((~meta.con_ok).sum())}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args()
    build(a.workers, a.force)
