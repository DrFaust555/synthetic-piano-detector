"""Build an SFZ patch for the Iowa Steinway B samples (audio/ref_top_grand_iowa_steinway).

    python corpus/build_iowa_sfz.py      (writes into <SPD_AUDIO>/audio/ref_top_grand_iowa_steinway/)

No sample is modified; everything is done with SFZ opcodes:
  * one region per recorded file, key = its own pitch (no transposition; 88 keys A0-C8);
  * velocity layers split in equal thirds, as in the other libraries (K18, Salamander):
    pp 1-42, mf 43-85, ff 86-127; A0 (ff only) and Bb0 (no mf) give each velocity to the
    recorded layer whose centre is nearest;
  * offset = start of the attack: the first 1-ms frame within 20 dB of the file peak, moved back
    (at most 50 ms) to the last frame at or below max(2 x pre-attack noise, peak - 40 dB), minus 2 ms;
  * loudness: the recordings keep their natural level (median peaks pp -36.8, mf -23.7,
    ff -12.0 dBFS), so velocity must not scale them a second time. Per KEY, loudness follows a
    piecewise-linear dB curve through that key's own layer peaks placed at the layer centres
    (21, 64, 106), extrapolated with the neighbouring slope; each region gets the matching
    amp_velcurve. Level is then continuous across layer boundaries at every key (a curve through
    the layer medians left steps of up to +-6 dB there, which the SFZ build itself would have
    introduced); only timbre switches. A missing layer's level = the key's nearest recorded layer
    + the median gap between the two layers. One global shift: loudest point = 0 dB;
  * ampeg_release = 0.5 s (no release samples exist; same value as K18).
Writes audio/ref_top_grand_iowa_steinway/IowaSteinwayB.sfz and iowa_sfz_regions.csv next to it.
"""
from __future__ import annotations
import os
import re
import csv
import glob

import numpy as np
import soundfile as sf

import sys   # noqa: E402
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))   # repository root
from detector.paths import AUDIO_ROOT   # noqa: E402
ROOT = AUDIO_ROOT
IOWA = os.path.join(ROOT, 'audio', 'ref_top_grand_iowa_steinway')
OUT_SFZ = os.path.join(IOWA, 'IowaSteinwayB.sfz')
OUT_CSV = os.path.join(IOWA, 'iowa_sfz_regions.csv')

LAYERS = {'pp': (1, 42), 'mf': (43, 85), 'ff': (86, 127)}
CENTRE = {'pp': 21, 'mf': 64, 'ff': 106}
RELEASE_S = 0.5
PREROLL_MS = 2
SEMI = {'C': 0, 'Db': 1, 'D': 2, 'Eb': 3, 'E': 4, 'F': 5, 'Gb': 6, 'G': 7, 'Ab': 8, 'A': 9, 'Bb': 10, 'B': 11}


def midi_of(name: str, octave: str) -> int:
    return 12 * (int(octave) + 1) + SEMI[name]


def attack_offset(x: np.ndarray, sr: int) -> tuple[int, float]:
    """(offset in frames, peak dBFS)."""
    a = np.abs(x).max(axis=1)
    pk = float(a.max())
    h = sr // 1000
    env = a[:len(a) // h * h].reshape(-1, h).max(axis=1)
    i20 = int(np.argmax(env > pk * 10 ** (-20 / 20)))
    pre = env[:max(i20 - 50, 1)]
    nf = float(np.percentile(pre, 95)) if len(pre) > 5 else 0.0
    thr = max(2 * nf, pk * 10 ** (-40 / 20))
    j = i20
    while j > 0 and i20 - j < 50 and env[j - 1] > thr:
        j -= 1
    return max((j - PREROLL_MS) * h, 0), 20 * np.log10(pk)


def level_curve(layer_db: dict) -> callable:
    """Target loudness (dB) vs velocity: piecewise-linear through (layer centre, layer level)."""
    xs = [CENTRE[k] for k in ('pp', 'mf', 'ff')]
    ys = [layer_db[k] for k in ('pp', 'mf', 'ff')]

    def T(v):
        if v <= xs[1]:
            return ys[0] + (v - xs[0]) * (ys[1] - ys[0]) / (xs[1] - xs[0])
        return ys[1] + (v - xs[1]) * (ys[2] - ys[1]) / (xs[2] - xs[1])
    return T


def main():
    regs = []
    for f in sorted(glob.glob(os.path.join(IOWA, 'Piano.*.aiff'))):
        dyn, nm, octv = re.match(r'Piano\.(pp|mf|ff)\.([A-G]b?)(\d)\.aiff', os.path.basename(f)).groups()
        x, sr = sf.read(f, always_2d=True)
        assert sr == 44100
        off, pk_db = attack_offset(x, sr)
        regs.append(dict(file=os.path.basename(f), dyn=dyn, key=midi_of(nm, octv), offset=off,
                         offset_s=round(off / sr, 4), peak_dbfs=round(pk_db, 2), dur_s=round(len(x) / sr, 2)))
    keys = sorted({r['key'] for r in regs})
    assert keys == list(range(21, 109)), keys
    layer_db = {d: float(np.median([r['peak_dbfs'] for r in regs if r['dyn'] == d])) for d in LAYERS}
    T = level_curve(layer_db)              # median curve, for the report only
    key_db, Tk = {}, {}
    for k in keys:
        own = {r['dyn']: r['peak_dbfs'] for r in regs if r['key'] == k}
        for d in LAYERS:
            if d not in own:
                n = min(own, key=lambda q: abs(CENTRE[q] - CENTRE[d]))
                own[d] = own[n] + layer_db[d] - layer_db[n]
        key_db[k] = own
        Tk[k] = level_curve(own)
    # velocity -> layer; a key missing a layer (A0: ff only, Bb0: no mf) gives each velocity to
    # the recorded layer whose centre is nearest
    nominal = {v: d for d, (lo, hi) in LAYERS.items() for v in range(lo, hi + 1)}
    vel_ranges = {}                        # (key, dyn) -> (lovel, hivel)
    for k in keys:
        have = [r['dyn'] for r in regs if r['key'] == k]
        owner = {v: (nominal[v] if nominal[v] in have else min(have, key=lambda d: abs(CENTRE[d] - v)))
                 for v in range(1, 128)}
        for d in have:
            vs = [v for v in range(1, 128) if owner[v] == d]
            assert vs == list(range(vs[0], vs[-1] + 1))
            vel_ranges[(k, d)] = (vs[0], vs[-1])
    # region gain (dB) = target loudness - its layer's median level; one global shift -> loudest = 0 dB
    gshift = max(Tk[k](v) - key_db[k][d] for (k, d), (lo, hi) in vel_ranges.items() for v in (lo, hi))

    lines = ['// Iowa EMS Steinway & Sons Model B (2001), built by build_iowa_sfz.py - see its docstring.',
             '// Samples: University of Iowa Electronic Music Studios, free for any use.',
             '// layer median peaks dBFS: ' + ', '.join(f'{d} {layer_db[d]:.1f}' for d in LAYERS),
             f'// loudness per key through its own layer peaks; median: vel 1 {T(1) - T(127):.1f} dB ... vel 127 0 dB', '',
             f'<group> amp_veltrack=100 ampeg_release={RELEASE_S}']
    for r in sorted(regs, key=lambda r: (r['key'], CENTRE[r['dyn']])):
        lo, hi = vel_ranges[(r['key'], r['dyn'])]
        pts = sorted({lo, hi, *range(((lo + 7) // 8) * 8, hi, 8)})
        k, d = r['key'], r['dyn']
        curve = ' '.join(f'amp_velcurve_{v}={10 ** ((Tk[k](v) - key_db[k][d] - gshift) / 20):.5f}' for v in pts)
        r['lovel'], r['hivel'] = lo, hi
        lines.append(f'<region> sample={r["file"]} key={r["key"]} pitch_keycenter={r["key"]} '
                     f'lovel={lo} hivel={hi} offset={r["offset"]} {curve}')
    with open(OUT_SFZ, 'w', encoding='ascii', newline='\n') as fh:
        fh.write('\n'.join(lines) + '\n')
    with open(OUT_CSV, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(regs[0].keys()))
        w.writeheader()
        w.writerows(sorted(regs, key=lambda r: (r['key'], CENTRE[r['dyn']])))
    offs = np.array([r['offset_s'] for r in regs])
    print(f'{len(regs)} regions, keys {keys[0]}-{keys[-1]}; incomplete keys: '
          + ', '.join(f'{k}: ' + '/'.join(f'{d} {lo}-{hi}' for (kk, d), (lo, hi) in vel_ranges.items() if kk == k)
                      for k in keys if sum(kk == k for kk, _ in vel_ranges) < 3))
    print('layer median peaks', {d: round(v, 1) for d, v in layer_db.items()})
    print(f'offset s: median {np.median(offs):.3f} [{offs.min():.3f} .. {offs.max():.3f}]')
    gaps = np.array([[key_db[k]['mf'] - key_db[k]['pp'], key_db[k]['ff'] - key_db[k]['mf']] for k in keys])
    print(f'per-key layer gaps dB: pp->mf median {np.median(gaps[:, 0]):.1f} [{gaps[:, 0].min():.1f}..{gaps[:, 0].max():.1f}], '
          f'mf->ff median {np.median(gaps[:, 1]):.1f} [{gaps[:, 1].min():.1f}..{gaps[:, 1].max():.1f}]')
    print('median-curve loudness re vel 127: ' + ', '.join(f'v{v} {T(v) - T(127):.1f}' for v in (1, 21, 42, 43, 64, 85, 86, 106, 127)) + ' dB')
    print('->', OUT_SFZ)


if __name__ == '__main__':
    main()
