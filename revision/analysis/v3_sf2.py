"""Minimal SoundFont 2 (pdta) reader for control C2: which sample a rendered note plays, its transposition
from the sample's root key, and whether / when the sample loop is reached.

Follows the SF2.01 spec: phdr/pbag/pgen -> inst/ibag/igen -> shdr; preset = bank 0, program 0 (as
quality/render_soundfonts.py: program_select(0, sfid, 0, 0)); global zones supply defaults; preset-level
generators are additive offsets (only the tuning ones are used here).
Generators used: 0/4 startAddrs(+coarse), 2/45 startloopAddrs(+coarse), 1/12 endAddrs(+coarse),
3/50 endloopAddrs(+coarse), 43 keyRange, 44 velRange, 41 instrument, 53 sampleID, 54 sampleModes,
58 overridingRootKey, 51 coarseTune, 52 fineTune, 56 scaleTuning.
"""
from __future__ import annotations

import struct
import numpy as np


def _chunks(buf, off, end):
    while off + 8 <= end:
        cid = buf[off:off + 4].decode('latin-1')
        size = struct.unpack('<I', buf[off + 4:off + 8])[0]
        yield cid, off + 8, size
        off += 8 + size + (size & 1)


def read_pdta(path):
    buf = open(path, 'rb').read()
    pdta = None
    for cid, o, s in _chunks(buf, 12, len(buf)):
        if cid == 'LIST' and buf[o:o + 4] == b'pdta':
            pdta = (o + 4, o + s)
    if pdta is None:
        raise ValueError('no pdta')
    sub = {cid: buf[o:o + s] for cid, o, s in _chunks(buf, pdta[0], pdta[1])}

    def recs(name, fmt, size):
        b = sub[name]
        return [struct.unpack(fmt, b[i:i + size]) for i in range(0, len(b) - size + 1, size)]
    phdr = [(r[0].split(b'\0')[0].decode('latin-1', 'ignore'), r[1], r[2], r[3]) for r in recs('phdr', '<20sHHH12x', 38)]
    pbag = recs('pbag', '<HH', 4)
    pgen = recs('pgen', '<Hh', 4)
    inst = [(r[0].split(b'\0')[0].decode('latin-1', 'ignore'), r[1]) for r in recs('inst', '<20sH', 22)]
    ibag = recs('ibag', '<HH', 4)
    igen = recs('igen', '<Hh', 4)
    shdr = [dict(name=r[0].split(b'\0')[0].decode('latin-1', 'ignore'), start=r[1], end=r[2], loopstart=r[3],
                 loopend=r[4], rate=r[5], root=r[6], corr=r[7], link=r[8], type=r[9])
            for r in recs('shdr', '<20sIIIIIBbHH', 46)]
    return dict(phdr=phdr, pbag=pbag, pgen=pgen, inst=inst, ibag=ibag, igen=igen, shdr=shdr)


def _zones(bags, gens, first, last):
    """list of generator dicts for bag indices [first, last)."""
    out = []
    for b in range(first, last):
        g0, g1 = bags[b][0], bags[b + 1][0]
        d = {}
        for oper, amt in gens[g0:g1]:
            if oper in (43, 44):                         # ranges: lo, hi bytes
                lo, hi = amt & 0xFF, (amt >> 8) & 0xFF
                d[oper] = (lo, hi)
            else:
                d[oper] = amt
        out.append(d)
    return out


def _inr(d, oper, v):
    lo, hi = d.get(oper, (0, 127))
    return lo <= v <= hi


def note_samples(pd_, key, vel, bank=0, program=0):
    """samples played for (key, vel): list of dicts with sample header + effective generators."""
    ph = pd_['phdr']
    idx = next(i for i, p in enumerate(ph[:-1]) if p[1] == program and p[2] == bank)
    pz = _zones(pd_['pbag'], pd_['pgen'], ph[idx][3], ph[idx + 1][3])
    pglob = pz[0] if pz and 41 not in pz[0] else {}
    res = []
    for z in pz:
        if 41 not in z or not (_inr(z, 43, key) and _inr(z, 44, vel)):
            continue
        ii = z[41]
        inst = pd_['inst']
        iz = _zones(pd_['ibag'], pd_['igen'], inst[ii][1], inst[ii + 1][1])
        iglob = iz[0] if iz and 53 not in iz[0] else {}
        for zz in iz:
            if 53 not in zz:
                continue
            g = {**iglob, **zz}
            if not (_inr(g, 43, key) and _inr(g, 44, vel)):
                continue
            s = pd_['shdr'][g[53]]
            # preset-level tuning offsets are additive
            coarse = g.get(51, 0) + z.get(51, pglob.get(51, 0))
            fine = g.get(52, 0) + z.get(52, pglob.get(52, 0))
            scale = g.get(56, 100)
            root = g.get(58, -1)
            root = root if 0 <= root <= 127 else s['root']
            start = s['start'] + g.get(0, 0) + 32768 * g.get(4, 0)
            end = s['end'] + g.get(1, 0) + 32768 * g.get(12, 0)
            ls = s['loopstart'] + g.get(2, 0) + 32768 * g.get(45, 0)
            le = s['loopend'] + g.get(3, 0) + 32768 * g.get(50, 0)
            semis = (key - root) * scale / 100.0 + coarse + (fine + s['corr']) / 100.0
            res.append(dict(sample=s['name'], root=root, rate=s['rate'], start=start, end=end, loopstart=ls,
                            loopend=le, mode=int(g.get(54, 0)) & 3, transpose=float(key - root),
                            semis=float(semis)))
    return res


def note_info(pd_, key, vel, out_sr=44100):
    """per-note summary: |transposition| (semitones), loop mode, time (s of output) until the loop start
    and until the sample end, loop length (s); over all layers the most 'altered' layer is reported."""
    ss = note_samples(pd_, key, vel)
    if not ss:
        return None
    best = None
    for s in ss:
        speed = (2.0 ** (s['semis'] / 12.0)) * s['rate'] / out_sr          # source samples per output sample
        t_loop = (s['loopstart'] - s['start']) / (s['rate'] * 2.0 ** (s['semis'] / 12.0))
        t_end = (s['end'] - s['start']) / (s['rate'] * 2.0 ** (s['semis'] / 12.0))
        loop_len = (s['loopend'] - s['loopstart']) / (s['rate'] * 2.0 ** (s['semis'] / 12.0))
        rec = dict(n_layers=len(ss), sample=s['sample'], root=s['root'], transpose=s['transpose'],
                   abs_transpose=abs(s['transpose']), looped=s['mode'] in (1, 3), t_loop=float(t_loop),
                   t_end=float(t_end), loop_len=float(loop_len), speed=float(speed), mode=s['mode'])
        if best is None or (rec['abs_transpose'], rec['looped'], -rec['t_loop']) > \
                (best['abs_transpose'], best['looped'], -best['t_loop']):
            best = rec
    return best


def category(info, window_s):
    """C2 category for an analysed window of window_s seconds after note-on."""
    if info is None:
        return 'unknown'
    transposed = info['abs_transpose'] >= 0.5
    looped = info['looped'] and info['t_loop'] < window_s
    if not transposed and not looped:
        return 'unaltered sample'
    if transposed and not looped:
        return 'transposed only'
    if looped and not transposed:
        return 'looped only'
    return 'transposed + looped'
