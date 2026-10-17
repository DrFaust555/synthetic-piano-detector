"""Fetch ONLY the real-Disklavier ISOL/NO notes of MAPS (ENSTDkCl, ENSTDkAm) from Zenodo via HTTP range
requests (no full 2.6-2.7 GB zip download).

MAPS - MIDI Aligned Piano Sounds, v0.4 (V. Emiya, N. Bertin, R. Badeau), Zenodo DOI 10.5281/zenodo.18160555,
licence CC BY-NC-SA 2.0 (FR) -- non-commercial research use with attribution.
Only ENSTDkCl / ENSTDkAm (Yamaha Disklavier Mark III upright, recorded with microphones, close / ambient)
are used; the software-piano subsets (AkPn*, Sptk*, Stbg*) are NOT downloaded.
Selection: ISOL/NO (normal isolated notes), sustain pedal off (S0), all three dynamics P/M/F.
Output: audio/_maps_isol/<subset>/*.wav (+ the .txt annotations), license.txt, readme.txt.

    python Stattia/revision/analysis/v4/get_maps_isol.py
"""
import io
import os
import re
import sys
import zipfile
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
DST = os.path.join(ROOT, 'audio', '_maps_isol')
BASE = 'https://zenodo.org/records/18160555/files/'
SUBSETS = ['ENSTDkCl', 'ENSTDkAm']
RX = re.compile(r'MAPS_ISOL_NO_([PMF])_S0_M(\d+)_(ENSTDk(?:Cl|Am))\.(wav|txt)$')


class HttpFile(io.RawIOBase):
    def __init__(self, url):
        self.url = url
        req = urllib.request.Request(url, method='HEAD')
        with urllib.request.urlopen(req) as r:
            self.size = int(r.headers['Content-Length'])
        self.pos = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        if n == 0 or self.pos >= self.size:
            return b''
        end = min(self.size, self.pos + n) - 1
        req = urllib.request.Request(self.url, headers={'Range': f'bytes={self.pos}-{end}'})
        for attempt in range(5):
            try:
                with urllib.request.urlopen(req) as r:
                    data = r.read()
                break
            except Exception:
                if attempt == 4:
                    raise
        self.pos += len(data)
        return data

    def readinto(self, b):
        d = self.read(len(b))
        b[:len(d)] = d
        return len(d)


def main():
    os.makedirs(DST, exist_ok=True)
    for f in ('license.txt', 'readme.txt'):
        p = os.path.join(DST, f)
        if not os.path.exists(p):
            urllib.request.urlretrieve(BASE + f + '?download=1', p)
    for sub in SUBSETS:
        od = os.path.join(DST, sub)
        os.makedirs(od, exist_ok=True)
        hf = io.BufferedReader(HttpFile(BASE + sub + '.zip?download=1'), buffer_size=1 << 20)
        z = zipfile.ZipFile(hf)
        names = [n for n in z.namelist() if RX.search(n.split('/')[-1]) and '/ISOL/NO/' in n]
        print(sub, 'selected', len(names), 'of', len(z.namelist()), flush=True)
        for i, n in enumerate(names):
            out = os.path.join(od, n.split('/')[-1])
            if os.path.exists(out) and os.path.getsize(out) == z.getinfo(n).file_size:
                continue
            with z.open(n) as src, open(out + '.part', 'wb') as dst:
                dst.write(src.read())
            os.replace(out + '.part', out)
            if i % 50 == 0:
                print(f'  {sub} {i}/{len(names)}', flush=True)
    print('done')


if __name__ == '__main__':
    sys.exit(main())
