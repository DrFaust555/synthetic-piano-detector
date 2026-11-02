"""Content-duplicate check for the cleaned digital_pianos_v2 files (re-uploads of the same
recording under another video ID, e.g. dealer channels re-posting manufacturer demos).

    python corpus/digital_dupcheck.py  -> results/corpus_digital/dpv2_dupcheck.csv

Fingerprint: onset-strength envelope (22.05 kHz, hop 512 = 23 ms), z-scored.  For every
pair the Pearson correlation is computed at the best lag over an overlap >= 20 s (FFT
cross-correlation, then exact r at the top 5 lags).  Identical audio gives r ~ 0.9+,
different performances r < 0.3.  Also compared against the 14 old 'yt' demo files.
"""
from __future__ import annotations
import os, sys, glob, csv
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))   # repository root
import corpus.digital_common as common
import numpy as np
import librosa

SR = 22050
HOP = 512
MIN_OV = int(20 * SR / HOP)


def env(path):
    y, _ = librosa.load(path, sr=SR, mono=True)
    o = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP)
    return (o - o.mean()) / (o.std() + 1e-9)


def best_r(a, b):
    n = len(a) + len(b) - 1
    nf = 1 << int(np.ceil(np.log2(n)))
    xc = np.fft.irfft(np.fft.rfft(a, nf) * np.conj(np.fft.rfft(b, nf)), nf)
    lags = np.concatenate([np.arange(0, len(a)), np.arange(-(len(b) - 1), 0)])
    vals = np.concatenate([xc[:len(a)], xc[nf - (len(b) - 1):]])
    best = -1.0
    for i in np.argsort(-vals)[:5]:
        L = lags[i]
        if L >= 0:
            x = a[L:L + len(b)]; y = b[:len(x)]
        else:
            y = b[-L:-L + len(a)]; x = a[:len(y)]
        if len(x) < MIN_OV:
            continue
        r = np.corrcoef(x, y)[0, 1]
        best = max(best, float(r))
    return best


def main():
    new = sorted(glob.glob(os.path.join(common.AUDIO_OUT, '*.wav')))
    old = sorted(glob.glob(os.path.join(common.ROOT, 'audio', 'SynthPianoSamples', 'yt', '*.wav')))
    E = {p: env(p) for p in new + old}
    rows = []
    for i, p in enumerate(new):
        for q in new[i + 1:] + old:
            r = best_r(E[p], E[q])
            if r > 0.5:
                rows.append({'file_a': os.path.basename(p), 'file_b': os.path.basename(q),
                             'r': round(r, 3)})
                print(f'{r:.3f}  {os.path.basename(p)}  ~  {os.path.basename(q)}', flush=True)
    out = os.path.join(common.SCRIPTS, 'dpv2_dupcheck.csv')
    with open(out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['file_a', 'file_b', 'r'])
        w.writeheader()
        for x in rows:
            w.writerow(x)
    print(f'{len(rows)} pairs with r > 0.5 -> {out}')


if __name__ == '__main__':
    main()
