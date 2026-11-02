"""Instrument-identity gate: "is this even a piano?" via a pretrained net (PANNs).

    python corpus/pianoness.py <wav> [<wav> ...]   # tag arbitrary files
    python corpus/pianoness.py --dataset [--limit N]  # summarise over the hand-selected set
    python corpus/pianoness.py --selftest          # synthetic noise/sine/pluck probe

Uses PANNs CNN14 (AudioSet tagging, 527 classes) — it outputs a `Piano` probability
directly, so no training and no labelled negatives are needed. It is the gate of
the corpus collection (is the recording a piano at all?); a pretrained net is acceptable
here because instrument identity is robust to the recording channel. It also tags
digital, FM and modelled pianos as "piano": real vs synthetic is the detector's job.

Checkpoint (~300 MB) auto-downloads to ~/panns_data on first use (needs internet
once). PANNs expects 32 kHz mono.
"""

from __future__ import annotations
import os
import sys
import csv
import shutil
import argparse
import urllib.request
import numpy as np

# panns_inference auto-downloads via `wget` (absent on Windows) AT IMPORT TIME,
# so we fetch its two assets ourselves (requests handles Zenodo redirects/CDN that
# 403 plain urllib) before importing it.  Several URLs are tried per asset.
_PANNS_ASSETS = {
    'class_labels_indices.csv': [
        'http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv',
    ],
    'Cnn14_mAP=0.431.pth': [
        # Zenodo 403s automated clients; HuggingFace mirror is reliable.
        'https://huggingface.co/thelou1s/panns-inference/resolve/main/Cnn14_mAP=0.431.pth?download=true',
        'https://huggingface.co/datasets/Jinbo-HU/PSELDNets/resolve/main/model/Cnn14_mAP=0.431.pth?download=true',
        'https://zenodo.org/records/3987831/files/Cnn14_mAP%3D0.431.pth?download=1',
    ],
}
_UA = ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
       '(KHTML, like Gecko) Chrome/120.0 Safari/537.36')


def _download(url: str, dest: str):
    import requests
    with requests.get(url, headers={'User-Agent': _UA}, stream=True,
                      timeout=180, allow_redirects=True) as r:
        r.raise_for_status()
        with open(dest, 'wb') as f:
            for chunk in r.iter_content(1 << 20):
                if chunk:
                    f.write(chunk)


def _ensure_assets():
    d = os.path.join(os.path.expanduser('~'), 'panns_data')
    os.makedirs(d, exist_ok=True)
    for fn, urls in _PANNS_ASSETS.items():
        p = os.path.join(d, fn)
        min_size = 3e8 if fn.endswith('.pth') else 1000
        if os.path.exists(p) and os.path.getsize(p) >= min_size:
            continue
        print(f"Downloading {fn} ...")
        last = None
        for url in urls:
            try:
                _download(url, p)
                if os.path.getsize(p) >= min_size:
                    break
            except Exception as e:
                last = e
                print(f"  (failed from {url[:60]}...: {e})")
        if not (os.path.exists(p) and os.path.getsize(p) >= min_size):
            raise RuntimeError(f"Could not download {fn}: {last}")
        print(f"  -> {p} ({os.path.getsize(p)/1e6:.1f} MB)")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..')))   # repository root
from detector.paths import DATA, RESULTS, AUDIO_ROOT, results_dir   # noqa: E402
OUT = results_dir('corpus')
AUDIO_DIR = os.path.join(AUDIO_ROOT, 'audio')
MANIFEST = os.path.join(DATA, 'protocol', 'manifest.csv')
PANNS_SR = 32000
MAX_SECONDS = 30.0   # central window fed to the tagger (bounds memory/time)

# piano-family tags (the gate) and contrast tags (for diagnostics)
PIANO_TAGS = ['Piano', 'Electric piano', 'Keyboard (musical)']
# non-keyboard instruments that should "lose" to piano for the gate to pass
OTHER_INSTR_TAGS = ['Guitar', 'Acoustic guitar', 'Electric guitar',
                    'Plucked string instrument', 'Violin, fiddle', 'Drum kit',
                    'Singing', 'Speech']
CONTRAST_TAGS = OTHER_INSTR_TAGS + ['Organ', 'Synthesizer']
# absolute floor so pure noise/silence (piano≈0) is rejected even if "other"≈0 too
PIANO_FLOOR = 0.08


def _labels():
    try:
        from panns_inference import labels
    except ImportError:
        from panns_inference.config import labels
    return list(labels)


def load_model():
    import torch
    from panns_inference import AudioTagging
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"PANNs device: {device}")
    return AudioTagging(checkpoint_path=None, device=device)


def _to_panns(y: np.ndarray, sr: int) -> np.ndarray:
    import librosa
    if y.ndim > 1:
        y = y.mean(axis=1)
    n_keep = int(MAX_SECONDS * sr)
    if len(y) > n_keep:                       # central window
        a = (len(y) - n_keep) // 2
        y = y[a:a + n_keep]
    if sr != PANNS_SR:
        y = librosa.resample(y.astype(np.float64), orig_sr=sr, target_sr=PANNS_SR)
    return y.astype(np.float32)[None, :]      # PANNs wants [1, n]


def tag_audio(y: np.ndarray, sr: int, at, lab: list) -> dict:
    clip, _ = at.inference(_to_panns(y, sr))   # clip: [1, 527]
    probs = clip[0]
    idx = {name: lab.index(name) for name in (PIANO_TAGS + CONTRAST_TAGS) if name in lab}
    tags = {name: float(probs[i]) for name, i in idx.items()}
    pianoness = max(tags.get(t, 0.0) for t in PIANO_TAGS)
    other_instr = max((tags.get(t, 0.0) for t in OTHER_INSTR_TAGS), default=0.0)
    # robust gate: piano-family dominates other instruments AND clears a small floor
    is_piano = (pianoness >= PIANO_FLOOR) and (pianoness > other_instr)
    top = sorted(((float(probs[i]), lab[i]) for i in range(len(probs))), reverse=True)[:5]
    return {'pianoness': pianoness, 'other_instr': other_instr,
            'is_piano': is_piano, 'tags': tags, 'top5': top}


def tag_file(path: str, at, lab: list) -> dict:
    from detector.segment import load_mono
    y, sr = load_mono(path)
    return tag_audio(y, sr, at, lab)


# ---------------------------------------------------------------------------
def _print_one(name: str, r: dict):
    gate = 'PIANO' if r['is_piano'] else 'NO'
    print(f"\n{name}")
    print(f"  pianoness={r['pianoness']:.3f}  other_instr={r['other_instr']:.3f}  "
          f"=> {gate}")
    print(f"  (Piano={r['tags'].get('Piano', 0):.2f}, "
          f"E.piano={r['tags'].get('Electric piano', 0):.2f}, "
          f"Guitar={r['tags'].get('Guitar', 0):.2f})")
    print("  top-5: " + ", ".join(f"{n} {p:.2f}" for p, n in r['top5']))


def _dataset(at, lab, limit):
    rows = []
    with open(MANIFEST, encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if (r.get('rating') or '').strip():
                rows.append(r)
    if limit:
        rows = rows[:limit]
    by_src = {}
    rejected = []
    for i, r in enumerate(rows, 1):
        path = os.path.join(AUDIO_DIR, r['file'])
        if not os.path.exists(path):
            continue
        res = tag_file(path, at, lab)
        by_src.setdefault(r.get('source', '?'), []).append(res)
        if not res['is_piano']:
            rejected.append((res['pianoness'], res['other_instr'], r['file'],
                             res['top5'][0][1]))
        print(f"  [{i}/{len(rows)}] {r['file']:<42} "
              f"pn={res['pianoness']:.3f} other={res['other_instr']:.3f} "
              f"{'OK' if res['is_piano'] else 'REJECT'}")
    print("\nBy source (gate = the piano family dominates the other instruments):")
    for src, vals in sorted(by_src.items()):
        pn = np.array([v['pianoness'] for v in vals])
        n_ok = sum(v['is_piano'] for v in vals)
        print(f"  {src:<10} n={len(vals):3d}  pass={n_ok:3d}  "
              f"pn median={np.median(pn):.3f} min={pn.min():.3f}")
    if rejected:
        print(f"\nRejected by the gate ({len(rejected)}), candidates for 'not a piano':")
        for pn, oth, fn, top in sorted(rejected):
            print(f"  pn={pn:.3f} other={oth:.3f}  {fn:<40} (top: {top})")
    else:
        print("\nAll files passed the gate.")


def _selftest(at, lab):
    sr = 44100
    t = np.arange(int(3 * sr)) / sr
    rng = np.random.RandomState(0)
    sine = 0.3 * np.sin(2 * np.pi * 440 * t)
    noise = 0.3 * rng.randn(len(t))
    # Karplus-Strong pluck (~guitar-ish) at ~196 Hz
    N = sr // 196
    buf = rng.randn(N)
    ks = np.zeros(len(t))
    for n in range(len(t)):
        ks[n] = buf[n % N]
        buf[n % N] = 0.5 * (buf[n % N] + buf[(n + 1) % N]) * 0.996
    ks *= 0.4
    for name, y in [('SINE 440Hz', sine), ('WHITE NOISE', noise),
                    ('KARPLUS-STRONG pluck', ks)]:
        _print_one(name, tag_audio(y, sr, at, lab))


def main():
    ap = argparse.ArgumentParser(description="Is it a piano? (PANNs)")
    ap.add_argument('wavs', nargs='*')
    ap.add_argument('--dataset', action='store_true')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--selftest', action='store_true')
    args = ap.parse_args()

    print("Checking / downloading the PANNs weights (about 300 MB the first time) ...")
    _ensure_assets()
    lab = _labels()
    at = load_model()
    print("Ready.")

    if args.selftest:
        _selftest(at, lab)
    if args.dataset:
        _dataset(at, lab, args.limit)
    for w in args.wavs:
        if os.path.exists(w):
            _print_one(os.path.basename(w), tag_file(w, at, lab))
        else:
            print(f"[no such file] {w}")
    if not (args.selftest or args.dataset or args.wavs):
        print("Give <wav>, --dataset or --selftest.")


if __name__ == '__main__':
    main()
