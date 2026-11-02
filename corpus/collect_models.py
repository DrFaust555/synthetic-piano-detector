"""Collect extra YouTube recordings for EXISTING Yamaha models into a staging dir,
gating by "is it a piano?" (PANNs) and removing speech. Rating deferred.

    python corpus/collect_models.py <model_key> [--target 10] [--per-search 14]
    python corpus/collect_models.py --list

Staging: <SPD_AUDIO>/audio/yamaha_extra/  (+ _raw/ for raw downloads, index.csv catalog).
Dedup against ALL existing manifest video-ids AND the staging index.
"""
from __future__ import annotations
import os, sys, csv, re, subprocess, json

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..')))   # repository root
from detector.paths import DATA, RESULTS, AUDIO_ROOT, results_dir   # noqa: E402
OUT = results_dir('corpus')
ROOT = AUDIO_ROOT
AUDIO = os.path.join(ROOT, 'audio')
STAGE = os.path.join(AUDIO, 'yamaha_extra')
RAW = os.path.join(STAGE, '_raw')
INDEX = os.path.join(STAGE, 'index.csv')
MANIFEST = os.path.join(DATA, 'protocol', 'manifest.csv')
# yt-dlp needs ffmpeg to extract wav: from PATH, or from the folder in FFMPEG_DIR.
FFDIR = os.environ.get('FFMPEG_DIR', '')

# existing Yamaha upright models -> search query
MODELS = {
    'u1':   'Yamaha U1 upright piano',
    'u3':   'Yamaha U3 upright piano',
    'b2':   'Yamaha b2 upright piano',
    'b3':   'Yamaha b3 upright piano',
    'm1':   'Yamaha M1 console upright piano',
    'm2':   'Yamaha M2 console upright piano',
    'm500': 'Yamaha M500 console upright piano',
    'yus1': 'Yamaha YUS1 upright piano',
    'yus3': 'Yamaha YUS3 upright piano',
    'yus5': 'Yamaha YUS5 upright piano',
    'ju109': 'Yamaha JU109 upright piano',
    'jx113': 'Yamaha JX113 upright piano',
    'su7':  'Yamaha SU7 upright piano',
    'p22':  'Yamaha P22 upright piano',
    'p114': 'Yamaha P114 upright piano',
    'eterna': 'Yamaha Eterna upright piano',
}

VID_RE = re.compile(r'(?:v=|/shorts/|youtu\.be/)([A-Za-z0-9_-]{11})')
COMPARISONS = os.path.join(STAGE, 'comparisons.csv')

# title-based rejection: avoid two-instrument / wrong-model files
OTHER_BRANDS = ['kawai', 'kemble', 'steinway', 'boston', 'essex', 'samick', 'reid',
                'korg', 'belarus', 'беларусь', 'petrof', 'bechstein', 'baldwin',
                'young chang', 'pearl river', 'casio', 'roland', 'nord', 'feurich',
                'ritm', 'kayserburg', 'hailun', 'schimmel', 'seiler', 'bosendorfer',
                'bösendorfer', 'fazioli', 'estonia', 'grotrian', 'bluthner', 'blüthner',
                'calisia', 'legnica', 'weber', 'knabe', 'sojin', 'wurlitzer',
                'hoffmann', 'hofmann', 'kingsburg', 'brodmann', 'perzina', 'irmler',
                'sauter', 'förster', 'forster', 'pear'].copy()
CMP_KW = [' vs ', ' vs.', 'vs ', 'versus', 'v.s', 'compar', 'порівн', 'сравн',
          'shootout', 'battle', 'which is best', 'which is better', 'head to head']

MODEL_TOKENS = {
    'u1': ['u1'], 'u3': ['u3'], 'b2': ['b2'], 'b3': ['b3'], 'm1': ['m1'], 'm2': ['m2'],
    'm500': ['m500', 'm-500', 'm 500'], 'yus1': ['yus1', 'yus 1', 'yus-1'],
    'yus3': ['yus3', 'yus 3', 'yus-3'], 'yus5': ['yus5', 'yus 5', 'yus-5'],
    'ju109': ['ju109', 'ju 109', 'ju-109', 'ju 9'], 'jx113': ['jx113', 'jx 113', 'jx-113'],
    'su7': ['su7', 'su-7', 'su 7'], 'p22': ['p22'], 'p114': ['p114'], 'eterna': ['eterna'],
}
_OTHER_MODEL_TOKENS = ['u5', 'u7', 'u30', 'ux', 'b1', 'p660', 'p116', 'p121', 'p121',
                       'c1x', 'c3', 'c7x', 'cfx', 'gb1', 't116', 't118', 't121', 'gc1']


def is_multi_instrument(title: str) -> bool:
    t = ' ' + (title or '').lower() + ' '
    if any(k in t for k in CMP_KW):
        return True
    if any(b in t for b in OTHER_BRANDS):
        return True
    return False


def title_wrong_model(title: str, key: str) -> bool:
    t = (title or '').lower()
    own = MODEL_TOKENS.get(key, [key])
    if any(o in t for o in own):
        return False                      # own model named -> fine
    others = set(_OTHER_MODEL_TOKENS)
    for k, toks in MODEL_TOKENS.items():
        if k != key:
            others.update(toks)
    for o in others:
        if re.search(r'(?<![a-z0-9])' + re.escape(o) + r'(?![0-9])', t):
            return True                   # a DIFFERENT model is named, own absent
    return False


def log_comparison(model, vid, title):
    new = not os.path.exists(COMPARISONS)
    with open(COMPARISONS, 'a', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        if new:
            w.writerow(['searched_model', 'video_id', 'url', 'title'])
        w.writerow([model, vid, f'https://www.youtube.com/watch?v={vid}', title])


def existing_video_ids():
    ids = set()
    with open(MANIFEST, encoding='utf-8') as f:
        for r in csv.DictReader(f):
            m = VID_RE.search(r.get('url', ''))
            if m:
                ids.add(m.group(1))
    return ids


def read_index():
    if not os.path.exists(INDEX):
        return []
    with open(INDEX, encoding='utf-8') as f:
        return list(csv.DictReader(f))


def append_index(row):
    new = not os.path.exists(INDEX)
    with open(INDEX, 'a', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        if new:
            w.writerow(['file', 'model', 'video_id', 'url', 'title', 'dur_s', 'kept_frac', 'speech_max', 'flagged'])
        w.writerow(row)


def yt_search(query, n):
    cmd = ['yt-dlp', f'ytsearch{n}:{query}', '--skip-download', '--no-warnings',
           '--print', '%(id)s@@@%(title)s@@@%(duration)s']
    out = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
    cands = []
    for line in (out.stdout or '').splitlines():
        p = line.split('@@@')
        if len(p) >= 3:
            cands.append((p[0], p[1], p[2]))
    return cands


def download(vid):
    os.makedirs(RAW, exist_ok=True)
    out = os.path.join(RAW, f'{vid}.wav')
    if os.path.exists(out):
        return out
    env = dict(os.environ)
    if FFDIR:
        env['PATH'] = FFDIR + os.pathsep + env.get('PATH', '')
    for fmt in ('251/140/bestaudio', '140/bestaudio'):
        cmd = ['yt-dlp', '--no-playlist', '--no-warnings', '--retries', '5', '-f', fmt,
               '-x', '--audio-format', 'wav', '--audio-quality', '0',
               '--postprocessor-args', 'ffmpeg:-ar 44100',
               '-o', os.path.join(RAW, f'{vid}.%(ext)s'),
               f'https://www.youtube.com/watch?v={vid}']
        subprocess.run(cmd, capture_output=True, text=True, env=env)
        if os.path.exists(out):
            return out
    return None


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('model', nargs='?')
    ap.add_argument('--target', type=int, default=10)
    ap.add_argument('--per-search', type=int, default=14)
    ap.add_argument('--list', action='store_true')
    a = ap.parse_args()
    if a.list or not a.model:
        print("models:", ", ".join(MODELS)); return
    if a.model not in MODELS:
        print("unknown model; use --list"); return

    os.makedirs(STAGE, exist_ok=True); os.makedirs(RAW, exist_ok=True)
    from corpus.pianoness import load_model, _labels, _ensure_assets, tag_file
    from corpus.despeech import despeech_file
    _ensure_assets(); lab = _labels(); at = load_model()

    skip = existing_video_ids()
    idx = read_index()
    for r in idx:
        skip.add(r['video_id'])
    have_model = sum(1 for r in idx if r['model'] == a.model)
    need = max(0, a.target - have_model)
    print(f"[{a.model}] already staged: {have_model}; target {a.target}; still needed ~{need}")
    if need == 0:
        return

    cands = yt_search(MODELS[a.model], a.per_search)
    print(f"found {len(cands)} candidates for the query {MODELS[a.model]!r}")
    got = 0
    for vid, title, dur in cands:
        if got >= need:
            break
        if vid in skip:
            print(f"  - {vid} skip (already known)"); continue
        skip.add(vid)
        if is_multi_instrument(title):
            log_comparison(a.model, vid, title)
            print(f"  - {vid} COMPARISON -> comparisons.csv ({title[:45]})"); continue
        if title_wrong_model(title, a.model):
            print(f"  - {vid} skip wrong-model ({title[:45]})"); continue
        try:
            d = float(dur) if dur and dur != 'NA' else 0
        except ValueError:
            d = 0
        if d and (d < 8 or d > 1200):
            print(f"  - {vid} skip (duration {d}s)"); continue
        raw = download(vid)
        if not raw:
            print(f"  - {vid} DOWNLOAD FAIL ({title[:40]})"); continue
        # piano gate
        g = tag_file(raw, at, lab)
        if not g['is_piano']:
            print(f"  - {vid} REJECT not-piano (pn={g['pianoness']:.2f}, top={g['top5'][0][1]}) {title[:40]}")
            os.remove(raw); continue
        out = os.path.join(STAGE, f'yam_{a.model}_{vid}.wav')
        r = despeech_file(raw, out, at, lab)
        append_index([os.path.basename(out), a.model, vid,
                      f'https://www.youtube.com/watch?v={vid}', title,
                      f"{r['dur_out']:.1f}", f"{r['kept_frac']:.2f}",
                      f"{r['speech_max']:.2f}", '1' if r['flagged'] else '0'])
        got += 1
        print(f"  + {vid} OK pn={g['pianoness']:.2f} kept={r['kept_frac']*100:.0f}% "
              f"dur->{r['dur_out']:.1f}s  {title[:50]}")
    print(f"[{a.model}] added {got}; staged in total {have_model+got}")


if __name__ == '__main__':
    main()
