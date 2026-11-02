"""Collect every YouTube video ID already present anywhere in the project's corpora, so the
new digital-piano set can exclude duplicates.

Sources scanned (read-only):
  audio/manifest.csv (url column)                    -- rated real corpus ("flat" files)
  audio/*_extra/index.csv (video_id)                 -- bulk YouTube real corpus (2,144 files)
  audio/*_extra/comparisons.csv (video_id)           -- rejected comparison videos (logged)
  audio/sources_licenses.csv (any URL)               -- licence list
  data/corpus/features_cal.csv + synth_features.csv file names (11-char IDs after '_')
  audio/SynthPianoSamples/yt/**.wav names            -- the 14 existing demos + raw extras
  audio/**/*.wav names that end in an 11-char ID      -- anything else on disk

    python corpus/digital_known_ids.py   -> results/corpus_digital/known_video_ids.csv (id, source)
"""
from __future__ import annotations
import os, sys, re, csv, glob
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))   # repository root
import corpus.digital_common as common

VID_RE = re.compile(r'(?:v=|/shorts/|youtu\.be/|/embed/)([A-Za-z0-9_-]{11})')
TAIL_RE = re.compile(r'_([A-Za-z0-9_-]{11})(?:_clean)?\.(?:wav|flac|mp3|ogg|webm|m4a|opus)$')


def collect():
    A = os.path.join(common.ROOT, 'audio')
    ids = {}

    def add(v, src):
        if v and len(v) == 11:
            ids.setdefault(v, set()).add(src)

    man = os.path.join(common.DATA, 'protocol', 'manifest.csv')
    with open(man, encoding='utf-8') as f:
        for r in csv.DictReader(f):
            for m in VID_RE.finditer(r.get('url', '') or ''):
                add(m.group(1), 'manifest.csv')
    lic = os.path.join(A, 'sources_licenses.csv')
    if os.path.exists(lic):
        txt = open(lic, encoding='utf-8', errors='replace').read()
        for m in VID_RE.finditer(txt):
            add(m.group(1), 'sources_licenses.csv')
    for idx in glob.glob(os.path.join(A, '*_extra', '*.csv')):
        with open(idx, encoding='utf-8', errors='replace') as f:
            for r in csv.DictReader(f):
                v = r.get('video_id') or ''
                if not v:
                    m = VID_RE.search(r.get('url', '') or '')
                    v = m.group(1) if m else ''
                add(v, os.path.relpath(idx, A).replace('\\', '/'))
    for csvp in [os.path.join(common.DATA, 'corpus', 'features_cal.csv'),
                 os.path.join(common.DATA, 'corpus', 'synth_features.csv')]:
        if os.path.exists(csvp):
            with open(csvp, encoding='utf-8') as f:
                for r in csv.DictReader(f):
                    m = TAIL_RE.search(r.get('file', ''))
                    if m:
                        add(m.group(1), os.path.basename(csvp))
    for p in glob.glob(os.path.join(A, '**', '*.*'), recursive=True):
        if 'digital_pianos_v2' in p:
            continue
        m = TAIL_RE.search(os.path.basename(p))
        if m:
            add(m.group(1), 'audio-filename')
        else:
            b = os.path.splitext(os.path.basename(p))[0]
            if re.fullmatch(r'[A-Za-z0-9_-]{11}', b):
                add(b, 'audio-filename(_raw)')
    # existing demos: one ID is stored with a leading underscore in the file name
    add('_2deTG9cAa0', 'SynthPianoSamples/yt (Korg D1)')
    add('CVq4rSEjgnA', 'SynthPianoSamples/yt/raw (SC-55)')
    return ids


def main():
    ids = collect()
    out = os.path.join(common.REV, 'known_video_ids.csv')
    with open(out, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['video_id', 'sources'])
        for v in sorted(ids):
            w.writerow([v, ';'.join(sorted(ids[v]))])
    print(f'{len(ids)} known IDs -> {out}')


if __name__ == '__main__':
    main()
