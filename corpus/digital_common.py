"""Shared paths and environment of the digital-piano demo collection (corpus/digital_*.py).

Order of the steps:
    digital_known_ids.py   video ids already in the corpus (so no demo is collected twice)
    digital_search.py      YouTube search per model -> title-filtered candidates
    digital_acquire.py     download -> solo acoustic-piano excerpt (PANNs windows) -> raw excerpt
    digital_finalize.py    clean (clean_demos.py) and the 47 features
    digital_dupcheck.py    content duplicates (re-uploads of one recording)
    digital_manifest.py    manifest: model, sound engine, capture (line / mic), effects, exclusions

Hand-written inputs (model list, manufacturer sound-engine table, acquisition plans, hand review of
capture and effects) are read from data/performances/collection/.  Working files (candidate list, attempt
log, yt-dlp metadata cache, PANNs window probabilities, feature QC, manifest and feature table) are written
to results/corpus_digital/; the tables used in the paper are data/performances/digital_v2_manifest.csv and
digital_v2_features.csv.  Audio goes to <SPD_AUDIO>/audio/digital_pianos_v2/.

Environment: yt-dlp is run as `python -m yt_dlp` (a recent version; set DENO to a deno executable for
YouTube's JavaScript challenge).  panns_inference looks for its checkpoint in ~/panns_data; PANNS_HOME
points it to another home folder.  DPV2_SCRATCH holds the full downloads (deleted after cutting).
"""
from __future__ import annotations
import os
import sys

from detector.paths import DATA, RESULTS, AUDIO_ROOT

ROOT = AUDIO_ROOT                                              # audio files are named relative to it
COLLECTION = os.path.join(DATA, 'performances', 'collection')   # hand-written inputs
REV = os.path.join(RESULTS, 'corpus_digital')                   # working folder
SCRIPTS = REV
AUDIO_OUT = os.path.join(ROOT, 'audio', 'digital_pianos_v2')
RAW_DIR = os.path.join(AUDIO_OUT, '_raw')
TRENDS = os.path.join(DATA, 'corpus', 'register_trends_cal.json')
SCRATCH = os.environ.get('DPV2_SCRATCH', os.path.join(REV, 'scratch'))
PANNS_HOME = os.environ.get('PANNS_HOME', '')
DENO = os.environ.get('DENO', '')

MANIFEST = os.path.join(REV, 'digital_v2_manifest.csv')
FEATURES = os.path.join(REV, 'digital_v2_features.csv')
CANDIDATES = os.path.join(REV, 'digital_v2_candidates.csv')
MODELS_JSON = os.path.join(COLLECTION, 'digital_models.json')
os.makedirs(REV, exist_ok=True)


def setup():
    """Environment for PANNs (home folder of its checkpoint) and UTF-8 output."""
    if PANNS_HOME:
        os.environ['USERPROFILE'] = PANNS_HOME
        os.environ['HOME'] = PANNS_HOME
    os.environ.setdefault('PYTHONUTF8', '1')


def ytdlp_cmd():
    """yt-dlp as a module of this interpreter (with deno for the JavaScript challenge when DENO is set).
    Downloads use polite sleeps: YouTube rate-limits bursts (HTTP 429 / bot check)."""
    cmd = [sys.executable, '-m', 'yt_dlp']
    if DENO and os.path.exists(DENO):
        cmd += ['--js-runtimes', f'deno:{DENO}']
    return cmd


def ytdlp_env():
    env = dict(os.environ)
    env['PYTHONUTF8'] = '1'
    return env
