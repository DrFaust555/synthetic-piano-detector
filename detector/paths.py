"""Locations used by every script of the repository.

DATA     feature tables, manifests and out-of-fold predictions shipped with the repository
RESULTS  what the experiment scripts write
AUDIO    root of the audio files named in the tables (not distributed; set SPD_AUDIO to its folder).
         Paths in the tables are relative to it, as in the authors' working copy (e.g. audio/..., otey/...).
"""
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
DATA = os.path.join(ROOT, 'data')
RESULTS = os.environ.get('SPD_RESULTS', os.path.join(ROOT, 'results'))
AUDIO_ROOT = os.environ.get('SPD_AUDIO', os.path.join(ROOT, 'audio_root'))


def audio_path(rel: str) -> str:
    """Absolute path of an audio file named in a table (relative to AUDIO_ROOT)."""
    return os.path.join(AUDIO_ROOT, rel)


def results_dir(name: str) -> str:
    d = os.path.join(RESULTS, name)
    os.makedirs(d, exist_ok=True)
    return d
