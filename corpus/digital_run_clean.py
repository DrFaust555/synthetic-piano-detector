"""Run the demo cleaner (corpus/clean_demos.py) on a folder.

    python corpus/digital_run_clean.py <folder>      -> <folder>/clean/<name>.wav  (44.1 kHz mono FLOAT)

Only sets up the PANNs location (see digital_common.py) and then executes the module
exactly as `python corpus/clean_demos.py <folder>` would.
"""
from __future__ import annotations
import os, sys, runpy
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))   # repository root
import corpus.digital_common as common
common.setup()

if __name__ == '__main__':
    folder = sys.argv[1]
    sys.argv = ['clean_demos', folder]
    runpy.run_module('corpus.clean_demos', run_name='__main__')
