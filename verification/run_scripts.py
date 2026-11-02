"""Run the paper analyses in fresh processes and keep their complete logs.

Usage: python verification/run_scripts.py [--root PATH] [--audio-root PATH] [--audio]
Audio collection and full network training are deliberately separate from reproduction
of the paper's statistics: they need original sources and substantially more time.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument('--audio-root', type=Path)
    ap.add_argument('--audio', action='store_true', help='also run both quality versions from audio')
    ap.add_argument('--workers', type=int, default=2)
    args = ap.parse_args()
    root = args.root.resolve()
    out = root / 'results' / 'validation'
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    if args.audio_root:
        env['SPD_AUDIO'] = str(args.audio_root.resolve())
    audio_only = {'table6_quality.py'}
    scripts = [(str(p.relative_to(root)), []) for p in sorted((root / 'experiments').rglob('*.py'))
               if args.audio or p.name not in audio_only]
    scripts.append(('figures/fig_scores.py', ['fus']))

    def run(job):
        rel, extra = job
        started = time.monotonic()
        log = out / (rel.replace('\\', '_').replace('/', '_') + '.log')
        with log.open('w', encoding='utf-8') as h:
            try:
                result = subprocess.run([sys.executable, '-X', 'utf8', '-u', rel, *extra],
                                        cwd=root, env=env, stdout=h, stderr=subprocess.STDOUT,
                                        timeout=3600)
                code = result.returncode
            except subprocess.TimeoutExpired:
                code = 'timeout'
        record = dict(script=rel, args=extra, exit_code=code,
                      seconds=round(time.monotonic() - started, 2), log=str(log.relative_to(root)))
        print(f'{code}: {rel} ({record["seconds"]} s)', flush=True)
        return record

    records = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run, job) for job in scripts]
        for f in as_completed(futures):
            records.append(f.result())
            (out / 'runs.json').write_text(json.dumps(sorted(records, key=lambda r: r['script']),
                                                    indent=2), encoding='utf-8')
    failed = [r for r in records if r['exit_code'] != 0]
    print(f'{len(records) - len(failed)}/{len(records)} scripts passed', flush=True)
    return bool(failed)


if __name__ == '__main__':
    raise SystemExit(main())
