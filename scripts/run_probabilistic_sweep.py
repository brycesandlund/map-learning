"""Manifest-driven online sweep with bounded worker processes and exact coverage AUC.

Each worker is a fresh Python process, keeping model/planner RNGs isolated. Raw
run artifacts and source snapshots are preserved. Re-running a manifest skips
completed runs only after checking their full config. Failed runs need a new
output directory (partial runs are never silently overwritten).
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from map_learning.config import load_config


def coverage_statistics(positions, next_positions, denominator, rooms=False):
    points = np.stack([positions, next_positions], axis=1).reshape(-1, 2)
    points = np.floor(points) if rooms else np.round(points, 5)
    _, first = np.unique(points, axis=0, return_index=True)
    first_steps = np.sort(first//2+1)
    steps = len(positions)
    result = {'auc': float(np.sum(steps-first_steps+1)/(steps*denominator)),
              'final': len(first_steps)/denominator}
    for fraction in (.25, .5, .8, .95, 1.):
        count = int(np.ceil(fraction*denominator))
        result[f'to_{round(100*fraction)}'] = int(first_steps[count-1]) if count <= len(first_steps) else None
    return result


def summarize(path, variant, seed):
    metrics = json.loads((path/'metrics.json').read_text())
    config = json.loads((path/'config.json').read_text())
    curve = [json.loads(line) for line in (path/'exploration.jsonl').read_text().splitlines()]
    with np.load(path/'replay.npz') as replay:
        lattice = coverage_statistics(replay['positions'], replay['next_positions'], metrics['oracle_nodes'])
        rooms = coverage_statistics(replay['positions'], replay['next_positions'], config['maze']['size']**2, True)
    return {'variant': variant, 'seed': seed, 'run': str(path.resolve()),
            'collection_seconds': curve[-1]['seconds'], 'coverage_lattice_stats': lattice,
            'coverage_room_stats': rooms, 'metrics': metrics}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--jobs', type=int, default=3)
    args = parser.parse_args()
    manifest_text = Path(args.manifest).read_text()
    manifest = json.loads(manifest_text)
    root = Path(args.out).resolve()
    root.mkdir(parents=True, exist_ok=True)
    saved = root/'manifest.json'
    if saved.exists() and json.loads(saved.read_text()) != manifest:
        raise ValueError('Output directory belongs to a different manifest')
    saved.write_text(manifest_text)
    rows, failures = [], []
    started = time.monotonic()

    def run(variant, seed):
        overrides = [*manifest.get('overrides', []), *variant.get('overrides', []), f'seed={seed}']
        cfg = load_config(manifest['config'], overrides)
        name = f"{variant['name']}-seed{seed}"
        path = root/name
        if path.exists():
            if not (path/'metrics.json').exists():
                raise ValueError(f'Incomplete run needs inspection: {path}')
            if asdict(load_config(path/'config.json')) != asdict(cfg):
                raise ValueError(f'Existing config mismatch: {path}')
        else:
            command = [sys.executable, '-m', 'map_learning', 'online', '--config', manifest['config'], '--out', str(path)]
            for override in overrides:
                command += ['--set', override]
            with (root/f'{name}.log').open('w') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        return summarize(path, variant['name'], seed)

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        pending = {pool.submit(run, variant, seed): (variant['name'], seed)
                   for variant in manifest['variants'] for seed in manifest['seeds']}
        for future in as_completed(pending):
            try:
                row = future.result()
                rows.append(row)
                rows.sort(key=lambda x: (x['variant'], x['seed']))
                (root/'summary.json').write_text(json.dumps(rows, indent=2, allow_nan=False)+'\n')
                print(json.dumps({'done': pending[future], 'runs_completed': len(rows),
                      'lattice': row['coverage_lattice_stats'], 'rooms': row['coverage_room_stats'],
                      'seconds': round(time.monotonic()-started, 1)}), flush=True)
            except Exception as exc:
                failures.append({'run': pending[future], 'error': str(exc)})
                (root/'failures.json').write_text(json.dumps(failures, indent=2)+'\n')
                print(json.dumps(failures[-1]), flush=True)
    if failures:
        raise SystemExit(f'{len(failures)} runs failed; inspect failures.json and individual logs')
    previous = root/'failures.json'
    if previous.exists():
        archive = root/'resolved-failures.json'
        history = json.loads(archive.read_text()) if archive.exists() else []
        history.append({'failures': json.loads(previous.read_text()),
                        'resolution': 'All manifest runs now have completed, config-validated evaluations.'})
        archive.write_text(json.dumps(history, indent=2)+'\n')
        previous.unlink()


if __name__ == '__main__':
    main()
