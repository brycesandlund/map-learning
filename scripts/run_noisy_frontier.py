"""Launch matched two-arm study with bounded concurrency; never overwrite runs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--jobs', type=int, default=3)
    p.add_argument('--phases', type=int, default=20)
    p.add_argument('--expansions', type=int, default=1000)
    p.add_argument('--epochs', type=int, default=1)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    jobs = [(arm, seed) for seed in range(3) for arm in ['distance', 'disagreement']]
    manifest = {'arms': ['distance', 'disagreement'], 'seeds': [0, 1, 2], 'phases': a.phases,
                'expansions': a.expansions, 'epochs': a.epochs, 'nuisance_dims': 8}
    (a.out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    def run(job):
        arm, seed = job
        name = f'{arm}-seed{seed}'
        command = [sys.executable, 'scripts/noisy_frontier.py', '--arm', arm, '--seed', str(seed),
                   '--phases', str(a.phases), '--expansions', str(a.expansions), '--epochs', str(a.epochs), '--out', str(a.out/name)]
        with (a.out/f'{name}.log').open('x') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        print(json.dumps({'run': name, 'exit_code': result.returncode}), flush=True)
        return result.returncode
    with ThreadPoolExecutor(max_workers=a.jobs) as pool: codes = list(pool.map(run, jobs))
    if any(codes): raise SystemExit(1)

if __name__ == '__main__': main()
