"""Matched online interaction budgets; every method trains on its own experience."""
import argparse
import json
from pathlib import Path
from map_learning.config import load_config
from map_learning.experiment import online


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/probabilistic_maze.json')
    parser.add_argument('--out', required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--methods', nargs='+', choices=['random', 'mcts', 'shooting'], default=['random', 'mcts', 'shooting'])
    parser.add_argument('--set', action='append', default=[])
    args = parser.parse_args()
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=False)
    rows = []
    for seed in args.seeds:
        for method in args.methods:
            behavior = 'random' if method == 'random' else 'disagreement'
            search = 'mcts' if method == 'random' else method
            cfg = load_config(args.config, [*args.set, f'seed={seed}',
                              f'planner.behavior={behavior}', f'planner.search={search}'])
            path = online(cfg, root/f'{method}-seed{seed}')
            metrics = json.loads((path/'metrics.json').read_text())
            rows.append({'method': method, 'seed': seed, **metrics})
            (root/'summary.json').write_text(json.dumps(rows, indent=2)+'\n')


if __name__ == '__main__':
    main()
