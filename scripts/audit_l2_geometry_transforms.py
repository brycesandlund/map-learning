"""Frozen-queue geometry intervention; covariance fitted only on visited latents."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from map_learning.training import Trainer, seed_all
from map_learning.maze import Maze, reachable_graph
from audit_frontier_efficiency import reconstruct, encode, priority_stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--geometry', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists(): p.error('Output already exists')
    seed_all(0, 1); results = []
    for row in json.loads((a.geometry/'results.json').read_text()):
        t = Trainer.load(row['checkpoint'])
        with np.load(Path(row['source'])/'replay.npz') as data:
            qo, qp, vo, vp = reconstruct(data, row['prefix'])
        points, adj, index = reachable_graph(Maze(t.config.maze))
        ids = lambda ps: np.array([index[tuple(np.round(p, 5))] for p in ps])
        qi, vi = ids(qp), ids(vp)
        stored = np.load(a.geometry/f'{row["kind"]}-seed{row["seed"]}-at{row["prefix"]}.npz')
        seen = stored['counts'] > 0; expanded = np.zeros(len(points), bool); expanded[vi] = True
        productive = (~seen[adj]).any(1)
        q, v = encode(t, qo).astype('float64'), encode(t, vo).astype('float64')
        center = v.mean(0)
        _, s, axes = np.linalg.svd(v-center, full_matrices=False)
        sd = s/np.sqrt(len(v))
        # Numerical floor only, fixed before examining outcomes; no tuning.
        sd = np.maximum(sd, 1e-12)
        transforms = {'raw': np.eye(q.shape[1]), 'scale_x2': 2*np.eye(q.shape[1]),
                      'top2': axes[:2].T, 'top2_whitened': axes[:2].T/sd[:2],
                      'all8_whitened': axes.T/sd}
        stats = {}
        for name, transform in transforms.items():
            scores = cKDTree((v-center)@transform).query((q-center)@transform)[0]
            if name == 'raw':
                original = scores
                np.testing.assert_allclose(scores, stored['scores'], rtol=1e-10, atol=1e-12)
            if name == 'scale_x2': np.testing.assert_allclose(scores, 2*original, rtol=1e-10, atol=1e-12)
            stats[name] = priority_stats(scores, qi, productive, expanded)
        result = {k: row[k] for k in ['kind', 'seed', 'prefix']}
        result.update(visited_axis_std=sd.tolist(), modes=stats)
        results.append(result)
        print(row['prefix'], row['seed'], row['kind'], {k: round(v['top100_productive_fraction'], 2) for k, v in stats.items()}, flush=True)
    a.out.write_text(json.dumps(results, indent=2)+'\n')


if __name__ == '__main__': main()
