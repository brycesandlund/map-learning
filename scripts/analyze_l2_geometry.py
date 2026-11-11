"""Shared-replay geometry and full-dimensional queue diagnostics; no training."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import rankdata

from map_learning.training import Trainer, seed_all
from map_learning.maze import Maze, reachable_graph
from audit_frontier_efficiency import reconstruct, encode, priority_stats


KINDS = ['baseline', 'kl', 'l2']


def model_path(kind, seed):
    if kind == 'l2':
        return Path(f'runs/l2-noise-fixed-20261110/seed{seed}')
    arm = 'mean_zero' if kind == 'baseline' else 'mean_kl'
    return Path(f'runs/kl-mechanisms-fixed-20261110/zero-{arm}-seed{seed}')


def quantiles(x):
    return np.quantile(x, [.1, .5, .9]).tolist()


def auc(positive, negative):
    ranks = rankdata(np.r_[positive, negative])
    n, m = len(positive), len(negative)
    return float((ranks[:n].sum() - n*(n+1)/2) / (n*m))


def analyze(kind, seed, prefix, out):
    t = Trainer.load(model_path(kind, seed)/f'checkpoint-{prefix//5000}.pt')
    source = Path(f'runs/moment-prior-confirm-controls-20261110/learned-rep0.01-beta0-seed{seed-3}')
    with np.load(source/'replay.npz') as data:
        queue_obs, queue_pos, visited_obs, visited_pos = reconstruct(data, prefix)
        # Count each acquired observation once; replay repeats a restored source
        # observation across its five transitions, which must not count as five draws.
        acquired_obs = np.concatenate([data['observations'][:1], data['next_observations'][:prefix]])
        acquired_pos = np.concatenate([data['positions'][:1], data['next_positions'][:prefix]])
    points, adj, index = reachable_graph(Maze(t.config.maze))
    ids = lambda x: np.array([index[tuple(np.round(p, 5))] for p in x])
    acquired_ids, queue_ids, visited_ids = map(ids, [acquired_pos, queue_pos, visited_pos])
    z = encode(t, acquired_obs).astype('float64')
    n, dim = len(points), z.shape[1]
    counts = np.bincount(acquired_ids, minlength=n)
    seen = counts > 0
    centroids = np.zeros((n, dim))
    np.add.at(centroids, acquired_ids, z)
    centroids /= np.maximum(counts[:, None], 1)
    residuals = z-centroids[acquired_ids]
    residual_sq = np.bincount(acquired_ids, weights=(residuals**2).sum(1), minlength=n)
    noise_rms = np.sqrt(residual_sq/np.maximum(counts, 1))

    # Equal weighting of observed physical positions prevents visitation frequency
    # from dominating the display. These labels never enter learning or search.
    center = centroids[seen].mean(0)
    _, singular, axes = np.linalg.svd(centroids[seen]-center, full_matrices=False)
    spectrum = singular**2 / (singular**2).sum()
    scale = np.sqrt(np.mean(np.sum((centroids[seen]-center)**2, axis=1)))
    for j in range(len(axes)):
        if axes[j, np.argmax(np.abs(axes[j]))] < 0:
            axes[j] *= -1
    pc = (centroids-center) @ axes[:3].T / scale
    rng = np.random.default_rng(810100+seed+prefix)
    selected = np.concatenate([rng.choice(np.flatnonzero(acquired_ids == i), min(counts[i], 12), replace=False)
                               for i in np.flatnonzero(seen)])
    cloud_pc = (z[selected]-center) @ axes[:3].T / scale

    # Local stretch in two open orthogonal directions, using actual observation
    # centroids with >=8 samples per location. No noise resampling is used here.
    valid = (counts >= 8) & (adj[:, 0] != np.arange(n)) & (adj[:, 2] != np.arange(n))
    valid &= (counts[adj[:, 0]] >= 8) & (counts[adj[:, 2]] >= 8)
    local_ids = np.flatnonzero(valid)
    jac = np.stack([centroids[adj[valid, 0]]-centroids[valid],
                    centroids[adj[valid, 2]]-centroids[valid]], axis=-1)
    u, stretch, _ = np.linalg.svd(jac, full_matrices=False)
    anisotropy = stretch[:, 0] / np.maximum(stretch[:, 1], 1e-15)
    weak_directions = np.zeros_like(centroids)
    weak_directions[valid] = u[:, :, 1]
    weak_residual_sq = np.sum(residuals*weak_directions[acquired_ids], axis=1)**2
    weak_noise = np.sqrt(np.bincount(acquired_ids, weights=weak_residual_sq, minlength=n) / np.maximum(counts, 1))
    weak_ratio = weak_noise[valid] / np.maximum(stretch[:, 1], 1e-15)

    edges = np.array([(i, j) for i in np.flatnonzero(seen) for j in adj[i, :4] if i < j and seen[j]])
    edge_dist = np.linalg.norm(centroids[edges[:, 0]]-centroids[edges[:, 1]], axis=1)
    observed = np.flatnonzero(seen)
    nn = cKDTree(centroids[seen]).query(centroids[seen], k=2)[1][:, 1]
    nearest_centroid_steps = np.linalg.norm(points[observed]-points[observed[nn]], axis=1)/t.config.maze.step_size

    qz, vz = encode(t, queue_obs), encode(t, visited_obs)
    scores, nearest = cKDTree(vz).query(qz)
    expanded = np.zeros(n, bool)
    expanded[visited_ids] = True
    productive = (~seen[adj]).any(1)
    repeated_entry = expanded[queue_ids]
    productive_entry = productive[queue_ids]
    wrong_steps = np.linalg.norm(queue_pos-visited_pos[nearest], axis=1)/t.config.maze.step_size
    correct_repeat = queue_ids[repeated_entry] == visited_ids[nearest][repeated_entry]
    order = np.argsort(-scores, kind='stable')
    stats = priority_stats(scores, queue_ids, productive, expanded)
    stats.update(
        productive_vs_repeat_auc=auc(scores[productive_entry], scores[repeated_entry]),
        productive_median_over_repeat_median=float(np.median(scores[productive_entry])/np.median(scores[repeated_entry])),
        repeated_wrong_nearest_fraction=float(1-correct_repeat.mean()),
        repeated_nearest_more_than_two_grid_steps=float((wrong_steps[repeated_entry] > 2+1e-6).mean()),
        wrong_repeat_physical_steps=quantiles(wrong_steps[repeated_entry & (wrong_steps > 0)]),
    )
    row = dict(kind=kind, seed=seed, prefix=prefix, source=str(source),
               checkpoint=str(model_path(kind, seed)/f'checkpoint-{prefix//5000}.pt'),
               acquired_observations=len(z), observed_positions=int(seen.sum()),
               pca_spectrum=spectrum.tolist(), pca3_retained=float(spectrum[:3].sum()),
               rms_centroid_scale=float(scale), participation_dimension=float(1/(spectrum**2).sum()),
               local_probe_positions=len(local_ids), local_anisotropy=quantiles(anisotropy),
               local_weak_direction_noise_ratio=quantiles(weak_ratio),
               adjacent_distance=quantiles(edge_dist),
               adjacent_p90_over_p10=float(np.quantile(edge_dist, .9)/np.quantile(edge_dist, .1)),
               centroid_nearest_more_than_two_grid_steps=float((nearest_centroid_steps > 2+1e-6).mean()),
               queue=stats)
    np.savez_compressed(out/f'{kind}-seed{seed}-at{prefix}.npz',
                        points=points, adjacency=adj, centroids=centroids, counts=counts,
                        pc=pc, cloud_pc=cloud_pc, cloud_ids=acquired_ids[selected],
                        spectrum=spectrum, center=center, axes=axes, scale=scale, edges=edges,
                        noise_rms=noise_rms, local_ids=local_ids, anisotropy=anisotropy,
                        weak_ratio=weak_ratio, stretch=stretch,
                        queue_ids=queue_ids, scores=scores, expanded=expanded,
                        productive=productive, nearest_ids=visited_ids[nearest], top100=order[:100])
    print(json.dumps(row), flush=True)
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(exist_ok=False, parents=True)
    seed_all(0, 1)
    rows = []
    for prefix in [50000, 75000]:
        for seed in [9, 10, 11]:
            for kind in KINDS:
                rows.append(analyze(kind, seed, prefix, a.out))
                (a.out/'results.json').write_text(json.dumps(rows, indent=2)+'\n')


if __name__ == '__main__':
    main()
