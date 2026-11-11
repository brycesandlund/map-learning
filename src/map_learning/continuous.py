"""Continuous controls and evaluation-only spatial accounting.

The learner/frontier never receives coverage bins or physical state identities.
All five children are real simulator transitions, including rejected wall moves.
"""
import numpy as np
from scipy.spatial import cKDTree


def sample_controls(rng, mode='stratified'):
    """Four real-valued moves and one stay; lengths uniform in [.5,1].

    Stratified uses four perpendicular directions with a fresh random rotation.
    IID is an ablation: four independent directions. Neither snaps coordinates.
    The sampled control is observed by dynamics, so this adds no hidden noise.
    """
    if mode == 'stratified':
        angles = rng.uniform(-np.pi/4, np.pi/4) + np.arange(4)*np.pi/2
    elif mode == 'iid':
        angles = rng.uniform(-np.pi, np.pi, 4)
    else:
        raise ValueError('Unknown continuous action proposal')
    lengths = rng.uniform(.5, 1., 4)
    controls = np.zeros((5, 2), dtype=np.float32)
    controls[:4] = lengths[:, None]*np.stack([np.cos(angles), np.sin(angles)], -1)
    return controls


class SpatialCoverage:
    """Endpoint occupancy at four declared resolutions; NOT a visited-set key."""
    def __init__(self, size):
        self.size = size
        self.masks = {k: np.zeros((size*k, size*k), dtype=bool) for k in (1, 2, 4, 8)}
        self.counts = {k: 0 for k in self.masks}
        self.milestones = {}

    def add(self, position, interactions):
        p = np.asarray(position)
        if p.shape != (2,) or np.any(p < 0) or np.any(p >= self.size):
            raise ValueError('Invalid coverage point')
        for k, mask in self.masks.items():
            x, y = np.floor(p*k).astype(int)
            if not mask[y, x]:
                mask[y, x] = True
                self.counts[k] += 1
                for percent in (50, 90, 95, 99, 100):
                    key = f'k{k}_{percent}'
                    if self.counts[k] >= percent/100*mask.size:
                        self.milestones.setdefault(key, interactions)

    def metrics(self):
        return {f'coverage_k{k}': self.counts[k]/mask.size for k, mask in self.masks.items()}


def within_cell_fill(positions, size, subdivisions=8):
    """Distance to visited endpoints from a fixed grid of evaluation probes.

    Restrict matches to the same unit room to avoid credit through a wall. This
    is conservative at open room boundaries, not a geodesic-distance estimate.
    Zero-thickness walls leave every room interior free/reachable in this maze.
    """
    positions = np.asarray(positions)
    cell_ids = np.floor(positions).astype(int)
    offsets = (np.arange(subdivisions)+.5)/subdivisions
    offsets = np.stack(np.meshgrid(offsets, offsets), -1).reshape(-1, 2)
    distances = []
    for x in range(size):
        for y in range(size):
            samples = positions[np.all(cell_ids == [x, y], axis=1)]
            distances.extend(cKDTree(samples).query(offsets+[x, y])[0] if len(samples)
                             else np.full(len(offsets), np.inf))
    d = np.asarray(distances)
    return {f'fill_radius_{r:g}': float(np.mean(d <= r)) for r in (.05, .1, .2)}
