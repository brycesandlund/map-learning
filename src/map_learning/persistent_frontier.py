"""Persistent observation frontier with exact latent nearest-neighbor priorities."""
import numpy as np
from scipy.spatial import cKDTree


class PersistentDistanceFrontier:
    def __init__(self, capacity, dim, observation_shape, rng):
        self.latents = np.empty((capacity, dim), dtype=np.float64)
        self.observations = np.empty((capacity, *observation_shape), dtype=np.float32)
        self.priority = np.full(capacity, -np.inf)
        self.active = np.zeros(capacity, dtype=bool)
        self.visited = np.empty((capacity, dim), dtype=np.float64)
        self.visited_observations = np.empty_like(self.observations)
        self.n_visited = 0
        self.entries = []
        self.rng = rng
        self.tree = None
        self.indexed_through = 0

    def visit(self, obs, latent):
        latent = np.asarray(latent, dtype=np.float64)
        self.visited[self.n_visited] = latent
        self.visited_observations[self.n_visited] = obs
        self.n_visited += 1
        ids = np.flatnonzero(self.active)
        self.priority[ids] = np.minimum(self.priority[ids], np.linalg.norm(self.latents[ids]-latent, axis=1))

    def add(self, obs, latent, payload):
        i = len(self.entries)
        if i >= len(self.priority): raise OverflowError('Frontier capacity exhausted')
        latent = np.asarray(latent, dtype=np.float64)
        score = self.tree.query(latent, eps=0)[0] if self.tree is not None else np.inf
        recent = self.visited[self.indexed_through:self.n_visited]
        if len(recent): score = min(score, np.linalg.norm(recent-latent, axis=1).min())
        if not np.isfinite(score): raise ValueError('Visit the root before adding successors')
        self.entries.append(payload)
        self.observations[i] = obs
        self.latents[i] = latent
        self.priority[i] = score
        self.active[i] = True

    def pop(self):
        ids = np.flatnonzero(self.active)
        if not len(ids): raise IndexError('Empty frontier')
        best = self.priority[ids].max()
        i = int(self.rng.choice(ids[self.priority[ids] == best]))
        self.active[i] = False
        return self.observations[i].copy(), self.entries[i], self.latents[i].copy(), float(best)

    def refresh(self, encode, batch_size=1024):
        """Re-encode raw observations and rebuild ALL priorities, without clearing search."""
        for start in range(0, self.n_visited, batch_size):
            end = min(start+batch_size, self.n_visited)
            self.visited[start:end] = encode(self.visited_observations[start:end])
        self.tree = cKDTree(self.visited[:self.n_visited].copy())
        self.indexed_through = self.n_visited
        ids = np.flatnonzero(self.active)
        for start in range(0, len(ids), batch_size):
            batch = ids[start:start+batch_size]
            self.latents[batch] = encode(self.observations[batch])
            self.priority[batch] = self.tree.query(self.latents[batch], eps=0, workers=1)[0]


def epoch_slices(old_count, new_start, new_end, interval_index, intervals, rng, old_order, block_total, batch_size):
    """Causal slices of one full epoch with matched total samples and updates.

    Partition old data once per block. Add all new data from each interval. The
    ceil(block_total/batch_size) updates are apportioned across intervals; splitting
    each interval's samples evenly entails slightly variable minibatch sizes.
    """
    old = old_order[old_count*interval_index//intervals:old_count*(interval_index+1)//intervals]
    indices = rng.permutation(np.concatenate([old, np.arange(new_start, new_end)]))
    total_updates = (block_total+batch_size-1)//batch_size
    n_updates = total_updates*(interval_index+1)//intervals - total_updates*interval_index//intervals
    if n_updates < 1 or n_updates > len(indices): raise ValueError('Budget cannot support cadence')
    return np.array_split(indices, n_updates)
