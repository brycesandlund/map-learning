"""Exact maximum-priority frontier over observation entries, never physical IDs.

A vector-backed priority queue allows eager key decreases when the visited set
changes. No merging, deduplication, oracle coordinates, or latent threshold.
"""
import numpy as np


class LatentFrontier:
    def __init__(self, mode, capacity, dim, rng):
        if mode not in {'distance', 'disagreement'}:
            raise ValueError(mode)
        self.mode, self.rng = mode, rng
        self.latents = np.empty((capacity, dim), dtype=np.float32)
        self.priority = np.full(capacity, -np.inf)
        self.active = np.zeros(capacity, dtype=bool)
        self.visited = np.empty((capacity, dim), dtype=np.float32)
        self.n_visited = 0
        self.entries = []

    def visit(self, latent):
        latent = np.asarray(latent, dtype=np.float32)
        self.visited[self.n_visited] = latent
        self.n_visited += 1
        if self.mode == 'distance':
            ids = np.flatnonzero(self.active)
            distance = np.linalg.norm(self.latents[ids] - latent, axis=1)
            self.priority[ids] = np.minimum(self.priority[ids], distance)

    def add(self, latent, payload, disagreement):
        i = len(self.entries)
        if i >= len(self.priority):
            raise OverflowError('Frontier capacity exhausted')
        self.entries.append(payload)
        self.latents[i] = latent
        value = (np.linalg.norm(self.visited[:self.n_visited] - latent, axis=1).min()
                 if self.mode == 'distance' else disagreement)
        if not np.isfinite(value):
            raise FloatingPointError('Nonfinite frontier score')
        self.priority[i] = value
        self.active[i] = True
        return i

    def pop(self):
        ids = np.flatnonzero(self.active)
        if not len(ids):
            raise IndexError('Empty frontier')
        best = self.priority[ids].max()
        i = int(self.rng.choice(ids[self.priority[ids] == best]))
        self.active[i] = False
        return self.entries[i], self.latents[i].copy(), float(best)
