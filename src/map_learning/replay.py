"""Append-only replay; all raw observations remain available for re-encoding."""
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np


class Replay:
    def __init__(self):
        self.observations = []
        self.next_observations = []
        self.actions = []
        self.episodes = []
        self.positions = []  # diagnostics only, never sampled by training
        self.next_positions = []
        self._arrays = None
        self._pair_cache = {}

    def __len__(self):
        return len(self.actions)

    def add(self, obs, action, next_obs, episode, position, next_position):
        if np.isscalar(action):
            stored_action = int(action)
        else:
            stored_action = np.array(action, dtype=np.float32, copy=True)
            if stored_action.shape != (2,) or not np.isfinite(stored_action).all():
                raise ValueError("Continuous replay actions must be finite 2D vectors")
        self.observations.append(np.array(obs, copy=True))
        self.next_observations.append(np.array(next_obs, copy=True))
        self.actions.append(stored_action)
        self.episodes.append(int(episode))
        self.positions.append(np.array(position, copy=True))
        self.next_positions.append(np.array(next_position, copy=True))
        self._arrays = None
        self._pair_cache.clear()

    def arrays(self):
        if self._arrays is None:
            self._arrays = {name: np.asarray(getattr(self, name)) for name in
                           ["observations", "next_observations", "actions", "episodes",
                            "positions", "next_positions"]}
        return self._arrays

    def sample_transitions(self, batch, rng):
        idx = rng.integers(len(self), size=batch)
        return self.transitions_at(idx)

    def transitions_at(self, idx):
        """Read explicit transition indices, including a partial final batch."""
        if self._arrays is not None:
            data = self._arrays
            return data["observations"][idx], data["actions"][idx], data["next_observations"][idx]
        # Online appends invalidate arrays(). Materializing the full replay on
        # every training event makes a long run quadratic in collected steps.
        return (np.asarray([self.observations[i] for i in idx]),
                np.asarray([self.actions[i] for i in idx]),
                np.asarray([self.next_observations[i] for i in idx]))

    def sample_states(self, batch, rng):
        # Include terminal next states as well as transition sources.
        idx = rng.integers(2 * len(self), size=batch)
        if self._arrays is None:
            n = len(self)
            return np.asarray([self.observations[i] if i < n else self.next_observations[i-n]
                               for i in idx])
        data = self._arrays
        return np.where((idx < len(self)).reshape((-1,) + (1,) * (data["observations"].ndim-1)),
                        data["observations"][idx % len(self)], data["next_observations"][idx % len(self)])

    def _pair_index(self, max_steps, exact_min):
        cache_key = (max_steps, exact_min)
        if cache_key in self._pair_cache:
            return self._pair_cache[cache_key]
        data = self.arrays()
        episode = data["episodes"]
        remaining = np.ones(len(self), dtype=np.int64)
        for i in range(len(self)-2, -1, -1):
            remaining[i] = remaining[i+1]+1 if episode[i] == episode[i+1] else 1
        remaining = np.minimum(remaining, max_steps)
        minimum = {}
        if exact_min:
            # Exact observed-byte equality only: no position labels or oracle merges.
            obs_keys = [x.tobytes() for x in data["observations"]]
            next_keys = [x.tobytes() for x in data["next_observations"]]
            for i in range(len(self)):
                minimum[(obs_keys[i], obs_keys[i])] = 0
                for k in range(1, remaining[i]+1):
                    pair = (obs_keys[i], next_keys[i+k-1])
                    minimum[pair] = min(k, minimum.get(pair, k))
        result = remaining, minimum
        self._pair_cache[cache_key] = result
        return result

    def sample_pairs(self, batch, rng, horizon, max_steps, exact_min=False):
        data = self.arrays()
        remaining, minimum = self._pair_index(max_steps, exact_min)
        idx = rng.integers(len(self), size=batch)
        # Half the batch is near; half spans the full window (including far pairs).
        limit = np.where(np.arange(batch) < batch//2, np.minimum(remaining[idx], horizon), remaining[idx])
        gaps = np.floor(rng.random(batch) * (limit+1)).astype(int)
        target_idx = idx + np.maximum(gaps, 1) - 1
        x = data["observations"][idx]
        y = data["next_observations"][target_idx].copy()
        y[gaps == 0] = x[gaps == 0]
        if exact_min:
            gaps = np.array([minimum[(a.tobytes(), b.tobytes())] for a, b in zip(x, y)])
        return x, y, gaps

    def save(self, path, maze_config):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite replay: {path}")
        np.savez_compressed(path, **self.arrays(), maze_config=json.dumps(asdict(maze_config)))

    @classmethod
    def load(cls, path):
        result = cls()
        with np.load(path, allow_pickle=False) as data:
            for name in ["observations", "next_observations", "actions", "episodes", "positions", "next_positions"]:
                setattr(result, name, list(data[name]))
            maze_config = json.loads(str(data["maze_config"]))
        return result, maze_config
