"""Continuous point maze with swept wall collisions; no simulator dependency.

The layout is a seeded spanning-tree maze with additional doors, hence cycles.
This is inspired by the LGE figure, not a reconstruction of its exact layout.
Ground-truth positions/walls are used only by the environment and evaluation.
"""
from collections import deque
import numpy as np
from .config import MazeConfig


class Maze:
    ACTIONS = np.array([[1, 0], [-1, 0], [0, 1], [0, -1], [0, 0]], dtype=np.float32)

    def __init__(self, config: MazeConfig, seed=0):
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.walls = self._generate_walls()
        self.start = np.array([config.size // 2 + .5] * 2, dtype=np.float32)
        feature_rng = np.random.default_rng(config.seed + 10000)
        self.frequencies = feature_rng.normal(0, 2, (2, 16))
        self.background = self._raster_background()
        self.position = self.start.copy()
        self.elapsed = 0

    def _generate_walls(self):
        n = self.config.size
        rng = np.random.default_rng(self.config.seed)
        walls = set()
        for x in range(n + 1):
            for y in range(n):
                walls.add((x, y, x, y + 1))
        for y in range(n + 1):
            for x in range(n):
                walls.add((x, y, x + 1, y))

        def dividing(a, b):
            x, y = a
            u, v = b
            if x != u:
                return (max(x, u), y, max(x, u), y + 1)
            return (x, max(y, v), x + 1, max(y, v))

        stack, visited = [(n // 2, n // 2)], {(n // 2, n // 2)}
        while stack:
            x, y = stack[-1]
            neighbors = [(u, v) for u, v in [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
                         if 0 <= u < n and 0 <= v < n and (u, v) not in visited]
            if not neighbors:
                stack.pop()
                continue
            other = neighbors[int(rng.integers(len(neighbors)))]
            walls.remove(dividing((x, y), other))
            visited.add(other)
            stack.append(other)
        for wall in sorted(walls):
            x, y, u, v = wall
            boundary = (x == u and x in (0, n)) or (y == v and y in (0, n))
            if not boundary and rng.random() < self.config.loops:
                walls.remove(wall)
        return np.array(sorted(walls), dtype=np.float32).reshape(-1, 2, 2)

    def transition(self, position, action):
        """Pure physical transition. Accept a discrete index or bounded 2D control."""
        p = np.asarray(position, dtype=np.float32)
        control = self.ACTIONS[action] if np.isscalar(action) else np.asarray(action)
        if control.shape != (2,) or not np.all(np.isfinite(control)):
            raise ValueError("Action must be a finite 2D control or a discrete index")
        q = p + np.clip(control, -1, 1) * self.config.step_size
        if np.any(q <= 0) or np.any(q >= self.config.size):
            return p.copy()
        # Axis-aligned wall intersection, including touching an endpoint. Checking
        # the entire segment prevents tunneling, including for continuous controls.
        for axis in (0, 1):
            fixed = self.walls[:, 0, axis] == self.walls[:, 1, axis]
            segments = self.walls[fixed]
            delta = float(q[axis] - p[axis])
            if abs(delta) < 1e-9:
                # Parallel motion can still ENTER a wall through its endpoint.
                # Check overlap of collinear segments instead of skipping them.
                same_line = np.abs(segments[:, 0, axis] - p[axis]) <= 1e-6
                lo = segments[:, :, 1-axis].min(axis=1)
                hi = segments[:, :, 1-axis].max(axis=1)
                path_lo, path_hi = sorted([p[1-axis], q[1-axis]])
                overlap = (path_hi >= lo-1e-6) & (path_lo <= hi+1e-6)
                if (same_line & overlap).any():
                    return p.copy()
                continue
            t = (segments[:, 0, axis] - p[axis]) / delta
            other = p[1-axis] + t * (q[1-axis] - p[1-axis])
            lo = segments[:, :, 1-axis].min(axis=1)
            hi = segments[:, :, 1-axis].max(axis=1)
            hit = (t >= -1e-6) & (t <= 1 + 1e-6) & (other >= lo-1e-6) & (other <= hi+1e-6)
            if hit.any():
                return p.copy()
        return q.astype(np.float32)

    def _raster_background(self):
        res = self.config.image_size
        canvas = np.ones((res, res), dtype=np.float32)
        for segment in self.walls:
            a, b = np.rint(segment / self.config.size * (res-1)).astype(int)
            canvas[min(a[1], b[1]):max(a[1], b[1])+1,
                   min(a[0], b[0]):max(a[0], b[0])+1] = 0
        return canvas

    @property
    def observation_shape(self):
        if self.config.observation == "pixels":
            return (2, self.config.image_size, self.config.image_size)
        return ((2 if self.config.observation == "coordinates" else 32) + self.config.nuisance_dims,)

    def observe(self, position=None):
        p = self.position if position is None else np.asarray(position)
        normalized = 2 * p / self.config.size - 1
        if self.config.observation == "pixels":
            # Two channels preserve wall information even where the marker overlaps.
            marker = np.zeros_like(self.background)
            x, y = np.rint(p / self.config.size * (self.config.image_size-1)).astype(int)
            marker[max(0,y-1):y+2, max(0,x-1):x+2] = 1
            return np.stack([self.background, marker])
        if self.config.observation == "features":
            phase = normalized @ self.frequencies
            obs = np.concatenate([np.sin(phase), np.cos(phase)])
        else:
            obs = normalized
        if self.config.nuisance_dims:
            obs = np.concatenate([obs, self.rng.normal(size=self.config.nuisance_dims)])
        return obs.astype(np.float32)

    def reset(self):
        self.position = self.start.copy()
        self.elapsed = 0
        return self.observe()

    def step(self, action):
        self.position = self.transition(self.position, action)
        self.elapsed += 1
        return self.observe(), self.elapsed >= self.config.episode_steps

    def draw(self, ax):
        from matplotlib.collections import LineCollection
        ax.add_collection(LineCollection(self.walls, colors="#343a40", linewidths=.8))
        ax.scatter(*self.start, marker="*", s=60, color="#2864dc", zorder=4)
        ax.set(xlim=(0, self.config.size), ylim=(0, self.config.size), aspect="equal")


def reachable_graph(env):
    """Exact finite graph reachable under the five discrete controls (evaluation only)."""
    key = lambda p: tuple(np.round(p, 5))
    points, index, adjacency = [env.start.copy()], {key(env.start): 0}, []
    cursor = 0
    while cursor < len(points):
        row = []
        for action in range(len(env.ACTIONS)):
            p = env.transition(points[cursor], action)
            k = key(p)
            if k not in index:
                index[k] = len(points)
                points.append(p)
            row.append(index[k])
        adjacency.append(row)
        cursor += 1
        if cursor > 100000:
            raise ValueError("Evaluation lattice is too large; use a coarser rational step_size")
    return np.asarray(points), np.asarray(adjacency), index


def bfs(adjacency, start):
    distances = np.full(len(adjacency), -1, dtype=np.int32)
    distances[start] = 0
    queue = deque([start])
    while queue:
        i = queue.popleft()
        for j in adjacency[i]:
            if distances[j] < 0:
                distances[j] = distances[i] + 1
                queue.append(int(j))
    return distances
