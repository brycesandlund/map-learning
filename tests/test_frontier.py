import numpy as np
import pytest
from map_learning.frontier import LatentFrontier
from map_learning.maze import Maze
from map_learning.config import MazeConfig


def test_distance_keys_decrease_and_reorder():
    q = LatentFrontier('distance', 10, 2, np.random.default_rng(0))
    q.visit(np.array([0, 0]))
    q.add(np.array([10, 0]), 'far', 0)
    q.add(np.array([6, 0]), 'middle', 0)
    q.visit(np.array([10, 0]))
    payload, _, score = q.pop()
    assert payload == 'middle' and score == 4
    assert q.pop()[0] == 'far'


def test_disagreement_is_incoming_score_and_retains_duplicates():
    q = LatentFrontier('disagreement', 10, 2, np.random.default_rng(0))
    q.visit(np.zeros(2))
    q.add(np.zeros(2), 'same_a', .2)
    q.add(np.zeros(2), 'same_b', .3)
    q.visit(np.ones(2))
    assert q.pop()[0] == 'same_b'
    assert q.pop()[0] == 'same_a'
    with pytest.raises(IndexError): q.pop()


def test_noisy_maze_preserves_physics_changes_observation():
    noisy = Maze(MazeConfig(nuisance_dims=8), 1)
    clean = Maze(MazeConfig(), 1)
    first = noisy.observe()
    second = noisy.observe()
    assert first.shape == (10,)
    assert np.array_equal(first[:2], second[:2])
    assert not np.array_equal(first[2:], second[2:])
    assert np.array_equal(first[:2], clean.observe())
    for action in range(5):
        assert np.array_equal(noisy.transition(noisy.start, action), clean.transition(clean.start, action))


def test_distance_queue_matches_brute_force():
    rng = np.random.default_rng(4)
    q = LatentFrontier('distance', 100, 8, rng)
    visited = [rng.normal(size=8).astype(np.float32)]
    q.visit(visited[0])
    z = rng.normal(size=(30, 8)).astype(np.float32)
    active = set(range(30))
    for i in active: q.add(z[i], i, 0)
    for _ in range(30):
        values = {i: min(np.linalg.norm(z[i]-v) for v in visited) for i in active}
        i, latent, score = q.pop()
        assert score == pytest.approx(max(values.values()))
        active.remove(i); visited.append(latent); q.visit(latent)
