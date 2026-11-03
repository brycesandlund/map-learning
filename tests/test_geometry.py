import numpy as np
import torch
from map_learning.config import MazeConfig
from map_learning.maze import Maze, reachable_graph, bfs
from map_learning.models import IntervalQuasimetric


def test_collision_checks_whole_segment_and_endpoints():
    env = Maze(MazeConfig(size=3, step_size=.9))
    env.walls = np.array([[[1, 0], [1, 2]]], dtype=np.float32)
    p = np.array([.6, 1.5], dtype=np.float32)
    np.testing.assert_array_equal(env.transition(p, [1, 0]), p)
    np.testing.assert_array_equal(env.transition(p, [1, .5]), p)
    np.testing.assert_allclose(env.transition([.6, 2.5], [1, 0]), [1.5, 2.5])
    # Collinear entry through an endpoint must also be rejected.
    p = np.array([1., 2.5], dtype=np.float32)
    np.testing.assert_array_equal(env.transition(p, [0, -1]), p)


def test_reachable_positions_never_touch_any_wall():
    env = Maze(MazeConfig(size=5, step_size=.25))
    points, _, _ = reachable_graph(env)
    for a, b in env.walls:
        delta = b-a
        t = ((points-a)*delta).sum(-1)/(delta**2).sum()
        separation = np.linalg.norm(points-(a+t[:, None]*delta), axis=-1)
        assert not np.any((t >= 0) & (t <= 1) & (separation < 1e-6))


def test_seeded_maze_connected_and_has_cycles():
    config = MazeConfig(size=5, step_size=.5, loops=.5)
    env = Maze(config)
    np.testing.assert_array_equal(env.walls, Maze(config).walls)
    points, adjacency, _ = reachable_graph(env)
    assert len({tuple(np.floor(p).astype(int)) for p in points}) == 25
    assert np.all(bfs(adjacency, 0) >= 0)
    # Internal doors beyond a spanning tree, rather than a perfect/tree maze.
    possible_walls = 2*config.size*(config.size+1)
    assert possible_walls-len(env.walls) > config.size**2-1


def test_iqe_union_orientation_and_zero():
    d = IntervalQuasimetric(2, 2)
    x = torch.tensor([[0., 1.]])
    y = torch.tensor([[2., 3.]])
    assert d(x, y).item() == 3  # union [0,2] U [1,3]
    assert d(y, x).item() == 0
    assert d(x, x).item() == 0


def test_iqe_triangle_broadcast_and_gradients():
    torch.manual_seed(5)
    d = IntervalQuasimetric(32, 8).double()
    x, y, z = [torch.randn(1000, 32, dtype=torch.double) for _ in range(3)]
    assert torch.all(d(x, z) <= d(x, y)+d(y, z)+1e-10)
    assert torch.all(d(x, x) == 0)
    assert d(x[:4, None], y[:3]).shape == (4, 3)
    a, b = x[:2].requires_grad_(), y[:2].requires_grad_()
    assert torch.autograd.gradcheck(d, (a, b))
