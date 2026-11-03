from types import SimpleNamespace
import numpy as np
import torch
from map_learning.config import PlannerConfig
from map_learning.planning import LatentPlanner


class IdentityMap:
    bottleneck = None
    def encode(self, x):
        return x
    def distance(self, x, y):
        return (y-x).abs().sum(-1)


class KnownDynamics:
    heads = [None, None]
    def predict_head(self, index, z, actions):
        offsets = torch.tensor([[1., 0.], [-1., 0.], [0., 1.], [0., -1.], [0., 0.]])
        return z+offsets[actions]


def test_planner_reaches_goal_with_known_dynamics():
    trainer = SimpleNamespace(model=IdentityMap(), dynamics=KnownDynamics(), device="cpu",
                              tensor=lambda x: torch.as_tensor(x, dtype=torch.float32))
    planner = LatentPlanner(trainer, PlannerConfig(behavior="distance", horizon=3, beam_width=30, epsilon_random=0))
    position = np.array([0., 0.])
    goal = np.array([3., 0.])
    action = planner.act(position, position, goal_observation=goal)
    assert action == 0


def test_disagreement_search_selects_uncertain_action():
    class UncertainDynamics(KnownDynamics):
        def predict_head(self, index, z, actions):
            return z + (actions == 2)[:, None] * torch.tensor([[0., float(index)]])
    trainer = SimpleNamespace(model=IdentityMap(), dynamics=UncertainDynamics(), device="cpu",
                              tensor=lambda x: torch.as_tensor(x, dtype=torch.float32))
    planner = LatentPlanner(trainer, PlannerConfig(behavior="disagreement", horizon=2, beam_width=20, epsilon_random=0))
    assert planner.act(np.zeros(2), np.zeros(2)) == 2
