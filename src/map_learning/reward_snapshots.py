"""Read-only, pre-action visualization probes. Never inputs to learning/search."""
from pathlib import Path
import numpy as np
import torch
from .maze import Maze, reachable_graph
from .probabilistic import mixture_mean


@torch.no_grad()
def mean_reward(trainer, z, actions, scale):
    predictions = (mixture_mean(trainer.dynamics.distribution(z, actions))
                   if hasattr(trainer.dynamics, 'distribution') else trainer.dynamics(z, actions))
    return predictions.var(0, unbiased=False).mean(-1)/scale, predictions.mean(0)


@torch.no_grad()
def exhaustive_q(trainer, z, horizon, discount, scale):
    """Exact finite-horizon values in the deterministic learned model, not the maze."""
    actions = torch.arange(trainer.dynamics.actions, device=z.device).repeat(len(z))
    expanded = z.repeat_interleave(trainer.dynamics.actions, dim=0)
    reward, successor = mean_reward(trainer, expanded, actions, scale)
    if horizon > 1:
        reward = reward + discount*exhaustive_q(trainer, successor, horizon-1, discount, scale).max(-1).values
    return reward.reshape(len(z), trainer.dynamics.actions)


class RewardSnapshots:
    def __init__(self, trainer):
        self.trainer = trainer
        self.every = trainer.config.reward_snapshot_every
        env = Maze(trainer.config.maze, trainer.config.seed+71000)
        self.points, _, _ = reachable_graph(env)
        self.observations = np.stack([env.observe(p) for p in self.points])
        self.rows = []

    @torch.no_grad()
    def record(self, step, position, observation, action, scale, planner_metrics, ready):
        if step % self.every:
            return
        trainer, cfg = self.trainer, self.trainer.config
        normalization = max(float(scale), 1e-8) if scale is not None else 1.0
        rewards = []
        for start in range(0, len(self.points), 128):
            z = trainer.model.encode(trainer.tensor(self.observations[start:start+128]))
            actions = torch.arange(5, device=trainer.device).repeat(len(z))
            score, _ = mean_reward(trainer, z.repeat_interleave(5, dim=0), actions, normalization)
            rewards.append(score.reshape(-1, 5).cpu().numpy())
        z = trainer.model.encode(trainer.tensor(observation[None]))
        immediate, _ = mean_reward(trainer, z.expand(5, -1), torch.arange(5, device=trainer.device), normalization)
        exact = (exhaustive_q(trainer, z, cfg.planner.horizon, cfg.planner.uncertainty_discount, normalization)[0].cpu().numpy()
                 if cfg.planner.horizon <= 5 else np.full(5, np.nan))
        decision = ('warmup' if not ready else 'random' if cfg.planner.behavior == 'random' else
                    'mcts' if 'root_visits' in planner_metrics else
                    'shooting' if 'search_return' in planner_metrics else 'epsilon')
        self.rows.append(dict(step=step, position=np.array(position).copy(), action=action,
            scale=normalization, updates=trainer.updates, decision=decision,
            reward=np.concatenate(rewards), immediate=immediate.cpu().numpy(), exact_q=exact,
            search_q=np.asarray(planner_metrics.get('root_q', [np.nan]*5)),
            root_visits=np.asarray(planner_metrics.get('root_visits', [0]*5))))

    def save(self, path):
        if not self.rows:
            return
        fields = {key: np.asarray([r[key] for r in self.rows]) for key in self.rows[0]}
        np.savez_compressed(Path(path)/'reward_snapshots.npz', points=self.points, **fields)
