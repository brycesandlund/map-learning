"""Online training schedules and isolated before/after learning diagnostics."""
import numpy as np
import torch
from .maze import Maze, reachable_graph
from .probabilistic import ensemble_js, mixture_log_prob, mixture_mean, mixture_variance


class TrainingSchedule:
    def __init__(self, trainer):
        self.trainer = trainer
        self.trained_through = 0
        # Shuffling cannot perturb legacy replay sampling or the planner RNG.
        self.rng = np.random.default_rng(trainer.config.seed+61000)

    def train(self, replay):
        cfg = self.trainer.config
        before = self.trainer.updates
        start = 0 if cfg.training_schedule == 'epoch' else self.trained_through
        explicit_samples = 0
        if cfg.training_schedule != 'replay':
            orders = [self.rng.permutation(np.arange(start, len(replay)))
                      for _ in self.trainer.dynamics.heads]
            if cfg.train.ensemble_batches == "shared":
                orders = [orders[0]] * len(orders)
            for offset in range(0, len(replay)-start, cfg.train.batch_size):
                batches = [order[offset:offset+cfg.train.batch_size] for order in orders]
                metrics = self.trainer.update_probabilistic(replay, head_indices=batches)
            explicit_samples = len(replay)-start
            self.trained_through = len(replay)
        if cfg.training_schedule != 'epoch':
            for _ in range(cfg.updates_per_train):
                metrics = self.trainer.update(replay)
        metrics.update(block_updates=self.trainer.updates-before,
                       explicit_transitions_per_head=explicit_samples,
                       guaranteed_trained_through=self.trained_through)
        return metrics


class CadenceDiagnostics:
    """Privileged evaluation probes never enter replay, training, or action search.

    Fixed maze-wide probes can overlap experienced edges; they are NOT a disjoint
    held-out training set. Final evaluation reports seen/unseen edges separately.
    All randomness is isolated, with common MC seeds for each pre/post pair.
    """
    def __init__(self, trainer):
        self.trainer = trainer
        cfg = trainer.config
        self.rng = np.random.default_rng(cfg.seed+62000)
        env = Maze(cfg.maze, cfg.seed+63000)
        points, adjacency, _ = reachable_graph(env)
        indices = self.rng.integers(len(points), size=128)
        actions = self.rng.integers(5, size=128)
        self.probe = (np.stack([env.observe(p) for p in points[indices]]), actions,
                      np.stack([env.observe(p) for p in points[adjacency[indices, actions]]]))

    def batches(self, replay):
        start = max(0, len(replay)-self.trainer.config.train_every)
        batches = {'new': replay.transitions_at(np.arange(start, len(replay))), 'probe': self.probe}
        if start:
            batches['old'] = replay.transitions_at(self.rng.integers(start, size=128))
        return batches

    @torch.no_grad()
    def measure(self, batches, step):
        trainer = self.trainer
        reference = trainer.model.encode(trainer.tensor(np.concatenate([self.probe[0], self.probe[2]])))
        variance = float(reference.var(0, unbiased=False).mean())
        mse_mode = trainer.config.train.prediction_objective == "mse"
        result = {'probe_latent_variance': variance}
        for label, (obs, actions, nxt) in batches.items():
            z, target = trainer.model.encode(trainer.tensor(np.concatenate([obs, nxt]))).chunk(2)
            params = trainer.dynamics.distribution(z, torch.as_tensor(actions, device=trainer.device))
            generator = torch.Generator(device=trainer.device).manual_seed(trainer.config.seed+64000+step)
            error = float((mixture_mean(params).mean(0)-target).square().mean())
            result[label] = {
                'n': len(actions),
                'disagreement': (float(mixture_mean(params).var(0, unbiased=False).mean())/max(variance, 1e-8)
                                 if mse_mode else float(ensemble_js(params, 8, generator).mean())),
                'nll_per_dim': None if mse_mode else float(-mixture_log_prob(params, target).mean()/z.shape[-1]),
                'mse': error, 'normalized_mse': error/max(variance, 1e-8),
                'persistence_normalized_mse': float((z-target).square().mean())/max(variance, 1e-8),
                'predictive_variance': None if mse_mode else float(mixture_variance(params).mean()),
            }
        return result
