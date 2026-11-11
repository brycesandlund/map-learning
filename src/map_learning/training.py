from dataclasses import asdict
import random
import numpy as np
import torch
from torch.nn import functional as F
from .losses import qrl_loss, time_myopic_loss, dual_ascent
from .models import MapModel, DynamicsEnsemble
from .probabilistic import MixtureDynamicsEnsemble, mixture_log_prob, mixture_mean, mixture_variance, repulsion_loss


def seed_all(seed, threads=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)


class Trainer:
    def __init__(self, config, observation_shape):
        self.config = config
        self.device = torch.device(config.device)
        self.observation_shape = observation_shape
        self.rng = np.random.default_rng(config.seed+101)
        self.model = MapModel(observation_shape, config.model, config.train.loss, config.train.horizon).to(self.device)
        dynamics_class = MixtureDynamicsEnsemble if config.train.loss == "probabilistic" else DynamicsEnsemble
        self.dynamics = dynamics_class(config.model).to(self.device)
        self.map_optimizer = torch.optim.Adam(self.model.parameters(), lr=config.train.lr*config.train.encoder_lr_multiplier)
        self.dynamics_optimizer = torch.optim.Adam(self.dynamics.parameters(), lr=config.train.lr)
        self.multiplier = torch.tensor(config.train.initial_multiplier, device=self.device)
        self.updates = 0

    def tensor(self, x):
        return torch.as_tensor(x, dtype=torch.float32, device=self.device)

    def initialize(self, replay):
        bottle = self.model.bottleneck
        if bottle and not bottle.initialized:
            with torch.no_grad():
                samples = replay.sample_states(max(1024, 8*len(bottle.codes)), self.rng)
                bottle.initialize(self.model.encoder(self.tensor(samples)))

    def update(self, replay):
        if self.config.train.loss == "dynamics":
            return self.update_dynamics(replay)
        if self.config.train.loss == "probabilistic":
            return self.update_probabilistic(replay)
        self.initialize(replay)
        c = self.config.train
        obs, actions, nxt = replay.sample_transitions(c.batch_size, self.rng)
        obs, nxt = self.tensor(obs), self.tensor(nxt)
        actions = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        metrics = {}
        if c.loss == "qrl":
            left = self.tensor(replay.sample_states(c.batch_size, self.rng))
            right = self.tensor(replay.sample_states(c.batch_size, self.rng))
            # One encode call ensures usage penalty sees all four state samples.
            z, aux, _ = self.model.encode(torch.cat([obs, nxt, left, right]), with_aux=True)
            z0, z1, zl, zr = z.chunk(4)
            edges = self.model.distance(z0, z1)
            random_d = self.model.distance(zl, zr)
            map_loss, stats = qrl_loss(random_d, edges, self.multiplier, c.epsilon, c.push_scale)
            metrics.update({k: float(v.detach()) for k, v in stats.items()})
            metrics.update(edge_distance=float(edges.mean().detach()), pair_distance=float(random_d.mean().detach()))
        else:
            x, y, gaps = replay.sample_pairs(c.batch_size, self.rng, c.horizon, c.max_pair_steps, c.exact_revisit_min)
            z, aux, _ = self.model.encode(self.tensor(np.concatenate([x, y])), with_aux=True)
            zx, zy = z.chunk(2)
            pred = self.model.distance_head(zx, zy)
            map_loss = time_myopic_loss(pred, self.tensor(gaps), c.horizon)
            metrics["temporal_mse"] = float(map_loss.detach())
        if aux:
            regularizer = aux["codebook"] + c.commitment_weight*aux["commitment"] + c.compression_weight*aux["usage_entropy"]
            map_loss = map_loss + regularizer
            metrics.update({k: float(v.detach()) for k, v in aux.items()})
        if not torch.isfinite(map_loss):
            raise FloatingPointError("Map loss became nonfinite")
        self.map_optimizer.zero_grad()
        map_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
        self.map_optimizer.step()
        if c.loss == "qrl":
            dual_ascent(self.multiplier, stats["constraint"].detach(), c.dual_lr)
            metrics["multiplier"] = float(self.multiplier)
        # Re-encode after the map update. Dynamics cannot collapse the encoder:
        # its inputs AND targets are detached; only the distance/compression losses
        # train the representation. Every head gets an independent bootstrap batch.
        with torch.no_grad():
            z0, z1 = self.model.encode(obs), self.model.encode(nxt)
        head_losses = []
        for i in range(len(self.dynamics.heads)):
            idx = torch.as_tensor(self.rng.integers(c.batch_size, size=c.batch_size), device=self.device)
            predicted = self.dynamics.predict_head(i, z0[idx], actions[idx])
            head_losses.append(F.mse_loss(predicted, z1[idx]))
        dynamics_loss = torch.stack(head_losses).mean()
        if not torch.isfinite(dynamics_loss):
            raise FloatingPointError("Dynamics loss became nonfinite")
        self.dynamics_optimizer.zero_grad()
        dynamics_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.dynamics.parameters(), 10.0)
        self.dynamics_optimizer.step()
        self.updates += 1
        metrics.update(update=self.updates, map_loss=float(map_loss.detach()), dynamics_mse=float(dynamics_loss.detach()),
                       latent_std=float(z0.std(dim=0).mean()))
        return metrics

    def update_batch(self, replay, head_indices):
        """Joint training on explicit replay indices, for either dynamics formulation."""
        if self.config.train.loss == "dynamics":
            return self.update_dynamics(replay, head_indices)
        return self.update_probabilistic(replay, head_indices)

    def update_dynamics(self, replay, head_indices=None):
        """Deterministic z=f(o): joint transition MSE + repulsion + optional L2."""
        c = self.config.train
        adjacency_terms = []
        replay_sample = head_indices is None
        if head_indices is None:
            head_indices = [self.rng.integers(len(replay), size=c.batch_size)
                            for _ in self.dynamics.heads]
        shared = c.ensemble_batches == "shared" or (c.ensemble_batches == "shared_replay" and replay_sample)
        if shared:
            obs, actions, nxt = replay.transitions_at(head_indices[0])
            z = self.model.encode(self.tensor(obs))
            target = self.model.encode(self.tensor(nxt))
            prediction = (self.dynamics(z, torch.as_tensor(actions, device=self.device))-target).square().mean()
            if c.adjacency_weight:
                adjacency_terms.append((z-target).square().sum(-1).mean())
        else:
            losses = []
            for m, indices in enumerate(head_indices):
                obs, actions, nxt = replay.transitions_at(indices)
                z = self.model.encode(self.tensor(obs))
                target = self.model.encode(self.tensor(nxt))
                pred = self.dynamics.predict_head(m, z, torch.as_tensor(actions, device=self.device))
                losses.append((pred-target).square().mean())
                if c.adjacency_weight:
                    adjacency_terms.append((z-target).square().sum(-1).mean())
            prediction = torch.stack(losses).mean()
        states = self.model.encode(self.tensor(replay.sample_states(2*c.batch_size, self.rng)))
        repulsion = repulsion_loss(*states.chunk(2), c.repulsion_margin)
        latent_squared_norm = states.square().sum(-1).mean()
        adjacency = torch.stack(adjacency_terms).mean() if adjacency_terms else prediction.new_zeros(())
        total = c.dynamics_weight*prediction + c.repulsion_weight*repulsion
        if c.adjacency_weight:
            total = total + c.adjacency_weight*adjacency
        if c.latent_l2:
            total = total + c.latent_l2*latent_squared_norm
        if not torch.isfinite(total):
            raise FloatingPointError("Deterministic joint loss became nonfinite")
        self.map_optimizer.zero_grad()
        self.dynamics_optimizer.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.)
        torch.nn.utils.clip_grad_norm_(self.dynamics.parameters(), 10.)
        self.map_optimizer.step()
        self.dynamics_optimizer.step()
        self.updates += 1
        return {'update': self.updates, 'joint_loss': float(total.detach()),
                'prediction_loss': float(prediction.detach()), 'repulsion': float(repulsion.detach()),
                'adjacency_loss': float(adjacency.detach()),
                'weighted_dynamics': float((c.dynamics_weight*prediction).detach()),
                'weighted_adjacency': float((c.adjacency_weight*adjacency).detach()),
                'repulsion_active_fraction': float(((states[:c.batch_size]-states[c.batch_size:]).square().sum(-1)<c.repulsion_margin**2).float().mean().detach()),
                'random_pair_distance_mean': float((states[:c.batch_size]-states[c.batch_size:]).norm(dim=-1).mean().detach()),
                'latent_squared_norm': float(latent_squared_norm.detach()),
                'weighted_latent_l2': float((c.latent_l2*latent_squared_norm).detach())}

    def update_probabilistic(self, replay, head_indices=None):
        c = self.config.train
        head_losses, errors, variances, means = [], [], [], []
        self.map_optimizer.zero_grad()
        self.dynamics_optimizer.zero_grad()
        # Draw the same number of indices in both modes, preserving repulsion RNG.
        replay_sample = head_indices is None
        if head_indices is None:
            head_indices = [self.rng.integers(len(replay), size=c.batch_size)
                            for _ in self.dynamics.heads]
        if c.ensemble_batches == "shared" or (c.ensemble_batches == "shared_replay" and replay_sample):
            head_indices = [head_indices[0]] * len(self.dynamics.heads)
        # Both ends retain encoder gradients.
        for m in range(len(self.dynamics.heads)):
            obs, actions, nxt = replay.transitions_at(head_indices[m])
            z, target = self.model.encode(self.tensor(np.concatenate([obs, nxt]))).chunk(2)
            params = self.dynamics.parameters_head(m, z, torch.as_tensor(actions, device=self.device))
            head_losses.append((mixture_mean(params)-target).square().mean() if c.prediction_objective == "mse"
                               else -mixture_log_prob(params, target).mean()/z.shape[-1])
            errors.append((mixture_mean(params)-target).square().mean().detach())
            variances.append(mixture_variance(params).mean().detach())
            means.append(z.detach())
        states = self.tensor(replay.sample_states(2*c.batch_size, self.rng))
        left, right = self.model.encode(states).chunk(2)
        repulsion = repulsion_loss(left, right, c.repulsion_margin)
        nll = torch.stack(head_losses).mean()
        total = nll + c.repulsion_weight*repulsion
        if not torch.isfinite(total):
            raise FloatingPointError("Probabilistic joint loss became nonfinite")
        total.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.)
        torch.nn.utils.clip_grad_norm_(self.dynamics.parameters(), 10.)
        self.map_optimizer.step()
        self.dynamics_optimizer.step()
        self.updates += 1
        latent = torch.cat(means)
        return {"update": self.updates, "joint_loss": float(total.detach()),
                "prediction_loss": float(nll.detach()),
                "nll_per_dim": float(nll.detach()) if c.prediction_objective == "nll" else None, "repulsion": float(repulsion.detach()),
                "dynamics_mse": float(torch.stack(errors).mean()),
                "predictive_variance": float(torch.stack(variances).mean()) if c.prediction_objective == "nll" else None,
                "latent_std": float(latent.std(0).mean())}

    def save(self, path):
        torch.save({"config": asdict(self.config), "observation_shape": self.observation_shape,
                    "model": self.model.state_dict(), "dynamics": self.dynamics.state_dict(),
                    "map_optimizer": self.map_optimizer.state_dict(),
                    "dynamics_optimizer": self.dynamics_optimizer.state_dict(),
                    "multiplier": self.multiplier, "updates": self.updates,
                    "rng_state": self.rng.bit_generator.state, "torch_rng_state": torch.get_rng_state()}, path)

    @classmethod
    def load(cls, path, device="cpu"):
        from .config import config_from_dict
        data = torch.load(path, map_location=device, weights_only=True)
        if "distance_spec" in data:
            from .distance_learning import DistanceTrainer
            return DistanceTrainer.from_checkpoint(data, device)
        if "bottleneck_spec" in data:
            from .information_bottleneck import BottleneckTrainer
            return BottleneckTrainer.from_checkpoint(data, device)
        cfg = config_from_dict(data["config"])
        cfg.device = device
        trainer = cls(cfg, tuple(data["observation_shape"]))
        trainer.model.load_state_dict(data["model"])
        trainer.dynamics.load_state_dict(data["dynamics"])
        trainer.map_optimizer.load_state_dict(data["map_optimizer"])
        trainer.dynamics_optimizer.load_state_dict(data["dynamics_optimizer"])
        trainer.multiplier.copy_(data["multiplier"])
        trainer.updates = data["updates"]
        trainer.rng.bit_generator.state = data["rng_state"]
        torch.set_rng_state(data["torch_rng_state"].cpu())
        return trainer
