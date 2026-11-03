from dataclasses import asdict
import random
import numpy as np
import torch
from torch.nn import functional as F
from .losses import qrl_loss, time_myopic_loss, dual_ascent
from .models import MapModel, DynamicsEnsemble


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
        self.dynamics = DynamicsEnsemble(config.model).to(self.device)
        self.map_optimizer = torch.optim.Adam(self.model.parameters(), lr=config.train.lr)
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
