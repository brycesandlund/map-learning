"""Distance-centric encoder objectives for the persistent observation frontier.

These are explicitly adaptations: distances are Euclidean in the queue's space,
not learned directed heads. No physical-state labels enter the training API.
"""
from dataclasses import dataclass, asdict
import numpy as np
import torch
from torch.nn import functional as F
from .training import Trainer
from .probabilistic import repulsion_loss


@dataclass
class DistanceSpec:
    kind: str = 'adjacency'
    weight: float = .1
    horizon: int = 16
    edge_cost: float = .1
    epsilon: float = .05
    dual_lr: float = .1
    push_scale: float = 20.
    bandwidth: float = 1.

    def __post_init__(self):
        if self.kind not in {'adjacency', 'qrl_upper', 'temporal', 'temporal_upper', 'affinity'}:
            raise ValueError('Unknown distance objective')
        for key in ('weight', 'edge_cost', 'epsilon', 'dual_lr', 'push_scale', 'bandwidth'):
            if not np.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f'{key} must be positive and finite')
        if self.horizon < 1: raise ValueError('horizon must be positive')


class Ancestry:
    """Occurrence IDs, never deduplicated. Child i+1 is replay transition i.

    Parent IDs come from the popped frontier entry, not observation equality or
    chronological replay neighbors. A restore does not create a fictitious edge.
    """
    def __init__(self, capacity, shape):
        self.observations = np.empty((capacity+1, *shape), dtype=np.float32)
        self.parents = np.zeros(capacity+1, dtype=np.int64)
        self.depth = np.zeros(capacity+1, dtype=np.int64)
        self.size = 0

    def root(self, obs):
        if self.size: raise ValueError('Root already set')
        self.observations[0] = obs
        self.size = 1

    def add(self, parent, obs):
        if not 0 <= parent < self.size: raise ValueError('Invalid parent')
        if self.size >= len(self.parents): raise OverflowError('Ancestry full')
        child = self.size
        self.observations[child] = obs
        self.parents[child] = parent
        self.depth[child] = self.depth[parent]+1
        self.size += 1
        return child

    def pairs(self, transition_indices, rng, horizon):
        child = np.asarray(transition_indices, dtype=np.int64)+1
        if np.any(child >= self.size): raise IndexError('Unrecorded transition')
        # Half near, half extend beyond the clipping horizon. Include k=0.
        limit = np.minimum(self.depth[child], np.where(np.arange(len(child)) < len(child)//2, horizon, 4*horizon))
        lag = (rng.random(len(child))*(limit+1)).astype(np.int64)
        ancestor = child.copy()
        for step in range(int(lag.max(initial=0))):
            moving = lag > step
            ancestor[moving] = self.parents[ancestor[moving]]
        return self.observations[ancestor], self.observations[child], lag

    def state_dict(self):
        return {key: torch.from_numpy(getattr(self, key)[:self.size].copy())
                for key in ('observations', 'parents', 'depth')}

    def restore(self, data):
        self.size = len(data['parents'])
        for key in ('observations', 'parents', 'depth'):
            getattr(self, key)[:self.size] = data[key].cpu().numpy()


def student_neighbor_loss(left, right, bandwidth=1.):
    """Symmetric KL(P||Q): P has mass on each recorded source-successor pair.

    Q is normalized Student-t affinity over all non-diagonal minibatch pairs.
    P is temporal adjacency, not Gaussian affinities of observed pixels (t-SNE).
    Different occurrences of the same physical state remain distinct candidates.
    """
    z = torch.cat([left, right])
    n = len(left)
    squared = (z[:, None]-z[None, :]).square().sum(-1)
    log_kernel = -torch.log1p(squared/bandwidth**2)
    mask = torch.eye(2*n, dtype=torch.bool, device=z.device)
    log_kernel = log_kernel.masked_fill(mask, -torch.inf)
    log_normalizer = torch.logsumexp(log_kernel.flatten(), 0)
    i = torch.arange(n, device=z.device)
    positive_log_q = log_kernel[i, i+n]-log_normalizer
    return -positive_log_q.mean()-np.log(2*n)


def temporal_stress(distance, lag, horizon, upper_bound=False):
    prediction = -torch.expm1(-distance)
    label = (lag/horizon).clamp(max=1)
    residual = prediction-label
    return (F.relu(residual) if upper_bound else residual).square().mean()


class DistanceTrainer(Trainer):
    def __init__(self, config, observation_shape, spec):
        if config.train.loss != 'dynamics': raise ValueError('Use native deterministic architecture')
        super().__init__(config, observation_shape)
        self.distance_spec = spec if isinstance(spec, DistanceSpec) else DistanceSpec(**spec)
        self.ancestry = Ancestry(config.collect_steps, observation_shape)
        # Pair sampling must not change regularizer sample RNG across objectives.
        self.pair_rng = np.random.default_rng(config.seed+81001)

    def update(self, replay):
        idx = [self.rng.integers(len(replay), size=self.config.train.batch_size) for _ in self.dynamics.heads]
        return self.update_batch(replay, idx)

    def update_batch(self, replay, head_indices):
        c, s = self.config.train, self.distance_spec
        indices = np.asarray(head_indices[0])
        obs, actions, nxt = replay.transitions_at(indices)
        z, target = self.model.encode(self.tensor(np.concatenate([obs, nxt]))).chunk(2)
        states = self.model.encode(self.tensor(replay.sample_states(2*c.batch_size, self.rng)))
        left, right = states.chunk(2)
        repulsion = repulsion_loss(left, right, c.repulsion_margin)
        norm = states.square().sum(-1).mean()
        edge = torch.linalg.vector_norm(z-target, dim=-1)
        constraint = None
        extra = {}
        if s.kind == 'adjacency':
            task = edge.square().mean()
        elif s.kind == 'qrl_upper':
            pair_distance = torch.linalg.vector_norm(left-right, dim=-1)/s.edge_cost
            push = s.push_scale*F.softplus(-pair_distance/s.push_scale).mean()
            constraint = F.relu(edge/s.edge_cost-1).square().mean()-s.epsilon**2
            task = push+self.multiplier.detach()*constraint
            extra = {'constraint': float(constraint.detach()), 'push': float(push.detach())}
        elif s.kind in {'temporal', 'temporal_upper'}:
            a, b, lag = self.ancestry.pairs(indices, self.pair_rng, s.horizon)
            za, zb = self.model.encode(self.tensor(np.concatenate([a, b]))).chunk(2)
            distance = torch.linalg.vector_norm(za-zb, dim=-1)
            # Bounded metric (concave transform); actual observed lag is not a shortest path.
            task = temporal_stress(distance, self.tensor(lag), s.horizon, s.kind == 'temporal_upper')
            extra = {'mean_sampled_lag': float(np.mean(lag))}
        else:
            task = student_neighbor_loss(z, target, s.bandwidth)
        total = s.weight*task+c.repulsion_weight*repulsion+c.latent_l2*norm
        # Auxiliary dynamics are fitted for diagnostics ONLY. Detached at both ends.
        # Their architecture and initialization match the baseline exactly.
        prediction = self.dynamics(z.detach(), torch.as_tensor(actions, device=self.device))
        auxiliary = (prediction-target.detach()).square().mean()
        if not torch.isfinite(total+auxiliary): raise FloatingPointError('Nonfinite distance loss')
        self.map_optimizer.zero_grad()
        self.dynamics_optimizer.zero_grad()
        (total+auxiliary).backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.)
        torch.nn.utils.clip_grad_norm_(self.dynamics.parameters(), 10.)
        self.map_optimizer.step()
        self.dynamics_optimizer.step()
        if constraint is not None:
            with torch.no_grad(): self.multiplier.add_(s.dual_lr*constraint.detach()).clamp_(min=0)
        self.updates += 1
        return {'update': self.updates, 'joint_loss': float(total.detach()), 'distance_loss': float(task.detach()),
                'auxiliary_dynamics_mse': float(auxiliary.detach()), 'repulsion': float(repulsion.detach()),
                'edge_distance': float(edge.mean().detach()), 'latent_squared_norm': float(norm.detach()),
                'weighted_latent_l2': float((c.latent_l2*norm).detach()), 'multiplier': float(self.multiplier), **extra}

    def save(self, path):
        super().save(path)
        data = torch.load(path, map_location=self.device, weights_only=True)
        data.update(distance_spec=asdict(self.distance_spec), ancestry=self.ancestry.state_dict(),
                    pair_rng_state=self.pair_rng.bit_generator.state)
        torch.save(data, path)

    @classmethod
    def from_checkpoint(cls, data, device='cpu'):
        from .config import config_from_dict
        config = config_from_dict(data['config']); config.device = device
        trainer = cls(config, tuple(data['observation_shape']), data['distance_spec'])
        for key in ('model', 'dynamics', 'map_optimizer', 'dynamics_optimizer'):
            getattr(trainer, key).load_state_dict(data[key])
        trainer.multiplier.copy_(data['multiplier']); trainer.updates = data['updates']
        trainer.rng.bit_generator.state = data['rng_state']
        trainer.pair_rng.bit_generator.state = data['pair_rng_state']
        trainer.ancestry.restore(data['ancestry'])
        torch.set_rng_state(data['torch_rng_state'].cpu())
        return trainer
