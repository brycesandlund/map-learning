"""Checkpoint migration and evaluation for z=f(o), residual dynamics, repulsion, L2."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import torch

from .training import Trainer


def without_distributions(legacy):
    """Strip inactive distribution outputs, retaining weights and Adam history.

    Only a single-component MSE model with no active KL or sampled encoder inputs
    is equivalent. Smaller output GEMMs can change float32 rounding; this is
    functional/optimization equivalence, not a promise of identical long rollouts.
    """
    cfg = deepcopy(legacy.config)
    if cfg.train.loss != 'probabilistic' or cfg.train.prediction_objective != 'mse' or cfg.model.mixture_components != 1:
        raise ValueError('Conversion requires legacy single-component MSE dynamics')
    spec = getattr(legacy, 'spec', {})
    if spec and (spec['beta'] != 0 or spec['dynamics_input'] != 'mean'):
        raise ValueError('Active KL or sampled encoder inputs cannot be discarded equivalently')
    cfg.train.loss = 'dynamics'
    cfg.train.latent_l2 = spec.get('latent_l2', 0.)
    cfg.validate()
    with torch.random.fork_rng():
        result = Trainer(cfg, legacy.observation_shape)
    dim = cfg.model.latent_dim

    def transfer(old_module, new_module, old_optimizer, new_optimizer):
        old = dict(old_module.named_parameters())
        for group, previous in zip(new_optimizer.param_groups, old_optimizer.param_groups):
            group.update({k: deepcopy(v) for k, v in previous.items() if k != 'params'})
        for name, parameter in new_module.named_parameters():
            source = old[name]
            rows = slice(None) if source.shape == parameter.shape else slice(1, 1+dim)
            value = source[rows]
            if value.shape != parameter.shape:
                raise ValueError(f'Cannot convert parameter {name}')
            with torch.no_grad():
                parameter.copy_(value)
            if source in old_optimizer.state:
                state = {}
                for key, value in old_optimizer.state[source].items():
                    state[key] = (value[rows].clone() if torch.is_tensor(value) and value.shape == source.shape
                                  else value.clone() if torch.is_tensor(value) else deepcopy(value))
                new_optimizer.state[parameter] = state

    transfer(legacy.model, result.model, legacy.map_optimizer, result.map_optimizer)
    transfer(legacy.dynamics, result.dynamics, legacy.dynamics_optimizer, result.dynamics_optimizer)
    result.updates = legacy.updates
    result.rng.bit_generator.state = deepcopy(legacy.rng.bit_generator.state)
    result.multiplier.copy_(legacy.multiplier)
    return result


@torch.no_grad()
def evaluate_deterministic(trainer, replay, output, heldout=None):
    """Latent prediction/coverage diagnostics without distribution-valued outputs."""
    from .evaluation import encode_batch, coverage_plot, plotting, correlation_or_none
    from .maze import Maze, reachable_graph
    output = Path(output)
    output.mkdir(exist_ok=True, parents=True)
    cfg = trainer.config
    env = Maze(cfg.maze, cfg.seed+8000)
    points, adjacency, lookup = reachable_graph(env)
    data = replay.arrays()
    visited = {lookup[tuple(np.round(p, 5))] for p in np.concatenate([data['positions'], data['next_positions']])}
    seen_edges = {(lookup[tuple(np.round(p, 5))], int(a)) for p, a in zip(data['positions'], data['actions'])}
    z = encode_batch(trainer, np.stack([env.observe(p) for p in points]))
    spread = float(z.var(0, unbiased=False).mean())
    scale = max(spread, 1e-8)
    metrics = {'oracle_nodes': len(points), 'coverage_lattice': len(visited)/len(points),
               'latent_variance': spread, 'environment_steps': len(replay), 'updates': trainer.updates,
               'disagreement_kind': 'normalized_ensemble_prediction_variance'}
    rng = np.random.default_rng(cfg.seed+52000)
    for name, buffer in [('replay', replay), ('heldout', heldout)]:
        if buffer is None:
            continue
        o, a, nxt = buffer.sample_transitions(1024, rng)
        pred = trainer.dynamics(encode_batch(trainer, o), torch.as_tensor(a, device=trainer.device))
        target = encode_batch(trainer, nxt)
        metrics[f'{name}_normalized_mse'] = float((pred.mean(0)-target).square().mean())/scale
    source = np.repeat(np.arange(len(points)), 5)
    actions = np.tile(np.arange(5), len(points))
    target = adjacency.ravel()
    seen = np.array([(int(s), int(a)) in seen_edges for s, a in zip(source, actions)])
    errors, disagreement = [], []
    for start in range(0, len(source), 256):
        sl = slice(start, start+256)
        pred = trainer.dynamics(z[source[sl]], torch.as_tensor(actions[sl], device=trainer.device))
        errors.extend(((pred.mean(0)-z[target[sl]]).square().mean(-1)/scale).cpu().tolist())
        disagreement.extend((pred.var(0, unbiased=False).mean(-1)/scale).cpu().tolist())
    errors, disagreement = np.asarray(errors), np.asarray(disagreement)
    for name, mask in [('all', np.ones(len(seen), bool)), ('seen', seen), ('unseen', ~seen)]:
        metrics[f'oracle_{name}_state_actions'] = int(mask.sum())
        metrics[f'oracle_{name}_normalized_mse'] = float(errors[mask].mean()) if mask.any() else None
        metrics[f'oracle_{name}_disagreement'] = float(disagreement[mask].mean()) if mask.any() else None
    metrics['oracle_uncertainty_error_correlation'] = correlation_or_none(disagreement, errors)
    (output/'metrics.json').write_text(json.dumps(metrics, indent=2, allow_nan=False)+'\n')
    np.savez_compressed(output/'probe.npz', points=points, source=source, actions=actions, seen=seen,
                        normalized_mse=errors, disagreement=disagreement, latents=z.cpu().numpy())
    coverage_plot(env, replay, output/'coverage.png')
    plt = plotting()
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    for ax, values, title in zip(axs, [errors, disagreement], ['Normalized prediction MSE', 'Ensemble disagreement']):
        env.draw(ax)
        color = ax.scatter(*points.T, c=values.reshape(-1, 5).mean(1), s=12)
        fig.colorbar(color, ax=ax); ax.set_title(title)
    fig.savefig(output/'diagnostics.png', dpi=140); plt.close(fig)
    return metrics
