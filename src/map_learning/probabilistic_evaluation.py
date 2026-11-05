"""Evaluation-only access to maze geometry; never used for fitting or planning."""
import json
from pathlib import Path
import numpy as np
import torch
from .maze import Maze, reachable_graph, bfs
from .probabilistic import ensemble_js, mixture_log_prob, mixture_mean, mixture_variance, ensemble_sample
from .evaluation import encode_batch, plotting, coverage_plot, correlation_or_none


@torch.no_grad()
def evaluate_probabilistic(trainer, replay, output, heldout=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    cfg = trainer.config
    env = Maze(cfg.maze, cfg.seed+8000)
    points, adjacency, lookup = reachable_graph(env)
    data = replay.arrays()
    visited = {lookup[tuple(np.round(p, 5))] for p in np.concatenate([data['positions'], data['next_positions']])}
    rooms = {tuple(np.floor(p).astype(int)) for p in np.concatenate([data['positions'], data['next_positions']])}
    obs = np.stack([env.observe(p) for p in points])
    z = encode_batch(trainer, obs)
    latent_variance = float(z.var(0, unbiased=False).mean())
    singular = torch.linalg.svdvals(z-z.mean(0))
    mass = singular.square()/singular.square().sum().clamp_min(1e-12)
    metric = {"oracle_nodes": len(points), "coverage_lattice": len(visited)/len(points),
              "coverage_rooms": len(rooms)/cfg.maze.size**2,
              "latent_variance": latent_variance,
              "latent_effective_rank": float((-(mass*mass.clamp_min(1e-12).log()).sum()).exp()) if latent_variance>1e-12 else 0.,
              "encoder_saturation_fraction": (float((z.abs()>.99*cfg.model.latent_bound).float().mean())
                                               if getattr(cfg.model, "latent_activation", "tanh") == "tanh" else None),
              "updates": trainer.updates, "environment_steps": len(replay)}
    generator = torch.Generator(device=trainer.device).manual_seed(cfg.seed+51000)
    rng = np.random.default_rng(cfg.seed+52000)

    def diagnostics(z0, actions, z1):
        params = trainer.dynamics.distribution(z0, actions)
        uncertainty = ensemble_js(params, max(32, cfg.planner.disagreement_samples), generator)
        means = mixture_mean(params)
        error = (means.mean(0)-z1).square().mean(-1)
        nll = -mixture_log_prob(params, z1)/z0.shape[-1]
        return params, uncertainty, error, nll

    for label, buffer in [("replay", replay), ("heldout", heldout)]:
        if buffer is None:
            continue
        x, a, y = buffer.sample_transitions(1024, rng)
        params, uncertainty, error, nll = diagnostics(encode_batch(trainer, x),
                    torch.as_tensor(a, device=trainer.device), encode_batch(trainer, y))
        metric.update({f"{label}_nll_per_dim": float(nll.mean()),
                       f"{label}_normalized_mse": float(error.mean())/max(latent_variance, 1e-8),
                       f"{label}_disagreement": float(uncertainty.mean()),
                       f"{label}_predictive_variance": float(mixture_variance(params).mean())})

    # All reachable state-action pairs, including unvisited ones. Privileged
    # transitions provide diagnostics only; they are never appended to replay.
    errors, disagreements, nlls, within, between, effective_components = [], [], [], [], [], []
    source = np.repeat(np.arange(len(points)), 5)
    target = adjacency.reshape(-1)
    actions = np.tile(np.arange(5), len(points))
    seen_edges = {(lookup[tuple(np.round(p, 5))], int(a)) for p, a in zip(data['positions'], data['actions'])}
    seen = np.array([(int(i), int(a)) in seen_edges for i, a in zip(source, actions)])
    for start in range(0, len(source), 256):
        sl = slice(start, start+256)
        params, uncertainty, error, nll = diagnostics(z[source[sl]],
                      torch.as_tensor(actions[sl], device=trainer.device), z[target[sl]])
        errors.extend(error.cpu().tolist())
        disagreements.extend(uncertainty.cpu().tolist())
        nlls.extend(nll.mean(0).cpu().tolist())
        within.extend(mixture_variance(params).mean((0, 2)).cpu().tolist())
        between.extend(mixture_mean(params).var(0, unbiased=False).mean(-1).cpu().tolist())
        pi = params[0].softmax(-1)
        effective_components.extend((-(pi*pi.clamp_min(1e-12).log()).sum(-1)).exp().mean(0).cpu().tolist())
    errors, disagreements, nlls = map(np.asarray, (errors, disagreements, nlls))
    metric['oracle_uncertainty_error_correlation'] = correlation_or_none(disagreements, errors)
    metric['oracle_effective_mixture_components'] = float(np.mean(effective_components))
    for label, mask in [('all', np.ones(len(source), dtype=bool)), ('seen', seen), ('unseen', ~seen)]:
        metric[f'oracle_{label}_state_actions'] = int(mask.sum())
        for name, values in [('normalized_mse', errors/max(latent_variance, 1e-8)),
                             ('disagreement', disagreements), ('nll_per_dim', nlls)]:
            metric[f'oracle_{label}_{name}'] = float(values[mask].mean()) if mask.any() else None
    # Compare with the current-state persistence predictor, which is deceptively
    # strong when many attempted movements hit walls.
    metric['oracle_persistence_normalized_mse'] = float((z[source]-z[target]).square().mean())/max(latent_variance, 1e-8)
    z_again = encode_batch(trainer, np.stack([env.observe(p) for p in points]))
    metric['same_position_latent_mse'] = float((z-z_again).square().mean())

    # Evaluate recursive mean rollouts on real trajectories, never crossing reset.
    rollout_rng = np.random.default_rng(cfg.seed+53000)
    horizon = min(10, cfg.maze.episode_steps)
    beginnings = rollout_rng.integers(len(points), size=128)
    physical = points[beginnings].copy()
    state = z[beginnings].clone()
    for h in range(1, horizon+1):
        a = rollout_rng.integers(5, size=len(state))
        params = trainer.dynamics.distribution(state, torch.as_tensor(a, device=trainer.device))
        state = mixture_mean(params).mean(0)
        physical = np.stack([env.transition(p, int(action)) for p, action in zip(physical, a)])
        truth = encode_batch(trainer, np.stack([env.observe(p) for p in physical]))
        if h in {1, 3, 5, 10, horizon}:
            metric[f'oracle_mean_rollout_{h}_normalized_mse'] = float((state-truth).square().mean())/max(latent_variance, 1e-8)
    params = trainer.dynamics.distribution(z, torch.zeros(len(z), dtype=torch.long, device=trainer.device))
    draws = ensemble_sample(params, generator)
    metric['oracle_sample_outside_encoder_bounds_fraction'] = (float((draws.abs()>cfg.model.latent_bound).any(-1).float().mean())
                                                              if getattr(cfg.model, "latent_activation", "tanh") == "tanh" else None)
    np.savez_compressed(output/'probe.npz', points=points, source=source, actions=actions,
                        seen=seen, mse=errors, disagreement=disagreements, nll_per_dim=nlls,
                        within_variance=within, between_mean_variance=between, latents=z.cpu().numpy())
    (output/'metrics.json').write_text(json.dumps(metric, indent=2, allow_nan=False)+'\n')
    coverage_plot(env, replay, output/'coverage.png', f"Replay coverage: {len(rooms)}/{cfg.maze.size**2} rooms")
    plt = plotting()
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for ax, values, title in [(axes[0, 0], disagreements.reshape(-1, 5).mean(-1), 'Distribution disagreement (nats)'),
                              (axes[0, 1], errors.reshape(-1, 5).mean(-1)/max(latent_variance, 1e-8), 'One-step normalized latent MSE')]:
        env.draw(ax)
        color = ax.scatter(points[:, 0], points[:, 1], c=values, s=14, cmap='viridis')
        fig.colorbar(color, ax=ax, shrink=.8)
        ax.set_title(title)
    _, _, vectors = torch.linalg.svd(z-z.mean(0), full_matrices=False)
    projection = ((z-z.mean(0))@vectors[:2].T).cpu().numpy()
    color = axes[1, 0].scatter(*projection.T, c=bfs(adjacency, 0), cmap='viridis', s=12)
    axes[1, 0].set_title('Latent PCA; color = oracle steps from start')
    fig.colorbar(color, ax=axes[1, 0], shrink=.8)
    for name, mask in [('seen state-actions', seen), ('unseen state-actions', ~seen)]:
        axes[1, 1].scatter(disagreements[mask], errors[mask]/max(latent_variance, 1e-8), s=6, alpha=.3, label=name)
    axes[1, 1].set(xlabel='Disagreement (nats)', ylabel='Normalized latent MSE', title='Does uncertainty track error?')
    axes[1, 1].legend()
    fig.tight_layout()
    fig.savefig(output/'diagnostics.png', dpi=150)
    plt.close(fig)
    return metric
