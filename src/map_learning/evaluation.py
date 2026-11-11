"""Diagnostics with oracle geometry isolated from learning and action selection."""
from collections import Counter
import json
import os
from pathlib import Path
import numpy as np
import torch
from .maze import Maze, reachable_graph, bfs


def correlation_or_none(x, y):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if np.ptp(x) < 1e-8 or np.ptp(y) < 1e-8:
        return None
    value = float(np.corrcoef(x, y)[0, 1])
    return value if np.isfinite(value) else None


def plotting():
    os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd()/".mpl-cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def coverage_plot(env, replay, path, title="Collected trajectory coverage"):
    plt = plotting()
    fig, ax = plt.subplots(figsize=(7, 7))
    env.draw(ax)
    p = replay.arrays()["next_positions"]
    ax.scatter(p[:, 0], p[:, 1], s=2, alpha=.35, c="#ec3535")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def split_replay(replay, fraction=.2, seed=0):
    """Keep complete episodes together; both losses receive exactly the same split."""
    from .replay import Replay
    data = replay.arrays()
    episodes = np.unique(data["episodes"])
    if len(episodes) < 2:
        raise ValueError("Offline training needs >=2 episodes for a trajectory-held-out evaluation")
    shuffled = np.random.default_rng(seed).permutation(episodes)
    holdout = set(shuffled[:max(1, int(len(episodes)*fraction))])
    train, test = Replay(), Replay()
    for i, episode in enumerate(data["episodes"]):
        target = test if episode in holdout else train
        target.add(data["observations"][i], data["actions"][i], data["next_observations"][i],
                   episode, data["positions"][i], data["next_positions"][i])
    return train, test


@torch.no_grad()
def encode_batch(trainer, observations, batch=512):
    return torch.cat([trainer.model.encode(trainer.tensor(observations[i:i+batch]))
                      for i in range(0, len(observations), batch)])


@torch.no_grad()
def evaluate(trainer, replay, output, heldout=None):
    if trainer.config.train.loss == "dynamics":
        from .deterministic import evaluate_deterministic
        return evaluate_deterministic(trainer, replay, output, heldout)
    if trainer.config.train.loss == "probabilistic":
        from .probabilistic_evaluation import evaluate_probabilistic
        return evaluate_probabilistic(trainer, replay, output, heldout)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    env = Maze(trainer.config.maze, seed=trainer.config.seed+8000)
    points, adjacency, lookup = reachable_graph(env)
    rng = np.random.default_rng(trainer.config.seed+2000)
    observations = np.stack([env.observe(p) for p in points])
    z = encode_batch(trainer, observations)
    from_start = bfs(adjacency, 0)
    prediction = trainer.model.distance(z[:1], z).cpu().numpy()
    # Many sources are needed: a single radial distance field misses bad shortcuts.
    sources = rng.choice(len(points), min(32, len(points)), replace=False)
    lhs, rhs, true = [], [], []
    for source in sources:
        dist = bfs(adjacency, source)
        destinations = rng.integers(len(points), size=64)
        lhs.extend([source]*len(destinations))
        rhs.extend(destinations)
        true.extend(dist[destinations])
    pred = trainer.model.distance(z[lhs], z[rhs]).cpu().numpy()
    true = np.array(true)
    horizon = trainer.config.train.horizon
    target = true if trainer.config.train.loss == "qrl" else np.minimum(true, horizon)
    variance = z.var(dim=0, unbiased=False).mean().item()
    metric = {
        "oracle_nodes": len(points),
        "oracle_pair_mae": float(np.abs(pred-target).mean()),
        "oracle_pair_target": "shortest_steps" if trainer.config.train.loss == "qrl" else "capped_shortest_steps",
        "oracle_pair_correlation": correlation_or_none(pred, target),
        "latent_variance": variance,
        "latent_effective_rank": None,
    }
    nearby = true <= horizon
    metric["oracle_local_pair_mae"] = float(np.abs(pred[nearby]-target[nearby]).mean()) if nearby.any() else None
    singular = torch.linalg.svdvals(z-z.mean(0)).cpu().numpy()
    mass = singular**2 / max(float((singular**2).sum()), 1e-12)
    metric["latent_effective_rank"] = float(np.exp(-(mass*np.log(mass+1e-12)).sum())) if variance>1e-12 else 0.0
    visited = set()
    data = replay.arrays()
    for p in np.concatenate([data["positions"], data["next_positions"]]):
        k = tuple(np.round(p, 5))
        if k in lookup:
            visited.add(lookup[k])
    metric["coverage_lattice"] = len(visited)/len(points)
    seen_cells = {tuple(np.floor(p).astype(int)) for p in data["next_positions"]}
    metric["coverage_rooms"] = len(seen_cells)/env.config.size**2
    mask = np.array([a in visited and b in visited for a, b in zip(lhs, rhs)])
    for name, selected in [("visited_pairs", mask), ("pairs_with_unvisited_endpoint", ~mask)]:
        metric[f"oracle_mae_{name}"] = float(np.abs(pred[selected]-target[selected]).mean()) if selected.any() else None
    a, b, c = rng.integers(len(points), size=(3, 1024))
    violation = (trainer.model.distance(z[a], z[c]) - trainer.model.distance(z[a], z[b]) - trainer.model.distance(z[b], z[c])).clamp_min(0)
    metric["triangle_violation_max"] = float(violation.max())
    metric["self_distance_mean"] = float(trainer.model.distance(z, z).mean())
    # One-step prediction on genuinely held-out episodes (offline), or replay
    # diagnostics (online). The naming prevents mistaking training error for holdout.
    evaluation_replay = heldout if heldout is not None else replay
    prefix = "heldout" if heldout is not None else "replay"
    obs, actions, nxt = evaluation_replay.sample_transitions(1024, rng)
    z0, z1 = encode_batch(trainer, obs), encode_batch(trainer, nxt)
    next_predictions = trainer.dynamics(z0, torch.as_tensor(actions, device=trainer.device))
    metric[f"{prefix}_dynamics_mse"] = float((next_predictions.mean(0)-z1).square().mean())
    metric[f"{prefix}_dynamics_normalized_mse"] = metric[f"{prefix}_dynamics_mse"]/max(variance, 1e-8)
    edge = trainer.model.distance(z0, z1)
    metric[f"{prefix}_edge_violation_mse"] = float((edge-1).clamp_min(0).square().mean())
    if trainer.config.train.loss == "time_myopic":
        x, y, gaps = evaluation_replay.sample_pairs(1024, rng, horizon, trainer.config.train.max_pair_steps,
                                                    trainer.config.train.exact_revisit_min)
        xz, yz = encode_batch(trainer, x), encode_batch(trainer, y)
        p = trainer.model.distance_head(xz, yz).cpu().numpy()
        metric[f"{prefix}_temporal_mse"] = float(((p-np.minimum(gaps/horizon, 1))**2).mean())
    # Nuisance invariance is evaluated from two different renders of the SAME state.
    # These privileged equivalence labels never reach the trainer.
    second = np.stack([env.observe(p) for p in points[:min(len(points), 1024)]])
    z_again = encode_batch(trainer, second)
    metric["same_position_latent_mse"] = float((z[:len(second)]-z_again).square().mean())
    if trainer.model.bottleneck:
        ids = (z[:, None]-trainer.model.bottleneck.codes).square().mean(-1).argmin(-1).cpu().numpy()
        counts = np.bincount(ids, minlength=len(trainer.model.bottleneck.codes))
        p = counts[counts>0]/counts.sum()
        metric["oracle_used_codes"] = int((counts>0).sum())
        metric["oracle_code_perplexity"] = float(np.exp(-(p*np.log(p)).sum()))
    metric.update(export_graph(trainer, replay, output/"graph.json"))
    (output/"metrics.json").write_text(json.dumps(metric, indent=2, allow_nan=False)+"\n")
    plt = plotting()
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax in axes.flat[:3]:
        env.draw(ax)
    p = data["next_positions"]
    axes[0, 0].scatter(p[:, 0], p[:, 1], s=2, color="#ed4040", alpha=.3)
    axes[0, 0].set_title(f"Visited: {metric['coverage_rooms']:.0%} of rooms")
    for ax, values, title in [(axes[0, 1], from_start, "Oracle steps from start"),
                               (axes[1, 0], prediction, "Learned distance from start")]:
        plot = ax.scatter(points[:, 0], points[:, 1], c=values, s=6, cmap="viridis")
        fig.colorbar(plot, ax=ax, shrink=.8)
        ax.set_title(title)
    axes[1, 1].scatter(target, pred, s=5, alpha=.25)
    maximum = max(float(target.max()), float(pred.max()), 1)
    axes[1, 1].plot([0, maximum], [0, maximum], "k--", linewidth=1)
    axes[1, 1].set(xlabel="Oracle steps (capped for Time-Myopic)", ylabel="Predicted distance",
                   title=f"{trainer.config.train.loss}: pair MAE {metric['oracle_pair_mae']:.2f}")
    fig.tight_layout()
    fig.savefig(output/"diagnostics.png", dpi=150)
    plt.close(fig)
    # PCA is only a visualization, not the metric used by the planner.
    _, _, vectors = torch.linalg.svd(z-z.mean(0), full_matrices=False)
    projection = ((z-z.mean(0)) @ vectors[:2].T).cpu().numpy()
    fig, ax = plt.subplots(figsize=(6, 5))
    scatter = ax.scatter(projection[:, 0], projection[:, 1], c=from_start, s=6, cmap="viridis")
    fig.colorbar(scatter, ax=ax, label="Oracle distance from start")
    ax.set_title("Latent PCA (visualization only)")
    fig.tight_layout()
    fig.savefig(output/"latent.png", dpi=150)
    plt.close(fig)
    return metric


@torch.no_grad()
def export_graph(trainer, replay, path):
    data = replay.arrays()
    observations = np.concatenate([data["observations"], data["next_observations"]])
    if trainer.model.bottleneck:
        z = encode_batch(trainer, observations)
        # Avoid O(N*K*D) allocation for a full replay.
        ids = np.concatenate([(part[:, None]-trainer.model.bottleneck.codes).square().mean(-1).argmin(-1).cpu().numpy()
                              for part in z.split(512)])
        kind = "learned_codebook"
    else:
        index, ids = {}, []
        for obs in observations:
            key = obs.tobytes()
            if key not in index:
                index[key] = len(index)
            ids.append(index[key])
        ids = np.array(ids)
        kind = "exact_observation_identity_baseline"
    n = len(replay)
    edges = Counter((int(a), int(b), int(action)) for a, b, action in zip(ids[:n], ids[n:], data["actions"]))
    payload = {"node_definition": kind, "nodes": sorted(set(map(int, ids))),
               "edges": [{"source": a, "target": b, "action": action, "count": count}
                         for (a, b, action), count in sorted(edges.items())]}
    Path(path).write_text(json.dumps(payload, indent=2)+"\n")
    outcomes = {}
    for (a, b, action), count in edges.items():
        outcomes.setdefault((a, action), Counter())[b] += count
    ambiguous = sum(len(targets)>1 for targets in outcomes.values())
    return {"graph_nodes": len(payload["nodes"]), "graph_edges": len(edges),
            "graph_ambiguous_state_actions": ambiguous,
            "graph_ambiguous_state_action_fraction": ambiguous/max(len(outcomes), 1)}
