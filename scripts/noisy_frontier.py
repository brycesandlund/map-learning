"""Noisy observation frontier study; physical state used only to restore/evaluate.

Example: .venv/bin/python scripts/noisy_frontier.py --arm distance --seed 0 --out runs/example
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from map_learning.config import load_config
from map_learning.maze import Maze, reachable_graph
from map_learning.training import Trainer, seed_all
from map_learning.replay import Replay
from map_learning.cadence import TrainingSchedule
from map_learning.experiment import prepare_run, log_metrics
from map_learning.frontier import LatentFrontier


@torch.no_grad()
def diagnostics(t, replay, points):
    # Fixed independent noise draws across phases/arms. No evaluation RNG enters search.
    env = Maze(t.config.maze, 91000 + t.config.seed)
    rng = np.random.default_rng(92000 + t.config.seed)
    p = points[rng.choice(len(points), min(64, len(points)), replace=False)]
    obs = np.stack([env.observe(x) for x in p for _ in range(16)])
    z = t.model.encode(t.tensor(obs)).numpy().reshape(len(p), 16, -1)
    within = np.mean((z - z.mean(1, keepdims=True))**2)
    between = np.var(z.mean(1), axis=0).mean()
    adjacent = np.stack([env.transition(x, a) for x in p for a in range(4)])
    # Common nuisance draws across neighboring positions avoid mistaking finite-sample
    # noise in centroid estimates for spatial structure. Evaluation only.
    adjacent_obs = np.repeat(obs.reshape(len(p), 16, -1)[:, None], 4, axis=1)
    adjacent_obs[..., :2] = (2*adjacent.reshape(len(p), 4, 2)/env.config.size-1)[:, :, None]
    az = t.model.encode(t.tensor(adjacent_obs.reshape(-1, obs.shape[-1]))).numpy().reshape(len(p), 4, 16, -1).mean(2)
    moving = np.any(adjacent.reshape(len(p), 4, 2) != p[:, None], axis=-1)
    local = np.linalg.norm(az - z.mean(1)[:, None], axis=-1)[moving]
    noise = np.linalg.norm(z[:, :8] - z[:, 8:], axis=-1)
    result = {'noise_variance_fraction': float(within / max(within + between, 1e-12)),
              'noise_pair_distance': float(noise.mean()),
              'moving_centroid_distance': float(local.mean()),
              'noise_to_move_ratio': float(noise.mean() / max(local.mean(), 1e-12)),
              'probe_latent_variance': float(within + between)}
    if len(replay):
        o, a, n = replay.transitions_at(rng.integers(len(replay), size=2048))
        z0, z1 = t.model.encode(t.tensor(np.concatenate([o, n]))).chunk(2)
        pred = t.dynamics(z0, torch.as_tensor(a))
        variance = torch.cat([z0, z1]).var(0, unbiased=False).mean().clamp_min(1e-12)
        result.update(seen_normalized_mse=float((pred.mean(0)-z1).square().mean()/variance),
                      seen_normalized_disagreement=float(pred.var(0, unbiased=False).mean()/variance),
                      repulsion=float((1-(z0-z0.roll(1, 0)).square().sum(-1)).clamp_min(0).mean()))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/noisy_frontier.json')
    p.add_argument('--arm', choices=['distance', 'disagreement'], required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--phases', type=int, default=20)
    p.add_argument('--expansions', type=int, default=1000)
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if min(a.phases, a.expansions, a.epochs) < 1: p.error('Budgets must be positive')
    cfg = load_config(a.config, [f'seed={a.seed}', f'collect_steps={a.phases*a.expansions*5}'])
    seed_all(cfg.seed, cfg.threads)
    env = Maze(cfg.maze, cfg.seed + 71000)
    t = Trainer(cfg, env.observation_shape)
    replay, schedule = Replay(), TrainingSchedule(t)
    out = prepare_run(a.out, cfg, {'arm': a.arm, 'phases': a.phases, 'expansions': a.expansions, 'epochs': a.epochs,
        'protocol': 'Fresh random initialization; no warmup. 8 independent N(0,1) nuisance coordinates. Cached observation restored with physical snapshot; fresh noise for each successor. No physical or observation deduplication. Fresh queue/visited each phase. Shared full replay epoch(s) after each phase. Disagreement on incoming parent/action; raw variance averaged across latent dimensions. No variance normalization needed for ranking within frozen phase.'})
    (out/'runner.py').write_text(Path(__file__).read_text())
    t.save(out/'checkpoint-0.pt')
    initial_hash = hashlib.sha256(b''.join(x.detach().numpy().tobytes() for x in list(t.model.parameters())+list(t.dynamics.parameters()))).hexdigest()
    # Oracle graph is exclusively evaluation; never passed to queue or networks.
    points, _, index = reachable_graph(Maze(cfg.maze))
    np.save(out/'evaluation_points.npy', points)
    visited = {tuple(env.start)}
    frames, phase_rows = [], []
    start = time.monotonic()
    with (out/'progress.jsonl').open('w') as progress, (out/'phases.jsonl').open('w') as phases:
        log_metrics(phases, {'phase': 0, 'interactions': 0, 'updates': 0, **diagnostics(t, replay, points)})
        for phase in range(a.phases):
            obs = env.reset()
            with torch.no_grad(): z = t.model.encode(t.tensor(obs[None]))[0].numpy()
            queue = LatentFrontier(a.arm, a.expansions*5, len(z), np.random.default_rng(cfg.seed+72000+phase))
            position = env.position.copy()
            phase_expanded, repeated = set(), 0
            initial_count = len(visited)
            tic = time.monotonic()
            for expansion in range(a.expansions):
                if expansion:
                    (position, obs), z, score = queue.pop()
                else: score = 0.
                # Evaluation only. Neither set is consulted by search.
                key = tuple(position)
                repeated += key in phase_expanded
                phase_expanded.add(key)
                queue.visit(z)
                next_obs, next_pos = [], []
                for action in range(5):
                    env.position = position.copy()  # exact restore, never deduplicate
                    nxt, _ = env.step(action)
                    next_obs.append(nxt); next_pos.append(env.position.copy())
                    replay.add(obs, action, nxt, len(replay), position, env.position)
                    visited.add(tuple(env.position))
                with torch.no_grad():
                    nz = t.model.encode(t.tensor(np.stack(next_obs))).numpy()
                    pred = t.dynamics(t.tensor(np.repeat(z[None], 5, axis=0)), torch.arange(5))
                    disagreement = pred.var(0, unbiased=False).mean(-1).numpy()
                for action in range(5):
                    queue.add(nz[action], (next_pos[action], next_obs[action]), float(disagreement[action]))
                if (expansion+1) % 50 == 0 or expansion+1 == a.expansions:
                    row = {'phase': phase+1, 'expansion': expansion+1, 'interactions': len(replay),
                           'coverage': len(visited)/len(points), 'visited': len(visited),
                           'phase_unique_expanded': len(phase_expanded), 'repeat_fraction': repeated/(expansion+1),
                           'selected_priority': score, 'frontier_entries': int(queue.active.sum())}
                    log_metrics(progress, row)
                    frames.append((len(replay), phase+1, expansion+1, position.copy(), np.array(sorted(visited))))
            search_seconds = time.monotonic()-tic
            replay.arrays()  # cache before epoch training
            tic = time.monotonic()
            for _ in range(a.epochs): training = schedule.train(replay)
            row.update(new_states=len(visited)-initial_count, updates=t.updates,
                       search_seconds=search_seconds, training_seconds=time.monotonic()-tic,
                       seconds=time.monotonic()-start, training_last_batch=training,
                       **diagnostics(t, replay, points))
            phase_rows.append(dict(row)); log_metrics(phases, row)
            print(json.dumps(row), flush=True)
            if phase+1 in {1, 5, 10, a.phases}: t.save(out/f'checkpoint-{phase+1}.pt')
    t.save(out/'checkpoint.pt'); replay.save(out/'replay.npz', cfg.maze)
    # Dense cumulative occupancy frames for portable animations, no object arrays.
    mask = np.zeros((len(frames), len(points)), dtype=bool)
    for i, frame in enumerate(frames):
        mask[i, [index[tuple(np.round(x, 5))] for x in frame[4]]] = True
    np.savez_compressed(out/'frames.npz', interactions=[x[0] for x in frames], phase=[x[1] for x in frames],
                        expansion=[x[2] for x in frames], position=[x[3] for x in frames], visited=mask, points=points)
    result = {'arm': a.arm, 'seed': a.seed, 'initial_parameters_sha256': initial_hash,
              'reachable_states': len(points), 'interactions': len(replay), 'updates': t.updates,
              'trained_through': schedule.trained_through, 'visited': len(visited),
              'coverage': len(visited)/len(points), 'seconds': time.monotonic()-start, 'phases': phase_rows}
    (out/'result.json').write_text(json.dumps(result, indent=2)+'\n')

if __name__ == '__main__': main()
