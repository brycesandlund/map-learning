"""Continuous-coordinate, noisy persistent-frontier experiment (no lattice keys)."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from map_learning.config import load_config
from map_learning.continuous import sample_controls, SpatialCoverage, within_cell_fill
from map_learning.distance_learning import DistanceTrainer
from map_learning.experiment import prepare_run, log_metrics
from map_learning.maze import Maze
from map_learning.persistent_frontier import PersistentDistanceFrontier, epoch_slices
from map_learning.replay import Replay
from map_learning.training import Trainer, seed_all


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/continuous_frontier.json')
    p.add_argument('--objective', choices=['dynamics','hybrid','adjacency','oracle_xy','random_queue','frozen'], default='dynamics')
    p.add_argument('--seed', type=int, default=30)
    p.add_argument('--layout', type=int, default=7)
    p.add_argument('--blocks', type=int, default=20)
    p.add_argument('--nuisance-dims', type=int, default=8)
    p.add_argument('--proposal', choices=['stratified','iid'], default='stratified')
    p.add_argument('--step-size', type=float, default=.25)
    p.add_argument('--adjacency-weight', type=float, default=.3)
    p.add_argument('--dynamics-weight',type=float,default=1.)
    p.add_argument('--repulsion-weight',type=float)
    p.add_argument('--latent-l2',type=float)
    p.add_argument('--repulsion-margin',type=float,default=1.)
    p.add_argument('--regularizer', type=float, default=.03)
    p.add_argument('--extra-updates', type=int, default=0)
    p.add_argument('--extra-placement', choices=['early','distributed'], default='early')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.blocks < 1 or a.extra_updates < 0 or not np.isfinite(a.adjacency_weight) or a.adjacency_weight <= 0:
        p.error('Need positive blocks and adjacency weight')
    cfg = load_config(a.config, [f'seed={a.seed}', f'collect_steps={a.blocks*5000}',
        f'maze.seed={a.layout}', f'maze.nuisance_dims={a.nuisance_dims}', f'maze.step_size={a.step_size}',
        f'train.repulsion_weight={a.regularizer}', f'train.latent_l2={a.regularizer}'])
    cfg.train.dynamics_weight=a.dynamics_weight
    cfg.train.adjacency_weight=a.adjacency_weight if a.objective=='hybrid' else 0.
    if a.repulsion_weight is not None:cfg.train.repulsion_weight=a.repulsion_weight
    if a.latent_l2 is not None:cfg.train.latent_l2=a.latent_l2
    cfg.train.repulsion_margin=a.repulsion_margin
    cfg.validate()
    if cfg.model.continuous_action_dim != 2 or cfg.maze.observation != 'coordinates':
        p.error('Use continuous-action coordinate config')
    seed_all(cfg.seed, cfg.threads)
    env = Maze(cfg.maze, cfg.seed+71000)
    t = (DistanceTrainer(cfg, env.observation_shape, {'kind':'adjacency','weight':a.adjacency_weight})
         if a.objective == 'adjacency' else Trainer(cfg, env.observation_shape))
    trainable = a.objective in {'dynamics','hybrid','adjacency'}
    if a.extra_updates and not trainable: p.error('Extra updates require a learned objective')
    oracle = a.objective in {'oracle_xy','random_queue'}
    out = prepare_run(a.out, cfg, {'objective':a.objective,'proposal':a.proposal,
        'extra_updates':a.extra_updates,'extra_placement':a.extra_placement,
        'adjacency_weight':a.adjacency_weight,'protocol':'Five sampled continuous controls per expansion; exact restores; persistent raw observations. No dedup. Coverage bins evaluation only. One causal epoch per 5000 actions, ten training events. Oracle and random queue are untrained diagnostic controls.'})
    (out/'runner.py').write_text(Path(__file__).read_text())
    initial_hash = hashlib.sha256(b''.join(x.detach().numpy().tobytes() for x in
        list(t.model.parameters())+list(t.dynamics.parameters()))).hexdigest()
    t.save(out/'checkpoint-0.pt')
    @torch.no_grad()
    def encode(obs):
        return np.asarray(obs)[:, :2] if oracle else t.model.encode(t.tensor(obs)).numpy()
    replay = Replay()
    queue = PersistentDistanceFrontier(cfg.collect_steps, 2 if oracle else cfg.model.latent_dim,
                                      env.observation_shape, np.random.default_rng(cfg.seed+72000))
    rng = np.random.default_rng(cfg.seed+61000)
    controls_rng = np.random.default_rng(cfg.seed+91000)
    obs = env.reset(); position = env.position.copy(); z = encode(obs[None])[0]
    if a.objective == 'adjacency': t.ancestry.root(obs)
    coverage = SpatialCoverage(cfg.maze.size); coverage.add(position, 0)
    frames = []; sample_count = 0; blocked = 0; move_attempts = 0
    training_seconds = refresh_seconds = search_seconds = 0.
    start = time.monotonic()
    with (out/'progress.jsonl').open('w') as progress, (out/'training.jsonl').open('w') as training:
        for block in range(a.blocks):
            old_count = len(replay); old_order = rng.permutation(old_count)
            for offset in range(1000):
                tic = time.monotonic()
                expansion = block*1000+offset
                if expansion:
                    if a.objective == 'random_queue':
                        i = int(queue.rng.choice(np.flatnonzero(queue.active)))
                        queue.active[i] = False; queue.last_popped_index = i
                        obs, position, z, score = queue.observations[i].copy(), queue.entries[i], queue.latents[i].copy(), float(queue.priority[i])
                    else:
                        obs, position, z, score = queue.pop()
                else:
                    score = 0.
                node_id = queue.last_popped_index+1 if expansion else 0
                queue.visit(obs, z)
                controls = sample_controls(controls_rng, a.proposal)
                for action in controls:
                    env.position = position.copy()
                    nxt, _ = env.step(action); nxt_position = env.position.copy()
                    replay.add(obs, action, nxt, len(replay), position, nxt_position)
                    if np.any(action):
                        move_attempts += 1
                        blocked += int(np.array_equal(position, nxt_position))
                    if a.objective == 'adjacency':
                        assert t.ancestry.add(node_id, nxt) == len(replay)
                    queue.add(nxt, encode(nxt[None])[0], nxt_position)
                    coverage.add(nxt_position, len(replay))
                search_seconds += time.monotonic()-tic
                if (offset+1)%100 == 0:
                    if trainable:
                        tic = time.monotonic(); replay.arrays(); j=(offset+1)//100-1
                        batches = epoch_slices(old_count, old_count+j*500, len(replay), j, 10,
                                               rng, old_order, (block+1)*5000, cfg.train.batch_size)
                        for batch in batches:
                            metrics = t.update_batch(replay, [batch]*cfg.model.ensemble_size)
                            sample_count += len(batch)
                        if offset == 999:
                            extra = (a.extra_updates if block == 0 else 0) if a.extra_placement == 'early' else (
                                a.extra_updates*(block+1)//a.blocks-a.extra_updates*block//a.blocks)
                            for _ in range(extra):
                                metrics = t.update(replay)
                                sample_count += cfg.train.batch_size
                        training_seconds += time.monotonic()-tic
                        log_metrics(training, {'interactions':len(replay),'updates':t.updates,
                                    'samples_per_head':sample_count,**metrics})
                    tic = time.monotonic(); queue.refresh(encode); refresh_seconds += time.monotonic()-tic
                if (offset+1)%50 == 0:
                    row = {'interactions':len(replay),'expansions':expansion+1,'updates':t.updates,
                           'frontier_entries':int(queue.active.sum()),'selected_priority':score,
                           'blocked_fraction':blocked/max(1,move_attempts),**coverage.metrics(),
                           'seconds':time.monotonic()-start}
                    log_metrics(progress,row)
                    frames.append((len(replay),position.copy(),coverage.masks[4].copy(),coverage.masks[8].copy()))
            print(json.dumps(row),flush=True)
            if block+1 in {1,5,10,20,a.blocks}: t.save(out/f'checkpoint-{block+1}.pt')
    t.save(out/'checkpoint.pt'); replay.save(out/'replay.npz',cfg.maze)
    data = replay.arrays(); all_positions = np.concatenate([env.start[None],data['next_positions']])
    np.savez_compressed(out/'frames.npz',interactions=[f[0] for f in frames],position=[f[1] for f in frames],
                        occupied_k4=[f[2] for f in frames],occupied_k8=[f[3] for f in frames])
    ids = np.flatnonzero(queue.active)
    np.savez_compressed(out/'frontier.npz',observations=queue.observations[ids],positions=np.array(queue.entries)[ids],
                        priorities=queue.priority[ids],latents=queue.latents[ids],
                        visited_observations=queue.visited_observations[:queue.n_visited])
    moving = np.linalg.norm(data['next_positions']-data['positions'],axis=1)>1e-7
    lattice = np.all(np.abs(all_positions*4-np.rint(all_positions*4))<1e-5,axis=1)
    result = {**row,'objective':a.objective,'seed':a.seed,'layout':a.layout,'proposal':a.proposal,
              'extra_updates':a.extra_updates,'extra_placement':a.extra_placement,
              'nuisance_dims':a.nuisance_dims,'step_size':a.step_size,'adjacency_weight':a.adjacency_weight if a.objective in {'adjacency','hybrid'} else 0.,
              'dynamics_weight':cfg.train.dynamics_weight if a.objective!='adjacency' else 0.,
              'repulsion_weight':cfg.train.repulsion_weight,'latent_l2':cfg.train.latent_l2,'repulsion_margin':cfg.train.repulsion_margin,
              'initial_parameters_sha256':initial_hash,'samples_per_head':sample_count,
              'trained_through':len(replay) if trainable else 0,'milestones':coverage.milestones,
              'quarter_lattice_fraction':float(lattice.mean()),'moving_transitions':int(moving.sum()),
              'unique_endpoints':len(np.unique(all_positions,axis=0)),
              **within_cell_fill(all_positions,cfg.maze.size),
              'training_seconds':training_seconds,'refresh_seconds':refresh_seconds,'search_seconds':search_seconds,
              'seconds':time.monotonic()-start}
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)

if __name__ == '__main__': main()
