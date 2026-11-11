"""Persistent distance frontier, matched full-epoch vs causally interleaved epoch."""
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
from map_learning.persistent_frontier import PersistentDistanceFrontier, epoch_slices
from noisy_frontier import diagnostics


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--arm',choices=['epoch1000','interleaved100'],required=True)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--blocks',type=int,default=20)
    p.add_argument('--block-expansions',type=int,default=1000)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--extra-updates',type=int,default=0)
    p.add_argument('--extra-placement',choices=['early','distributed'],default='early')
    a=p.parse_args()
    if a.extra_updates<0: p.error('Extra updates must be nonnegative')
    if a.block_expansions%10 or a.block_expansions<500 or a.blocks<1: p.error('Block expansions must be >=500 and divisible by 10')
    cfg=load_config('configs/noisy_frontier.json',[f'seed={a.seed}',f'collect_steps={a.blocks*a.block_expansions*5}'])
    seed_all(cfg.seed,cfg.threads)
    env=Maze(cfg.maze,cfg.seed+71000); t=Trainer(cfg,env.observation_shape)
    replay=Replay(); schedule=TrainingSchedule(t)
    out=prepare_run(a.out,cfg,{'arm':a.arm,'blocks':a.blocks,'block_expansions':a.block_expansions,'extra_updates':a.extra_updates,'extra_placement':a.extra_placement,
        'protocol':'Persistent visited raw observations and frontier; no identity checks or merging. Re-encode both and recompute exact priorities after training. Same full-epoch sample exposure and gradient-step counts at block boundaries; frequent arm causally partitions epoch into 10 intervals (variable minibatch sizes). No search restart after initialization.'})
    (out/'runner.py').write_text(Path(__file__).read_text())
    (out/'diagnostics_source.py').write_text(Path(__file__).with_name('noisy_frontier.py').read_text())
    t.save(out/'checkpoint-0.pt')
    initial_hash=hashlib.sha256(b''.join(x.detach().numpy().tobytes() for x in list(t.model.parameters())+list(t.dynamics.parameters()))).hexdigest()
    @torch.no_grad()
    def encode(obs): return t.model.encode(t.tensor(obs)).numpy()
    obs=env.reset(); position=env.position.copy(); z=encode(obs[None])[0]
    queue=PersistentDistanceFrontier(cfg.collect_steps,cfg.model.latent_dim,env.observation_shape,np.random.default_rng(cfg.seed+72000))
    # Evaluation-only physical graph and accounting, never inputs to priority/learning.
    points,_,index=reachable_graph(Maze(cfg.maze)); visited={tuple(position)}; expanded=set()
    first_full=None; milestones={}; cumulative_repeats=0; frames=[]; rows=[]
    rng=np.random.default_rng(cfg.seed+61000)
    samples_per_head=0; min_batch=10**9; max_batch=0
    start=time.monotonic(); search_seconds=training_seconds=refresh_seconds=0.
    with (out/'progress.jsonl').open('w') as progress,(out/'blocks.jsonl').open('w') as blocks,(out/'training.jsonl').open('w') as training_file:
        log_metrics(blocks,{'block':0,'interactions':0,'updates':0,**diagnostics(t,replay,points)})
        for block in range(a.blocks):
            old_count=len(replay); old_order=rng.permutation(old_count)
            block_expanded=set(); block_repeats=0
            interval=a.block_expansions if a.arm=='epoch1000' else a.block_expansions//10
            for offset in range(a.block_expansions):
                tic=time.monotonic()
                if block or offset: obs,position,z,score=queue.pop()
                else: score=0.
                key=tuple(position)
                cumulative_repeats+=key in expanded; block_repeats+=key in block_expanded
                expanded.add(key); block_expanded.add(key)
                queue.visit(obs,z)
                for action in range(5):
                    env.position=position.copy()
                    nxt,_=env.step(action); nxt_pos=env.position.copy()
                    replay.add(obs,action,nxt,len(replay),position,nxt_pos)
                    nz=encode(nxt[None])[0]
                    queue.add(nxt,nz,nxt_pos)
                    visited.add(tuple(nxt_pos))
                    for percent in [50,90,95,99,100]:
                        if len(visited)>=len(points)*percent/100 and str(percent) not in milestones:
                            milestones[str(percent)]=len(replay)
                    if first_full is None and len(visited)==len(points): first_full=len(replay)
                search_seconds+=time.monotonic()-tic
                n_expansions=block*a.block_expansions+offset+1
                if (offset+1)%interval==0:
                    replay.arrays(); tic=time.monotonic(); before=t.updates
                    if a.arm=='epoch1000':
                        metrics=schedule.train(replay)
                        samples_per_head+=len(replay)
                        min_batch=min(min_batch,(len(replay)-1)%cfg.train.batch_size+1)
                        max_batch=max(max_batch,min(len(replay),cfg.train.batch_size))
                    else:
                        j=(offset+1)//interval-1
                        batches=epoch_slices(old_count,old_count+j*interval*5,len(replay),j,10,rng,old_order,
                            (block+1)*a.block_expansions*5,cfg.train.batch_size)
                        for batch in batches:
                            metrics=t.update_probabilistic(replay,head_indices=[batch]*cfg.model.ensemble_size)
                            samples_per_head+=len(batch);min_batch=min(min_batch,len(batch));max_batch=max(max_batch,len(batch))
                        schedule.trained_through=len(replay)
                    if offset+1==a.block_expansions:
                        extra=(a.extra_updates if block==0 else 0) if a.extra_placement=='early' else (a.extra_updates*(block+1)//a.blocks-a.extra_updates*block//a.blocks)
                        for _ in range(extra):
                            metrics=t.update(replay)
                            samples_per_head+=cfg.train.batch_size
                    training_seconds+=time.monotonic()-tic
                    tic=time.monotonic();queue.refresh(encode);refresh_seconds+=time.monotonic()-tic
                    log_metrics(training_file,{'interactions':len(replay),'updates':t.updates,'block_updates':t.updates-before,
                        'samples_per_head':samples_per_head,'frontier_entries':int(queue.active.sum()),
                        'visited_entries':queue.n_visited,'training_seconds':training_seconds,'refresh_seconds':refresh_seconds})
                if n_expansions%50==0:
                    row={'block':block+1,'expansions':n_expansions,'interactions':len(replay),'visited':len(visited),
                        'coverage':len(visited)/len(points),'updates':t.updates,'expanded_physical_positions':len(expanded),
                        'cumulative_repeat_fraction':cumulative_repeats/n_expansions,
                        'block_repeat_fraction':block_repeats/(offset+1),'frontier_entries':int(queue.active.sum()),
                        'selected_priority':score,'seconds':time.monotonic()-start}
                    log_metrics(progress,row)
                    frames.append((len(replay),position.copy(),np.array(sorted(visited))))
            row.update(samples_per_head=samples_per_head,first_full=first_full,milestones=dict(milestones),
                training_seconds=training_seconds,refresh_seconds=refresh_seconds,search_seconds=search_seconds,
                **diagnostics(t,replay,points))
            rows.append(dict(row)); log_metrics(blocks,row);print(json.dumps(row),flush=True)
            if block+1 in {1,5,10,a.blocks}:t.save(out/f'checkpoint-{block+1}.pt')
    t.save(out/'checkpoint.pt');replay.save(out/'replay.npz',cfg.maze)
    mask=np.zeros((len(frames),len(points)),dtype=bool)
    for i,f in enumerate(frames):mask[i,[index[tuple(np.round(x,5))] for x in f[2]]]=True
    np.savez_compressed(out/'frames.npz',interactions=[f[0] for f in frames],position=[f[1] for f in frames],visited=mask,points=points)
    ids=np.flatnonzero(queue.active)
    np.savez_compressed(out/'frontier.npz',observations=queue.observations[ids],positions=np.array(queue.entries)[ids],
        priorities=queue.priority[ids],latents=queue.latents[ids],visited_observations=queue.visited_observations[:queue.n_visited])
    result={'arm':a.arm,'seed':a.seed,'extra_updates':a.extra_updates,'extra_placement':a.extra_placement,'initial_parameters_sha256':initial_hash,'reachable_states':len(points),
        'interactions':len(replay),'updates':t.updates,'samples_per_head':samples_per_head,'batch_size_range':[min_batch,max_batch],
        'visited':len(visited),'coverage':len(visited)/len(points),'first_full':first_full,'milestones':milestones,
        'trained_through':schedule.trained_through,'frontier_entries':int(queue.active.sum()),'visited_entries':queue.n_visited,
        'training_seconds':training_seconds,'refresh_seconds':refresh_seconds,'search_seconds':search_seconds,
        'seconds':time.monotonic()-start,'blocks':rows}
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
