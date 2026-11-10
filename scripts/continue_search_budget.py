"""Controlled online branches from trained checkpoints; fresh matched planner/schedule RNGs."""
import argparse
from pathlib import Path
import json
import time
import numpy as np
import torch
from map_learning.training import Trainer, seed_all
from map_learning.replay import Replay
from map_learning.maze import Maze, reachable_graph
from map_learning.cadence import TrainingSchedule
from map_learning.uncertainty_planning import make_planner
from map_learning.reward_snapshots import RewardSnapshots
from map_learning.experiment import prepare_run, log_metrics


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--simulations',type=int,default=24);p.add_argument('--steps',type=int,default=1200);p.add_argument('--random',action='store_true');a=p.parse_args()
    seed_all(0,1)
    t=Trainer.load(a.source/'checkpoint.pt');cfg=t.config
    replay,_=Replay.load(a.source/'replay.npz');prefix=len(replay);initial_updates=t.updates
    cfg.collect_steps=prefix+a.steps;cfg.planner.simulations=a.simulations
    if a.random:cfg.planner.behavior='random'
    cfg.reward_snapshot_every=100
    out=prepare_run(a.out,cfg,{'source':str(a.source.resolve()),'prefix':prefix,'steps':a.steps,
        'protocol':'Restore model, optimizer, trainer RNG. Restart planner, shuffle, scale RNG identically per branch; reset at center. No warmup.'})
    env=Maze(cfg.maze,cfg.seed);obs=env.reset();planner=make_planner(t,cfg.planner,cfg.seed+18000)
    schedule=TrainingSchedule(t);schedule.trained_through=prefix
    scale_rng=np.random.default_rng(cfg.seed+19000)
    def scale_value():
        with torch.no_grad():return t.model.encode(t.tensor(replay.sample_states(256,scale_rng))).var(0,unbiased=False).mean().clamp_min(1e-8)
    scale=scale_value();snapshots=RewardSnapshots(t)
    visited={tuple(p) for p in replay.positions+replay.next_positions};initial=set(visited)
    episode=int(max(replay.episodes))+1;start=time.monotonic();train_seconds=0.;search_seconds=0.;new_auc=0.
    with (out/'exploration.jsonl').open('w') as f:
        for step in range(a.steps):
            tic=time.monotonic();action=planner.act(obs,latent_scale=scale);search_seconds+=time.monotonic()-tic
            position=env.position.copy();snapshots.record(prefix+step,position,obs,action,scale,planner.last_metrics,True)
            nxt,done=env.step(action);replay.add(obs,action,nxt,episode,position,env.position);obs=nxt
            visited.add(tuple(env.position));new_auc+=len(visited-initial)
            if (step+1)%cfg.train_every==0 or step+1==a.steps:
                tic=time.monotonic();schedule.train(replay);train_seconds+=time.monotonic()-tic;scale=scale_value()
            if (step+1)%100==0:
                row={'step':step+1,'environment_step':prefix+step+1,'new_states':len(visited-initial),'visited_positions':len(visited),
                     'updates':t.updates,'seconds':time.monotonic()-start,'training_seconds':train_seconds,'search_seconds':search_seconds}
                log_metrics(f,row);print(json.dumps(row),flush=True)
            if done:obs=env.reset();planner.reset();episode+=1
    snapshots.save(out);t.save(out/'checkpoint.pt');replay.save(out/'replay.npz',cfg.maze)
    tail=replay.arrays();x=tail['positions'][prefix:];y=tail['next_positions'][prefix:]
    back=np.all(y[1:]==x[:-1],axis=1)&np.any(y[1:]!=x[1:],axis=1)&(tail['episodes'][prefix+1:]==tail['episodes'][prefix:-1])
    result={'source':str(a.source),'seed':cfg.seed,'simulations':a.simulations,'behavior':cfg.planner.behavior,
        'prefix':prefix,'additional_steps':a.steps,'initial_visited':len(initial),'final_visited':len(visited),'new_states':len(visited-initial),
        'mean_cumulative_new_states':new_auc/a.steps,'stationary_fraction':float(np.all(x==y,axis=1).mean()),
        'immediate_backtrack_fraction':float(back.mean()),'seconds':time.monotonic()-start,'training_seconds':train_seconds,
        'search_seconds':search_seconds,'additional_updates':t.updates-initial_updates,'trained_through':schedule.trained_through}
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
