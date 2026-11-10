"""Evaluation-only open-loop rollout error, reward hallucination and manifold proximity.

Replay segments use ONLY actually experienced action sequences without resets.
Greedy/random paths start at visited states but can cross unseen edges. The oracle
is used for scoring, never fed back into imagined trajectories or training.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from map_learning.training import Trainer, seed_all
from map_learning.maze import Maze, reachable_graph
from map_learning.probabilistic import mixture_mean


@torch.no_grad()
def audit(trainer, replay, cutoff, horizon=128, batch=96):
    env=Maze(trainer.config.maze);points,adj,index=reachable_graph(env)
    obs=np.stack([env.observe(p) for p in points]);z=trainer.model.encode(trainer.tensor(obs))
    variance=max(float(z.var(0,unbiased=False).mean()),1e-12)
    adjacent=(z[torch.as_tensor(adj[:,:4])]-z[:,None]).square().mean(-1)
    spacing=max(float(adjacent[adjacent>0].median()),1e-12)
    source=np.array([index[tuple(p)] for p in replay['positions'][:cutoff]])
    actions=replay['actions'][:cutoff];episodes=replay['episodes'][:cutoff]
    seen=np.zeros((len(points),5),bool);seen[source,actions]=True
    rng=np.random.default_rng(8192)
    valid=np.arange(cutoff-horizon+1)
    valid=valid[episodes[valid]==episodes[valid+horizon-1]]
    if not len(valid):raise ValueError('Not enough reset-free replay')
    starts=rng.choice(valid,batch)
    visited=rng.choice(np.unique(source),batch)
    depths={1,2,4,8,16,32,64,128,horizon}
    result={}
    for mode in ['replay','persistent_random','greedy_reward']:
        real=source[starts].copy() if mode=='replay' else visited.copy()
        imagined=z[real].clone();origin=imagined.clone();known=np.ones(batch,bool)
        previous=rng.integers(5,size=batch)
        rows=[]
        for depth in range(1,horizon+1):
            if mode=='greedy_reward':
                params=trainer.dynamics.distribution(imagined.repeat_interleave(5,0),torch.arange(5).repeat(batch))
                means=mixture_mean(params).reshape(len(trainer.dynamics.heads),batch,5,-1)
                scores=means.var(0,unbiased=False).mean(-1)
                action=scores.argmax(-1).numpy()
                predicted=means[:,np.arange(batch),action]
            else:
                if mode=='replay':action=actions[starts+depth-1]
                else:
                    previous=np.where(rng.random(batch)<.8,previous,rng.integers(5,size=batch));action=previous
                predicted=mixture_mean(trainer.dynamics.distribution(imagined,torch.as_tensor(action)))
            teacher=mixture_mean(trainer.dynamics.distribution(z[real],torch.as_tensor(action)))
            known &= seen[real,action]
            real=adj[real,action];imagined=predicted.mean(0);target=z[real]
            if not torch.isfinite(imagined).all():
                rows.append({'depth':depth,'nonfinite':True});break
            if depth not in depths:continue
            squared=torch.cdist(imagined,z).square()/z.shape[-1]
            nearest=squared.argmin(-1).numpy()
            error=(imagined-target).square().mean(-1)
            reward=predicted.var(0,unbiased=False).mean(-1)
            actual=teacher.var(0,unbiased=False).mean(-1)
            pos_error=np.linalg.norm(points[nearest]-points[real],axis=-1)/env.config.step_size
            row={'depth':depth,'nonfinite':False,'normalized_mse':float(error.mean()/variance),
                 'teacher_forced_mse':float((teacher.mean(0)-target).square().mean()/variance),
                 'persistence_mse':float((origin-target).square().mean()/variance),
                 'nearest_state_accuracy':float(np.mean(nearest==real)),
                 'position_error_steps_mean':float(pos_error.mean()),'position_error_steps_median':float(np.median(pos_error)),
                 'manifold_distance_in_edge_units':float((squared.min(-1).values/spacing).sqrt().mean()),
                 'imagined_reward':float(reward.mean()/variance),'real_state_reward':float(actual.mean()/variance),
                 'reward_ratio':float(reward.mean()/actual.mean().clamp_min(1e-12)),
                 'known_prefix_fraction':float(known.mean()),
                 'known_prefix_mse':float(error[known].mean()/variance) if known.any() else None,
                 'stayed_at_origin_fraction':float((target==origin).all(-1).float().mean())}
            rows.append(row)
        result[mode]=rows
    return {'cutoff':cutoff,'batch':batch,'horizon':horizon,'latent_variance':variance,'median_edge_mse':spacing,'curves':result}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('runs',nargs='+',type=Path)
    p.add_argument('--intermediate',action='store_true');p.add_argument('--horizon',type=int,default=128)
    p.add_argument('--batch',type=int,default=96);args=p.parse_args()
    seed_all(0,1)
    for run in args.runs:
        replay=dict(np.load(run/'replay.npz'))
        checkpoints=sorted(run.glob('checkpoint-*.pt'),key=lambda p:int(p.stem.split('-')[1])) if args.intermediate else [run/'checkpoint.pt']
        results=[]
        for checkpoint in checkpoints:
            cutoff=int(checkpoint.stem.split('-')[1]) if '-' in checkpoint.stem else len(replay['actions'])
            trainer=Trainer.load(checkpoint);row=audit(trainer,replay,cutoff,args.horizon,args.batch)
            row['checkpoint']=checkpoint.name;results.append(row)
        output=run/('rollout-audit-history.json' if args.intermediate else 'rollout-audit.json')
        output.write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
        print(output,flush=True)

if __name__=='__main__':main()
