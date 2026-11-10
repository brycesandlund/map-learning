"""Does imagined return rank the same candidate paths as real-state reward?

Pure post-hoc diagnostic, not an oracle-assisted agent. A common bank of 64
persistent-random action sequences per visited start is evaluated both through
the frozen learned model and through true maze transitions. Both are scored by
that same frozen ensemble; true reward here is NOT environment novelty.
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
def audit(trainer, replay, starts=16, candidates=64, horizon=128):
    env=Maze(trainer.config.maze);points,adj,index=reachable_graph(env)
    z=trainer.model.encode(trainer.tensor(np.stack([env.observe(p) for p in points])))
    rng=np.random.default_rng(10924)
    visited=np.unique([index[tuple(p)] for p in replay['positions']])
    real=np.repeat(rng.choice(visited,starts),candidates);imagined=z[real].clone()
    center=z.mean(0)
    radius_scale=z.var(0,unbiased=False).mean().sqrt().clamp_min(1e-12)
    oracle_radius=float(((z-center).square().mean(-1).sqrt()/radius_scale).max())
    peak_radius=np.zeros(len(real))
    discount=trainer.config.planner.uncertainty_discount
    score=np.zeros(len(real));truth=score.copy();previous=rng.integers(5,size=len(real));rows=[]
    for depth in range(1,horizon+1):
        previous=np.where(rng.random(len(real))<.8,previous,rng.integers(5,size=len(real)))
        action=torch.as_tensor(previous)
        pred=mixture_mean(trainer.dynamics.distribution(imagined,action))
        true=mixture_mean(trainer.dynamics.distribution(z[real],action))
        score+=discount**(depth-1)*pred.var(0,unbiased=False).mean(-1).numpy()
        truth+=discount**(depth-1)*true.var(0,unbiased=False).mean(-1).numpy()
        imagined=pred.mean(0);real=adj[real,previous]
        radius=((imagined-center).square().mean(-1).sqrt()/radius_scale).numpy()
        peak_radius=np.maximum(peak_radius,radius)
        if depth not in {4,8,16,32,64,128}:continue
        a=score.reshape(starts,candidates);b=truth.reshape(starts,candidates)
        chosen=a.argmax(1);ii=np.arange(starts)
        # Pairwise ordering accuracy excludes tied true returns (often wall stays).
        delta=b[:,:,None]-b[:,None,:];pred_delta=a[:,:,None]-a[:,None,:]
        mask=np.triu(np.ones((candidates,candidates),bool),1)[None]&(delta!=0)
        concordance=np.where(pred_delta==0,.5,(pred_delta*delta>0).astype(float))
        rows.append({'horizon':depth,'pairwise_order_accuracy':float(concordance[mask].mean()),
            'selected_real_return_over_best':float(np.mean(b[ii,chosen]/np.maximum(b.max(1),1e-12))),
            'random_real_return_over_best':float(np.mean(b.mean(1)/np.maximum(b.max(1),1e-12))),
            'selected_imagined_over_real_return':float(np.mean(a[ii,chosen]/np.maximum(b[ii,chosen],1e-12))),
            'selected_optimism_median':float(np.median(a[ii,chosen]/np.maximum(b[ii,chosen],1e-12))),
            'selected_optimism_max':float(np.max(a[ii,chosen]/np.maximum(b[ii,chosen],1e-12))),
            'selected_imagined_sum_over_real_sum':float(a[ii,chosen].sum()/max(b[ii,chosen].sum(),1e-12)),
            'selected_max_radius_over_oracle_max':float(peak_radius.reshape(starts,candidates)[ii,chosen].max()/oracle_radius),
            'all_imagined_over_real_return':float(a.mean()/max(b.mean(),1e-12))})
    return {'starts':starts,'candidates':candidates,'discount':discount,'rows':rows}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('runs',nargs='+',type=Path);args=p.parse_args()
    seed_all(0,1)
    for run in args.runs:
        trainer=Trainer.load(run/'checkpoint.pt');replay=dict(np.load(run/'replay.npz'))
        result=audit(trainer,replay);out=run/'reward-ranking-audit.json'
        out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(out,flush=True)

if __name__=='__main__':main()
