"""Compare encoder gradient scales at the common memorization initialization."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from memorize_dynamics import prediction_loss
from map_learning.training import Trainer, seed_all
from map_learning.replay import Replay
from map_learning.probabilistic import MixtureDynamicsEnsemble, repulsion_loss


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True,type=Path)
    p.add_argument('--data',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    args=p.parse_args()
    trainer=Trainer.load(args.checkpoint);replay,_=Replay.load(args.data)
    seed_all(7);dynamics=MixtureDynamicsEnsemble(trainer.config.model)
    rng=np.random.default_rng(7);rows=[]
    for _ in range(32):
        obs,a,nxt=replay.sample_transitions(128,rng)
        z,target=trainer.model.encode(trainer.tensor(np.concatenate([obs,nxt]))).chunk(2)
        params=dynamics.distribution(z,torch.as_tensor(a))
        left,right=trainer.model.encode(trainer.tensor(replay.sample_states(256,rng))).chunk(2)
        losses={'mse':prediction_loss(params,target,'mse'),'nll':prediction_loss(params,target,'nll'),
                'repulsion_unweighted':repulsion_loss(left,right,1.)}
        row={}
        for name,loss in losses.items():
            gradients=torch.autograd.grad(loss,list(trainer.model.parameters()),retain_graph=True)
            row[name]=float(sum(g.square().sum() for g in gradients).sqrt())
        rows.append(row)
    summary={k:float(np.median([r[k] for r in rows])) for k in rows[0]}
    for weight,name,objective in [(10,'10','mse'),(.01,'001','mse'),(10,'10','nll')]:
        summary[f'median_repulsion{name}_to_{objective}_gradient_ratio']=float(np.median([weight*r['repulsion_unweighted']/r[objective] for r in rows]))
    args.out.write_text(json.dumps({'description':'32 shared minibatches across heads at common initial encoder and fresh seed-7 dynamics; encoder gradient norms, not post-training causal attribution',
                                    'summary':summary,'rows':rows},indent=2)+'\n')
    print(summary)


if __name__=='__main__':main()
