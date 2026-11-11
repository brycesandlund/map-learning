"""Fit distance or native dynamics encoder on an identical recorded online stream.

Replays the causal epoch slices exactly. No environment interactions are collected.
Occurrence ancestry is recovered from exact cached observation bytes (not physical
state identity); fresh nuisance draws make each acquired observation unique.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from map_learning.config import load_config
from map_learning.training import Trainer, seed_all
from map_learning.distance_learning import DistanceTrainer
from map_learning.replay import Replay
from map_learning.persistent_frontier import epoch_slices
from map_learning.experiment import prepare_run, log_metrics


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--arm',type=Path,required=True,help='Single arm dict (name/spec/l2/repulsion)')
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--blocks',type=int,default=20)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    arm=json.loads(a.arm.read_text())
    cfg=load_config(a.source/'config.json',[f'seed={a.seed}',f'collect_steps={a.blocks*5000}',
                   f'train.latent_l2={arm["l2"]}',f'train.repulsion_weight={arm["repulsion"]}'])
    seed_all(a.seed,cfg.threads)
    with np.load(a.source/'replay.npz') as f:data={k:f[k].copy() for k in ['observations','next_observations','actions','positions','next_positions']}
    if len(data['actions'])<a.blocks*5000:raise ValueError('Insufficient source replay')
    cls=DistanceTrainer if arm['spec'] else Trainer
    t=cls(cfg,data['observations'].shape[1:],arm['spec']) if arm['spec'] else cls(cfg,data['observations'].shape[1:])
    out=prepare_run(a.out,cfg,{'mode':'fixed-data-distance','source':str(a.source),'arm':arm})
    (out/'runner.py').write_text(Path(__file__).read_text())
    replay=Replay();rng=np.random.default_rng(a.seed+61000);lookup={data['observations'][0].tobytes():0}
    if arm['spec']:t.ancestry.root(data['observations'][0])
    t.save(out/'checkpoint-0.pt');samples=0;started=time.monotonic()
    with (out/'training.jsonl').open('w') as log:
        for block in range(a.blocks):
            old=len(replay);order=rng.permutation(old)
            for j in range(10):
                start=len(replay);end=start+500
                for i in range(start,end):
                    obs,nxt=data['observations'][i],data['next_observations'][i]
                    parent=lookup[obs.tobytes()]
                    if nxt.tobytes() in lookup:raise ValueError('Acquired observation collision; cannot reconstruct occurrence IDs')
                    if arm['spec']:assert t.ancestry.add(parent,nxt)==i+1
                    lookup[nxt.tobytes()]=i+1
                    replay.add(obs,int(data['actions'][i]),nxt,i,data['positions'][i],data['next_positions'][i])
                replay.arrays()
                batches=epoch_slices(old,start,end,j,10,rng,order,(block+1)*5000,cfg.train.batch_size)
                for batch in batches:
                    metrics=t.update_batch(replay,[batch]*cfg.model.ensemble_size);samples+=len(batch)
                log_metrics(log,{'interactions_in_source':end,'updates':t.updates,'samples_per_head':samples,'loss_metrics':metrics,'seconds':time.monotonic()-started})
            if block+1 in {1,5,10,15,a.blocks}:t.save(out/f'checkpoint-{block+1}.pt')
            print(json.dumps({'block':block+1,'updates':t.updates,'seconds':time.monotonic()-started}),flush=True)
    t.save(out/'checkpoint.pt')
    result={'source':str(a.source),'arm':arm,'seed':a.seed,'updates':t.updates,'samples_per_head':samples,'seconds':time.monotonic()-started}
    # Paired baseline reproduction check is exact on model AND dynamics weights.
    source_checkpoint=a.source/f'checkpoint-{a.blocks}.pt'
    source_cfg=load_config(a.source/'config.json')
    applicable=(arm['spec'] is None and cfg.seed==source_cfg.seed and cfg.train==source_cfg.train
                and cfg.model==source_cfg.model and cfg.maze==source_cfg.maze)
    result['control_reproduction_applicable']=applicable
    if applicable:
        ref=Trainer.load(source_checkpoint)
        result['encoder_exact_reproduction']=all(torch.equal(x,y) for x,y in zip(t.model.parameters(),ref.model.parameters()))
        result['dynamics_exact_reproduction']=all(torch.equal(x,y) for x,y in zip(t.dynamics.parameters(),ref.dynamics.parameters()))
        if not result['encoder_exact_reproduction'] or not result['dynamics_exact_reproduction']:raise AssertionError('Control reproduction mismatch')
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
