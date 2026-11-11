"""Fresh-noise transition prediction probes, balanced over physical edges.

Oracle labels/centroids are evaluation only. This measures noise generalization
on seen edges separately from prediction on edges absent from the source prefix.
It is not exact observed-latent memorization accuracy.
"""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree
from map_learning.training import Trainer,seed_all
from map_learning.maze import Maze


@torch.no_grad()
def audit(path):
    row=json.loads(path.read_text());data=dict(np.load(Path(str(path)[:-5]+'.npz')))
    t=Trainer.load(row['checkpoint']);env=Maze(t.config.maze);points=data['points'];adj=data['adjacency'];n=len(points)
    index={tuple(np.round(p,5)):i for i,p in enumerate(points)}
    seen=np.zeros((n,5),bool)
    with np.load(Path(row['source'])/'replay.npz') as replay:
        src=np.array([index[tuple(np.round(p,5))] for p in replay['positions'][:row['prefix']]])
        seen[src,replay['actions'][:row['prefix']]]=True
    tree=cKDTree(data['centroids']);rng=np.random.default_rng(90477);result={'audit':str(path),'groups':{}}
    for name,mask in [('seen_edges',seen),('untried_edges',~seen)]:
        pairs=np.argwhere(mask)
        if not len(pairs):continue
        pairs=pairs[rng.choice(len(pairs),size=min(1024,len(pairs)),replace=False)]
        pairs=np.repeat(pairs,4,axis=0);pos=points[pairs[:,0]];actions=pairs[:,1];target=adj[pairs[:,0],actions]
        obs=np.concatenate([2*pos/t.config.maze.size-1,rng.normal(size=(len(pos),t.config.maze.nuisance_dims))],axis=1).astype(np.float32)
        pred=[]
        for i in range(0,len(obs),512):
            z=t.model.encode(t.tensor(obs[i:i+512]));pred.append(t.dynamics(z,torch.as_tensor(actions[i:i+512])).numpy())
        pred=np.concatenate(pred,axis=1);mean=pred.mean(0)
        mean_ids=tree.query(mean)[1];head_ids=np.stack([tree.query(x)[1] for x in pred])
        blocked=target==pairs[:,0];metrics={'probes':len(pairs),'physical_edges':len(pairs)//4}
        for label,subset in [('all',np.ones(len(target),bool)),('blocked',blocked),('moving',~blocked)]:
            if subset.any():metrics[label]={'n':int(subset.sum()),'mean_prediction_accuracy':float((mean_ids[subset]==target[subset]).mean()),
                                            'member_accuracies':(head_ids[:,subset]==target[None,subset]).mean(1).tolist()}
        result['groups'][name]=metrics
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('audits',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args();seed_all(0,1)
    rows=[]
    for path in sorted(a.audits.glob('*-at*.json')):
        rows.append(audit(path));a.out.write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps({'audits':len(rows),'out':str(a.out)}))

if __name__=='__main__':main()
