"""Recompute acceptance and latent geometry from saved joint-model weights."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from map_learning.config import ModelConfig, MazeConfig
from map_learning.models import MapModel
from map_learning.probabilistic import MixtureDynamicsEnsemble
from map_learning.training import seed_all
from map_learning.replay import Replay
from map_learning.maze import Maze, reachable_graph


@torch.no_grad()
def verify(path):
    info=json.loads((path/'experiment.json').read_text())
    assert info['encoder']=='joint' and info['encoder_initialization']=='fresh'
    assert info['subset']==0 and info['uniform_edges']
    assert hashlib.sha256(Path(info['data']).read_bytes()).hexdigest()==info['dataset_sha256']
    weights=torch.load(path/'weights.pt',map_location='cpu',weights_only=False)
    encoder=MapModel((2,),ModelConfig(**info['encoder_config']),'probabilistic',20)
    dynamics=MixtureDynamicsEnsemble(ModelConfig(**info['dynamics_config']))
    encoder.load_state_dict(weights['encoder']);dynamics.load_state_dict(weights['dynamics'])
    replay,maze=Replay.load(info['data']);data=replay.arrays()
    idx=np.load(path/'training_indices.npy')
    x=data['observations'][idx];y=data['next_observations'][idx];a=data['actions'][idx]
    keys=np.concatenate([x,a[:,None]],1)
    assert len(np.unique(keys,axis=0))==len(idx)
    assert len(idx)==len(np.unique(np.concatenate([data['observations'],data['actions'][:,None]],1),axis=0))
    env=Maze(MazeConfig(**maze));points,_,lookup=reachable_graph(env)
    zcat=encoder.encode(torch.tensor(np.stack([env.observe(p) for p in points])))
    target=np.array([lookup[tuple(np.round(p,5))] for p in data['next_positions'][idx]])
    correct=[];ens_correct=[];errors=[]
    for offset in range(0,len(idx),128):
        sl=slice(offset,offset+128)
        z=encoder.encode(torch.tensor(x[sl]));truth=encoder.encode(torch.tensor(y[sl]))
        means=dynamics(z,torch.tensor(a[sl]))
        nearest=torch.cdist(means.reshape(-1,means.shape[-1]),zcat,compute_mode='donot_use_mm_for_euclid_dist').argmin(-1).reshape(len(dynamics.heads),-1).numpy()
        correct.append(nearest==target[sl][None])
        nearest_ens=torch.cdist(means.mean(0),zcat,compute_mode='donot_use_mm_for_euclid_dist').argmin(-1).numpy()
        ens_correct.extend((nearest_ens==target[sl]).tolist())
        errors.extend((means.mean(0)-truth).square().mean(-1).tolist())
    correct=np.concatenate(correct,1);accuracies=correct.mean(1)
    observations,inverse,counts=np.unique(np.concatenate([x,y]),axis=0,return_inverse=True,return_counts=True)
    z=encoder.encode(torch.tensor(observations));distance=torch.cdist(z,z,compute_mode='donot_use_mm_for_euclid_dist').double()
    probabilities=torch.tensor(counts/counts.sum())
    weights=probabilities[:,None]*probabilities[None,:]
    hinge=(1-distance.square()).clamp_min(0)
    floor=float(probabilities.square().sum())
    repulsion=float((weights*hinge).sum())
    mask=~torch.eye(len(z),dtype=torch.bool)
    distinct=distance[mask]
    edge=(encoder.encode(torch.tensor(x))-encoder.encode(torch.tensor(y))).norm(dim=-1)
    moving=np.any(x!=y,axis=1)
    metrics=json.loads((path/'metrics.json').read_text())
    np.testing.assert_allclose(accuracies,metrics['train']['per_member_unique_accuracy'],atol=0,rtol=0)
    result={'run':str(path.resolve()),'updates':metrics['update'],'unique_training_edges':len(idx),
        'per_member_accuracy':accuracies.tolist(),'ensemble_accuracy':float(np.mean(ens_correct)),
        'all_members_correct_fraction':float(correct.all(0).mean()),
        'wrong_edges_per_member':(~correct).sum(1).tolist(),
        'ensemble_mse':float(np.mean(errors)),'ensemble_max_mse':float(np.max(errors)),
        'training_state_count':len(z),'distinct_float32_latents':len(np.unique(z.numpy(),axis=0)),
        'training_latent_variance':float(z.var(0,unbiased=False).mean()),
        'min_distinct_distance':float(distinct.min()),'mean_distinct_distance':float(distinct.mean()),
        'p10_median_p90_distinct_distance':distinct.quantile(torch.tensor([.1,.5,.9],dtype=torch.float64)).tolist(),
        'mean_moving_edge_distance':float(edge[moving].mean()),
        'repulsion_exact':repulsion,'identical_state_repulsion_floor':floor,
        'repulsion_distinct_pairs_conditional':(repulsion-floor)/(1-floor),
        'probability_below_margin':float((weights*(distance<1)).sum()),
        'encoder_drift_mse':metrics['encoder_drift_mse'],
        'unseen_accuracy':metrics['unseen']['exact_retrieval'],
        'unseen_per_member_accuracy':metrics['unseen']['per_member_unique_accuracy']}
    result['passed']=bool(min(accuracies)>=.98 and result['distinct_float32_latents']==len(z)
                          and result['training_latent_variance']>1e-4 and result['encoder_drift_mse']>0)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',nargs='+',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    args=p.parse_args();seed_all(123)
    rows=[]
    for root in args.runs:
        for item in json.loads((root/'summary.json').read_text()):rows.append(verify(Path(item['run'])))
    args.out.write_text(json.dumps(rows,indent=2)+'\n')
    for row in rows:print(Path(row['run']).name,row['passed'],row['per_member_accuracy'])


if __name__=='__main__':main()
