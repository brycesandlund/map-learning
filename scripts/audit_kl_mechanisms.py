"""Audit immutable paired training and independent shared-archive geometry."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from map_learning.training import Trainer,seed_all
from map_learning.maze import Maze,reachable_graph
from audit_frontier_efficiency import reconstruct,encode
from scipy.spatial import cKDTree
from kl_mechanisms import ARMS,data_source,digest_model
from audit_moment_prior import full_moments


@torch.no_grad()
def geometry(t):
    env=Maze(t.config.maze,t.config.seed+931000);rng=np.random.default_rng(t.config.seed+932000)
    points,_,_=reachable_graph(env);p=points[rng.choice(len(points),64,replace=False)]
    obs=np.stack([env.observe(x) for x in p for _ in range(16)])
    z=encode(t,obs).reshape(64,16,-1);centers=z.mean(1)
    nxt=np.array([[env.transition(x,a) for a in range(4)] for x in p])
    no=np.repeat(obs.reshape(64,16,-1)[:,None],4,axis=1)
    no[...,:2]=(2*nxt/env.config.size-1)[:,:,None]
    nz=encode(t,no.reshape(-1,10)).reshape(64,4,16,-1).mean(2)
    moving=np.any(nxt!=p[:,None],axis=-1);dist=np.linalg.norm(nz-centers[:,None],axis=-1)
    local=np.sum(dist*moving,axis=1)/moving.sum(1)
    noise=np.linalg.norm(z[:,:8]-z[:,8:],axis=-1)
    quant=lambda v:np.quantile(v,[.5,.9,.99,1]).tolist()
    covariance=np.cov(centers,rowvar=False);eigen=np.linalg.eigvalsh(covariance).clip(0)
    return {'noise_distance_quantiles':quant(noise),'neighbor_distance_quantiles':quant(dist[moving]),
            'noise_to_local_move_quantiles':quant(noise/np.maximum(local[:,None],1e-12)),
            'centroid_covariance_eigenvalues':eigen.tolist(),'effective_dimension':float(eigen.sum()**2/max((eigen**2).sum(),1e-24))}


def audit_fixed(root):
    jobs=json.loads((root/'manifest.json').read_text())['jobs'];groups=defaultdict(list);rows=[]
    for j in jobs:groups[j['family'],j['seed']].append(j)
    for key,jj in groups.items():
        hashes=[];batchhashes=[];sources=[];rngs=[]
        for j in jj:
            d=Path(j['path']);r=json.loads((d/'result.json').read_text());t=Trainer.load(d/'checkpoint.pt')
            assert r['updates']==4112 and r['samples_per_head']==1050000 and r['interactions']==100000
            assert t.config.train.repulsion_weight==.03
            assert t.spec['beta']==ARMS[j['arm']]['beta'] and t.spec['dynamics_input']==ARMS[j['arm']]['dynamics_input']
            expected_steps=4112*t.spec['prior_steps'] if t.spec['beta'] else 0
            assert t.prior_updates==expected_steps
            hashes.append(r['initial_parameters_sha256']);batchhashes.append(r['batch_indices_sha256']);sources.append(r['source_replay_sha256'])
            rngs.append((json.dumps(t.rng.bit_generator.state,sort_keys=True),t.sample_rng.get_state()))
            row={**j,'checks_passed':True,'final_geometry':geometry(t),'prior_updates':t.prior_updates,
                 'source_replay_sha256':r['source_replay_sha256'],'batch_indices_sha256':r['batch_indices_sha256']}
            # Recompute 35k rankings from raw archive independently of saved stats.
            mid=Trainer.load(d/'checkpoint-7.pt')
            with np.load(Path(r['source'])/'replay.npz') as data:
                obs,pos,vo,vp=reconstruct(data,35000);points,adj,index=reachable_graph(Maze(mid.config.maze))
                ids=lambda ps:np.array([index[tuple(np.round(p,5))] for p in ps])
                seen=np.zeros(len(points),bool);seen[ids(data['positions'][:1])]=True;seen[ids(data['next_positions'][:35000])]=True
                score=cKDTree(encode(mid,vo)).query(encode(mid,obs))[0];order=np.argsort(-score,kind='stable');good=(~seen[adj[ids(pos)]]).any(1)
                q=next(q for q in r['queue'] if q['prefix']==35000)
                assert q['seen_states']==int(seen.sum())
                assert q['stats']['top100_productive_fraction']==float(good[order[:100]].mean())
                assert len(q['live'])==3 and all(x['rows'][-1]['k']==128 for x in q['live'])
                row['mid_geometry']=geometry(mid)
            rows.append(row)
        assert len(set(hashes))==len(set(batchhashes))==len(set(sources))==1
        assert all(x[0]==rngs[0][0] and torch.equal(x[1],rngs[0][1]) for x in rngs)
    return rows


def audit_online(root):
    jobs=json.loads((root/'manifest.json').read_text())['jobs'];groups=defaultdict(list);rows=[]
    for j in jobs:groups[j['seed']].append(j)
    for seed,jj in groups.items():
        base=Path(jj[0]['path']);b=np.load(base/'replay.npz');initial=Trainer.load(base/'checkpoint-0.pt')
        final_rngs=[]
        for j in jj:
            d=Path(j['path']);r=json.loads((d/'result.json').read_text());t=Trainer.load(d/'checkpoint.pt');init=Trainer.load(d/'checkpoint-0.pt')
            assert digest_model(init)==digest_model(initial)
            assert r['updates']==4112 and r['samples_per_head']==1050000 and r['interactions']==100000
            assert t.config.train.repulsion_weight==.03 and t.spec['beta']==ARMS[j['arm']]['beta']
            assert t.prior_updates==(4112*t.spec['prior_steps'] if t.spec['beta'] else 0)
            final_rngs.append((json.dumps(t.rng.bit_generator.state,sort_keys=True),t.sample_rng.get_state()))
            with np.load(d/'replay.npz') as data:
                for key in ['observations','actions','next_observations','positions','next_positions']:np.testing.assert_array_equal(data[key][:500],b[key][:500])
                np.testing.assert_array_equal(data['next_observations'][:,2:],b['next_observations'][:,2:])
                obs,pos,vo,vp=reconstruct(data,100000);z=encode(t,obs);v=encode(t,vo);scores=cKDTree(v).query(z)[0]
                with np.load(d/'frontier.npz') as f:
                    np.testing.assert_array_equal(obs,f['observations']);np.testing.assert_array_equal(z,f['latents'])
                    np.testing.assert_allclose(scores,f['priorities'],rtol=1e-12,atol=1e-12)
                moments=full_moments(t,data,100000)
            rows.append({**j,'checks_passed':True,'final_geometry':geometry(t),'prior_updates':t.prior_updates,'moments':moments})
        assert all(x[0]==final_rngs[0][0] and torch.equal(x[1],final_rngs[0][1]) for x in final_rngs)
        b.close()
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--fixed',type=Path);p.add_argument('--online',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args();seed_all(0,1)
    result={}
    if a.fixed:result['fixed']=audit_fixed(a.fixed)
    if a.online:result['online']=audit_online(a.online)
    a.out.write_text(json.dumps(result,indent=2)+'\n');print({k:len(v) for k,v in result.items()})

if __name__=='__main__':main()
