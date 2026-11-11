"""Rank-density and top-k counterfactuals. Physical labels are evaluation only."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from map_learning.training import Trainer,seed_all
from map_learning.maze import Maze,reachable_graph
from map_learning.persistent_frontier import PersistentDistanceFrontier
from audit_frontier_efficiency import reconstruct,encode

KS=[1,4,8,16,32,64,128,256,1024]
BINS=[(1,10),(11,100),(101,1000),(1001,10000),(10001,1000000)]


def counts(chosen,fid,expanded,seen,adj):
    physical=fid[chosen];unique=np.unique(physical)
    productive=(~seen[adj[physical]]).any(1)
    successors=np.unique(adj[physical])
    return {'k':len(chosen),'repeat_fraction':float(expanded[physical].mean()),
        'productive_fraction':float(productive.mean()),'unique_positions':len(unique),
        'unexpanded_unique_positions':int((~expanded[unique]).sum()),
        'productive_unique_positions':int((~seen[adj[unique]]).any(1).sum()),
        'new_states':int((~seen[successors]).sum())}


def select_diverse(z,scores,k,rng):
    """Greedy latent-distance updates within a batch; no labels or new children."""
    priorities=scores.copy();active=np.ones(len(z),bool);chosen=[]
    for _ in range(k):
        ids=np.flatnonzero(active);best=priorities[ids].max()
        i=int(rng.choice(ids[priorities[ids]==best]));chosen.append(i);active[i]=False
        priorities=np.minimum(priorities,np.linalg.norm(z-z[i],axis=1))
    return np.array(chosen)


def live_sequence(t,obs,pos,vo,scores,z,v,seen,expanded,points,index,k,noise_seed):
    env=Maze(t.config.maze,noise_seed);q=PersistentDistanceFrontier(len(obs)+len(vo)+5*k+1,z.shape[1],obs.shape[1:],np.random.default_rng(noise_seed+1))
    q.n_visited=len(vo);q.visited_observations[:len(vo)]=vo;q.visited[:len(vo)]=v
    q.tree=cKDTree(v.copy());q.indexed_through=len(vo)
    q.entries=list(pos.copy());q.observations[:len(obs)]=obs;q.latents[:len(obs)]=z
    q.priority[:len(obs)]=scores;q.active[:len(obs)]=True
    initial_seen=seen.copy();exp=expanded.copy();selected=[];rows=[]
    for step in range(k):
        o,p,latent,_=q.pop();pid=index[tuple(np.round(p,5))];selected.append(pid);q.visit(o,latent)
        for action in range(5):
            env.position=p.copy();nxt,_=env.step(action);newp=env.position.copy()
            seen[index[tuple(np.round(newp,5))]]=True
            q.add(nxt,encode(t,nxt[None])[0],newp)
        exp[pid]=True
        if step+1 in KS:
            unique=np.unique(selected)
            rows.append({'k':step+1,'new_states':int((seen&~initial_seen).sum()),'unique_positions':len(unique),
                'unexpanded_unique_positions':int((~expanded[unique]).sum()),'repeat_fraction':float(expanded[selected].mean())})
    return rows


def audit(source,checkpoint,condition,seed,block,out,run_dynamic):
    prefix=block*5000;t=Trainer.load(checkpoint);data=np.load(source/'replay.npz')
    obs,pos,vo,vp=reconstruct(data,prefix);env=Maze(t.config.maze);points,adj,index=reachable_graph(env)
    ids=lambda ps:np.array([index[tuple(np.round(p,5))] for p in ps])
    fid=ids(pos);expanded=np.zeros(len(points),bool);expanded[ids(vp)]=True
    seen=np.zeros(len(points),bool);seen[0]=True;seen[ids(data['next_positions'][:prefix])]=True
    z=encode(t,obs).astype(float);v=encode(t,vo).astype(float);scores=cKDTree(v).query(z,eps=0)[0]
    order=np.argsort(-scores,kind='stable')
    bins=[]
    for lo,hi in BINS:
        if lo>len(order):continue
        choice=order[lo-1:min(hi,len(order))]
        bins.append({'rank_low':lo,'rank_high':min(hi,len(order)),**counts(choice,fid,expanded,seen,adj)})
    cumulative=[counts(order[:k],fid,expanded,seen,adj) for k in KS if k<=len(order)]
    diverse=select_diverse(z,scores,128,np.random.default_rng(seed+98000)) if run_dynamic else None
    diversity=[counts(diverse[:k],fid,expanded,seen,adj) for k in KS if k<=128] if run_dynamic else []
    live=[]
    if run_dynamic:
        for replicate in range(3):
            live.append({'replicate':replicate,'rows':live_sequence(t,obs,pos,vo,scores,z,v,seen.copy(),expanded,points,index,128,seed+99000+prefix+replicate*100)})
    result={'condition':condition,'seed':seed,'prefix':prefix,'checkpoint':str(checkpoint),'frontier_entries':len(obs),
        'seen_states':int(seen.sum()),'whole_queue':counts(order,fid,expanded,seen,adj),
        'rank_bins':bins,'top_k':cumulative,'diverse_batch':diversity,'live_sequential':live}
    path=out/f'{condition}-seed{seed}-block{block}.json';path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'condition':condition,'seed':seed,'prefix':prefix,'top100':counts(order[:100],fid,expanded,seen,adj)}),flush=True)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);seed_all(0,1)
    (a.out/'runner.py').write_text(Path(__file__).read_text())
    rows=[]
    for arm in ['epoch1000','interleaved100']:
        for seed in range(3):
            source=Path(f'runs/persistent-frontier-20261110/{arm}-seed{seed}')
            for block in [1,5,10,20]:
                live=arm=='interleaved100' and (block==10 or (block==20 and seed==2))
                rows.append(audit(source,source/f'checkpoint-{block}.pt',arm,seed,block,a.out,live))
    for seed,block in [(0,10),(1,10),(2,10),(2,20)]:
        source=Path(f'runs/persistent-frontier-20261110/interleaved100-seed{seed}')
        checkpoint=Path(f'runs/frontier-fitting-branches-20261110/seed{seed}-block{block}-extra2000/fitted.pt')
        rows.append(audit(source,checkpoint,'fitted2000',seed,block,a.out,True))
    (a.out/'results.json').write_text(json.dumps(rows,indent=2)+'\n')

if __name__=='__main__':main()
