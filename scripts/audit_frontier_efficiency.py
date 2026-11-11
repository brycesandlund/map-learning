"""Privileged evaluation-only diagnostics of persistent noisy frontier efficiency."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree
from map_learning.training import Trainer, seed_all
from map_learning.maze import Maze, reachable_graph


def reconstruct(data, prefix):
    """Recover archive membership from logged entry observations, not physical IDs."""
    nxt=data['next_observations'][:prefix]
    lookup={o.tobytes():i for i,o in enumerate(nxt)}
    assert len(lookup)==len(nxt), 'This study has unique nuisance draws'
    active=np.ones(prefix,dtype=bool)
    for obs in data['observations'][5:prefix:5]:
        idx=lookup[obs.tobytes()]
        assert active[idx]
        active[idx]=False
    return nxt[active],data['next_positions'][:prefix][active],data['observations'][:prefix:5],data['positions'][:prefix:5]


@torch.no_grad()
def encode(t,x):
    return np.concatenate([t.model.encode(t.tensor(x[i:i+1024])).numpy() for i in range(0,len(x),1024)])


def priority_stats(scores, physical_ids, productive, expanded):
    good=productive[physical_ids]
    repeat=expanded[physical_ids]
    order=np.argsort(-scores,kind='stable')
    best_good=float(scores[good].max()) if good.any() else None
    rank=int((scores>best_good).sum()+1) if good.any() else None
    return {'productive_entries':int(good.sum()),'productive_positions':int(len(np.unique(physical_ids[good]))),
        'best_productive_score':best_good,'best_productive_rank':rank,'maximum_score':float(scores.max()),
        'top100_repeat_fraction':float(repeat[order[:100]].mean()),
        'top100_productive_fraction':float(good[order[:100]].mean()),
        'top100_distinct_positions':int(len(np.unique(physical_ids[order[:100]]))),
        'repeat_score_quantiles':np.quantile(scores[repeat],[.5,.9,.99,1]).tolist() if repeat.any() else [],
        'productive_score_quantiles':np.quantile(scores[good],[0,.5,1]).tolist() if good.any() else []}


def audit_run(path):
    data=np.load(path/'replay.npz');result=json.loads((path/'result.json').read_text())
    t=Trainer.load(path/'checkpoint.pt');env=Maze(t.config.maze);points,adj,index=reachable_graph(env)
    ids=lambda x:np.array([index[tuple(np.round(p,5))] for p in x])
    sources=ids(data['positions'][::5]);targets=ids(data['next_positions']).reshape(-1,5)
    seen=np.zeros(len(points),bool);seen[0]=True;expanded=np.zeros(len(points),bool)
    blocks=[];repeats=0;first_full=result['first_full'];end=(first_full+4)//5 if first_full else len(sources)
    for start in range(0,len(sources),1000):
        rep=0;new=0
        for i in range(start,min(start+1000,len(sources))):
            was=expanded[sources[i]];expanded[sources[i]]=True
            discovery=int(np.count_nonzero(~seen[np.unique(targets[i])]))
            assert not (was and discovery), 'Repeated expansion in deterministic all-action environment cannot discover new states'
            rep+=int(was);new+=discovery;seen[targets[i]]=True
            if i<end:repeats+=int(was)
        blocks.append({'block':start//1000+1,'repeat_expansions':rep,'new_states':new,'visited':int(seen.sum()),
            'unique_expanded_total':int(expanded.sum())})
    checkpoints=[]
    for block in [1,5,10,20]:
        prefix=block*5000;t=Trainer.load(path/f'checkpoint-{block}.pt')
        obs,pos,vo,vp=reconstruct(data,prefix)
        if block==20:
            with np.load(path/'frontier.npz') as f:
                np.testing.assert_array_equal(obs,f['observations']);np.testing.assert_array_equal(vo,f['visited_observations'])
        seen=np.zeros(len(points),bool);seen[0]=True;seen[ids(data['next_positions'][:prefix])]=True
        expanded=np.zeros(len(points),bool);expanded[ids(vp)]=True
        productive=(~seen[adj]).any(1);fid=ids(pos)
        z=encode(t,obs);v=encode(t,vo)
        scores,near=cKDTree(v).query(z,eps=0)
        base=priority_stats(scores,fid,productive,expanded)
        base['nearest_anchor_same_physical_fraction']=float((ids(vp)[near]==fid).mean())
        zero=obs.copy();zero[:,2:]=0;vzero=vo.copy();vzero[:,2:]=0
        zs=cKDTree(encode(t,vzero)).query(encode(t,zero),eps=0)[0]
        altered=priority_stats(zs,fid,productive,expanded)
        checks={'block':block,'prefix':prefix,'seen_states':int(seen.sum()),'raw':base,'zero_nuisance_counterfactual':altered}
        checkpoints.append(checks)
    return {'arm':result['arm'],'seed':result['seed'],'expansions_until_completion_or_budget':end,
        'repeat_expansions_until_completion_or_budget':repeats,'repeat_actions_until_completion_or_budget':repeats*5,
        'repeat_fraction_until_completion_or_budget':repeats/end,'blocks':blocks,'checkpoints':checkpoints}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('runs/persistent-frontier-20261110'))
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();seed_all(0,1)
    a.out.mkdir(parents=True,exist_ok=False)
    rows=[]
    for arm in ['epoch1000','interleaved100']:
        for seed in range(3):
            row=audit_run(a.root/f'{arm}-seed{seed}');rows.append(row)
            (a.out/f'{arm}-seed{seed}.json').write_text(json.dumps(row,indent=2)+'\n')
            print(json.dumps({'arm':arm,'seed':seed,'repeat_fraction':row['repeat_fraction_until_completion_or_budget'],
                'final_ranking':row['checkpoints'][-1]}),flush=True)
    (a.out/'results.json').write_text(json.dumps(rows,indent=2)+'\n')

if __name__=='__main__':main()
