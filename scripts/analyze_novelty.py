"""Evaluation-only novelty labels, prediction-error alignment and path selection.

Novelty is defined from the replay prefix at each checkpoint. Oracle geometry
provides labels only: no new transitions are collected or fed to training.
"""
import argparse
from collections import deque
import json
from pathlib import Path
import numpy as np
import torch
from map_learning.training import Trainer,seed_all
from map_learning.maze import Maze,reachable_graph
from map_learning.probabilistic import mixture_mean
from analyze_exploration_uncertainty import auc
from analyze_reward_stability import rank


def correlation(a,b):
    if len(a)<2 or np.ptp(a)==0 or np.ptp(b)==0:return None
    return float(np.corrcoef(rank(a),rank(b))[0,1])


def frontier_distances(adj,visited):
    reverse=[[] for _ in visited]
    for i,targets in enumerate(adj):
        for j in targets:reverse[j].append(i)
    dist=np.full(len(visited),-1,dtype=int);q=deque(np.flatnonzero(~visited).tolist());dist[~visited]=0
    while q:
        j=q.popleft()
        for i in reverse[j]:
            if dist[i]<0:dist[i]=dist[j]+1;q.append(i)
    return dist


def top_choice(scores,outcome):
    """Expected outcome under uniform tie breaking, one value per row."""
    best=scores==scores.max(-1,keepdims=True)
    return (best*outcome).sum(-1)/best.sum(-1)


def edge_metrics(adj,visited,seen,disagreement,error,physical):
    source=np.repeat(np.arange(len(visited)),5);target=adj.ravel()
    v=visited[source];tried=seen.ravel();novel=~visited[target]
    dist=frontier_distances(adj,visited)
    groups={'all':np.ones(len(source),bool),'visited_source':v,'tried':tried,
            'untried_at_visited':v&~tried,'unvisited_source':~v}
    scores={'disagreement':disagreement,'latent_error':error,'physical_error':physical}
    result={'visited_states':int(visited.sum()),'tried_edges':int(tried.sum()),'groups':{},'direction':{}}
    for name,mask in groups.items():
        result['groups'][name]={'n':int(mask.sum()),'novel_successors':int((mask&novel).sum()),
            'physical_mistake_rate_novel':float((physical[mask&novel]>0).mean()) if (mask&novel).any() else None,
            'physical_mistake_rate_non_novel':float((physical[mask&~novel]>0).mean()) if (mask&~novel).any() else None,
            'new_successor_rate':float(novel[mask].mean()) if mask.any() else None,
            'disagreement_error_spearman':correlation(disagreement[mask],error[mask]),
            'disagreement_physical_spearman':correlation(disagreement[mask],physical[mask]),
            'new_successor_auc':{k:auc(values,novel,mask) for k,values in scores.items()},
            'mean_scores':{k:float(values[mask].mean()) if mask.any() else None for k,values in scores.items()}}
    result['unvisited_source_auc']={k:auc(values,~v) for k,values in scores.items()}
    result['untried_edge_auc_at_visited']={k:auc(values,~tried,v) for k,values in scores.items()}
    toward=(dist[adj]>=0)&(dist[adj]<dist[:,None])
    for label,lo,hi in [('frontier',1,1),('near',2,4),('middle',5,16),('far',17,100000)]:
        mask=visited&(dist>=lo)&(dist<=hi)
        result['direction'][label]={'n':int(mask.sum()),'random_toward_rate':float(toward[mask].mean()) if mask.any() else None,
            **{k:float(top_choice(values.reshape(-1,5)[mask],toward[mask]).mean()) if mask.any() else None for k,values in scores.items()}}
    return result


@torch.no_grad()
def evaluate(run,checkpoint,cutoff,path_audit=False):
    trainer=Trainer.load(checkpoint);cfg=trainer.config
    env=Maze(cfg.maze);points,adj,index=reachable_graph(env)
    with np.load(run/'replay.npz') as saved:replay={k:saved[k].copy() for k in ['positions','next_positions','actions']}
    source=np.array([index[tuple(p)] for p in replay['positions'][:cutoff]])
    successors=np.array([index[tuple(p)] for p in replay['next_positions'][:cutoff]])
    visited=np.zeros(len(points),bool);visited[source]=True;visited[successors]=True
    seen=np.zeros((len(points),5),bool);seen[source,replay['actions'][:cutoff]]=True
    z=trainer.model.encode(trainer.tensor(np.stack([env.observe(p) for p in points])))
    variance=max(float(z.var(0,unbiased=False).mean()),1e-12)
    src=np.repeat(np.arange(len(points)),5);acts=np.tile(np.arange(5),len(points));target=adj.ravel()
    dis=[];err=[];physical=[]
    for start in range(0,len(src),256):
        sl=slice(start,start+256)
        means=mixture_mean(trainer.dynamics.distribution(z[src[sl]],torch.as_tensor(acts[sl])))
        pred=means.mean(0)
        dis.extend((means.var(0,unbiased=False).mean(-1)/variance).numpy())
        err.extend(((pred-z[target[sl]]).square().mean(-1)/variance).numpy())
        decoded=torch.cdist(pred,z,compute_mode='donot_use_mm_for_euclid_dist').argmin(-1).numpy()
        physical.extend(np.linalg.norm(points[decoded]-points[target[sl]],axis=-1)/cfg.maze.step_size)
    dis,err,physical=map(np.asarray,(dis,err,physical))
    result={'run':str(run),'checkpoint':checkpoint.name,'cutoff':cutoff,
            'edges':edge_metrics(adj,visited,seen,dis,err,physical)}
    if path_audit:
        result['paths']=paths(trainer,z,adj,visited,dis.reshape(-1,5),err.reshape(-1,5),variance)
    return result


@torch.no_grad()
def paths(trainer,z,adj,visited,dis,error,variance,starts=64,candidates=128,horizon=64):
    rng=np.random.default_rng(83721)
    initial=rng.choice(np.flatnonzero(visited),starts,replace=False)
    frontier=frontier_distances(adj,visited)[initial]
    results={};discount=.99
    for family in ['iid','persistent']:
        real=np.repeat(initial,candidates);latent=z[real].clone();previous=rng.integers(5,size=len(real))
        visited_on_path=np.zeros((len(real),len(visited)),bool)
        counts=np.zeros(len(real));scores={k:np.zeros(len(real)) for k in ['real_disagreement','real_error','imagined_disagreement']}
        ii=np.arange(len(real));rows=[]
        for depth in range(1,horizon+1):
            new=rng.integers(5,size=len(real))
            action=new if family=='iid' else np.where(rng.random(len(real))<.8,previous,new)
            previous=action
            means=mixture_mean(trainer.dynamics.distribution(latent,torch.as_tensor(action)))
            weight=discount**(depth-1)
            scores['imagined_disagreement']+=weight*means.var(0,unbiased=False).mean(-1).numpy()/variance
            scores['real_disagreement']+=weight*dis[real,action]
            scores['real_error']+=weight*error[real,action]
            real=adj[real,action];latent=means.mean(0)
            counts+=(~visited[real])&(~visited_on_path[ii,real]);visited_on_path[ii,real]=True
            if depth not in [4,8,16,32,64]:continue
            novelty=counts.reshape(starts,candidates);eligible=novelty.max(1)>0
            row={'horizon':depth,'starts':starts,'starts_with_novel_candidate':int(eligible.sum()),
                 'starts_with_reachable_novelty':int(((frontier>0)&(frontier<=depth)).sum()),
                 'random_unique_new_states':float(novelty.mean()),'best_unique_new_states':float(novelty.max(1).mean()),'scores':{}}
            for key,score in scores.items():
                values=score.reshape(starts,candidates)
                correlations=[correlation(a,b) for a,b in zip(values,novelty)]
                correlations=[v for v in correlations if v is not None]
                selected=top_choice(values,novelty)
                row['scores'][key]={'selected_unique_new_states':float(selected.mean()),
                    'selected_over_best_eligible':float(np.mean(selected[eligible]/novelty.max(1)[eligible])) if eligible.any() else None,
                    'zero_novelty_choice_rate_eligible':float(top_choice(values[eligible],novelty[eligible]==0).mean()) if eligible.any() else None,
                    'within_start_spearman':float(np.mean(correlations)) if correlations else None,
                    'spearman_starts':len(correlations)}
            rows.append(row)
        results[family]=rows
    return {'starts':starts,'candidates':candidates,'discount':discount,'start_frontier_distances':frontier.tolist(),'families':results}


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    seed_all(0,1);args.out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads(args.manifest.read_text())
    for group in manifest['groups']:
        for run_text in group['runs']:
            run=Path(run_text);cfg=json.loads((run/'config.json').read_text());total=cfg['collect_steps']
            for cutoff in group['cutoffs']:
                checkpoint=run/('checkpoint.pt' if cutoff==total else f'checkpoint-{cutoff}.pt')
                out=args.out/f"{group['name']}-seed{cfg['seed']}-step{cutoff}.json"
                if out.exists():continue
                row=evaluate(run,checkpoint,cutoff,path_audit=cutoff==total)
                row.update(group=group['name'],seed=cfg['seed'])
                out.write_text(json.dumps(row,indent=2,allow_nan=False)+'\n');print(out,flush=True)

if __name__=='__main__':main()
