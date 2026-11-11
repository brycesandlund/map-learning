"""Evaluation-only full-dimensional geometry and common-frontier ranking audit."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.stats import spearmanr, rankdata
from map_learning.training import Trainer, seed_all
from map_learning.maze import Maze, reachable_graph
from audit_frontier_efficiency import reconstruct, encode, priority_stats
from audit_queue_density import live_sequence


def auc(good,bad):
    if not len(good) or not len(bad):return None
    ranks=rankdata(np.r_[good,bad]);n,m=len(good),len(bad)
    return float((ranks[:n].sum()-n*(n+1)/2)/(n*m))


def analyze(checkpoint,source,prefix,out,live=False):
    t=Trainer.load(checkpoint);env=Maze(t.config.maze)
    if t.config.maze.observation != 'coordinates':raise ValueError('This audit uses coordinate+nuisance observations')
    points,adj,index=reachable_graph(env);n=len(points)
    ids=lambda x:np.array([index[tuple(np.round(p,5))] for p in x])
    with np.load(source/'replay.npz') as data:
        qo,qp,vo,vp=reconstruct(data,prefix)
        seen=np.zeros(n,bool);seen[ids(data['positions'][:1])]=True;seen[ids(data['next_positions'][:prefix])]=True
    expanded=np.zeros(n,bool);expanded[ids(vp)]=True;productive=(~seen[adj]).any(1);qid=ids(qp)
    qz,vz=encode(t,qo).astype(float),encode(t,vo).astype(float)
    scores,near=cKDTree(vz).query(qz,eps=0)
    stats=priority_stats(scores,qid,productive,expanded)
    stats['productive_vs_repeat_auroc']=auc(scores[productive[qid]],scores[expanded[qid]])
    stats['nearest_anchor_same_physical_fraction']=float((ids(vp)[near]==qid).mean())
    if live:
        stats['frozen_continuations']=[live_sequence(t,qo,qp,vo,scores,qz,vz,seen.copy(),expanded,points,index,128,
                                                     700100+prefix+i*100) for i in range(3)]
    # Probe encoder on the same independent noise bank at every physical point.
    # This is evaluation only; labels never enter the trainer or queue.
    rng=np.random.default_rng(77219);repeats=32
    bank=np.stack([env.observe(points[0]) for _ in range(repeats)])
    # Override noise RNG for cross-model reproducibility, leaving normalized position intact.
    bank[:,2:]=rng.normal(0,1.,(repeats,bank.shape[1]-2))
    obs=np.tile(bank[None],(n,1,1));obs[:,:,:2]=(2*points/t.config.maze.size-1)[:,None]
    z=encode(t,obs.reshape(-1,obs.shape[-1])).reshape(n,repeats,-1).astype(float)
    centroids=z.mean(1);center=centroids.mean(0);residual=z-centroids[:,None]
    _,singular,axes=np.linalg.svd(centroids-center,full_matrices=False)
    spectrum=singular**2/max(np.sum(singular**2),1e-30)
    scale=float(np.sqrt(np.mean(np.sum((centroids-center)**2,axis=-1))))
    for axis in axes:
        if axis[np.argmax(abs(axis))]<0:axis*=-1
    pc=(centroids-center)@axes[:3].T/max(scale,1e-15)
    src=np.repeat(np.arange(n),4);dst=adj[:,:4].flatten();moving=src!=dst
    edge=np.linalg.norm(centroids[src[moving]]-centroids[dst[moving]],axis=-1)
    pair_noise=np.linalg.norm(z[:,:16]-z[:,16:],axis=-1)
    valid=(adj[:,0]!=np.arange(n))&(adj[:,2]!=np.arange(n))
    jac=np.stack([centroids[adj[valid,0]]-centroids[valid],centroids[adj[valid,2]]-centroids[valid]],axis=-1)
    u,stretch,_=np.linalg.svd(jac,full_matrices=False)
    anisotropy=stretch[:,0]/np.maximum(stretch[:,1],1e-15)
    weak_noise=np.sqrt(np.mean(np.sum(residual[valid]*u[:,:,1][:,None],axis=-1)**2,axis=1))
    weak_ratio=weak_noise/np.maximum(stretch[:,1],1e-15)
    graph=csr_matrix((np.ones(moving.sum()),(src[moving],dst[moving])),shape=(n,n))
    geo=shortest_path(graph,directed=False,unweighted=True)
    pair=rng.integers(n,size=(20000,2));nonident=pair[:,0]!=pair[:,1];pair=pair[nonident]
    d=np.linalg.norm(centroids[pair[:,0]]-centroids[pair[:,1]],axis=-1)
    geodesic=geo[pair[:,0],pair[:,1]];euclidean=np.linalg.norm(points[pair[:,0]]-points[pair[:,1]],axis=-1)
    def corr(a,b):
        v=spearmanr(a,b).statistic
        return float(v) if np.isfinite(v) else None
    near_mask=geodesic<=16
    seen_pairs=seen[pair[:,0]]&seen[pair[:,1]]
    nn=cKDTree(centroids).query(centroids,k=2)[1][:,1]
    quant=lambda x:np.quantile(x,[.1,.5,.9,.99]).tolist()
    result={'checkpoint':str(checkpoint),'source':str(source),'prefix':prefix,'seen_positions':int(seen.sum()),
            'queue':stats,'pca_spectrum':spectrum.tolist(),'centroid_rms_radius':scale,
            'effective_dimension':float(1/max(np.sum(spectrum**2),1e-30)),
            'noise_to_move':float(pair_noise.mean()/max(edge.mean(),1e-15)),
            'local_anisotropy':quant(anisotropy),'weak_direction_noise_ratio':quant(weak_ratio),
            'edge_distances':quant(edge),'noise_pair_distances':quant(pair_noise),
            'geodesic_spearman':corr(d,geodesic),'euclidean_spearman':corr(d,euclidean),
            'local_geodesic_spearman':corr(d[near_mask],geodesic[near_mask]),
            'seen_geodesic_spearman':corr(d[seen_pairs],geodesic[seen_pairs]),
            'seen_euclidean_spearman':corr(d[seen_pairs],euclidean[seen_pairs]),
            'seen_local_geodesic_spearman':corr(d[seen_pairs&near_mask],geodesic[seen_pairs&near_mask]),
            'nearest_centroid_geodesic_steps':quant(geo[np.arange(n),nn])}
    out.parent.mkdir(parents=True,exist_ok=True)
    if Path(str(out)+'.json').exists():raise FileExistsError(out)
    Path(str(out)+'.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(Path(str(out)+'.npz'),points=points,adjacency=adj,centroids=centroids,seen=seen,
                        expanded=expanded,productive=productive,pc=pc,spectrum=spectrum,noise_rms=np.sqrt((residual**2).sum(-1).mean(1)),
                        local_ids=np.flatnonzero(valid),anisotropy=anisotropy,weak_ratio=weak_ratio,
                        queue_ids=qid,scores=scores,edge_src=src[moving],edge_dst=dst[moving])
    print(json.dumps(result),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--prefix',type=int,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--live',action='store_true');a=p.parse_args()
    seed_all(0,1);analyze(a.checkpoint,a.source,a.prefix,a.out,a.live)

if __name__=='__main__':main()
