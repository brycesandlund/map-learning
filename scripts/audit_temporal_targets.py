"""Measure observed-ancestry lag versus oracle geodesics, evaluation only."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.stats import spearmanr
from map_learning.training import Trainer,seed_all
from map_learning.maze import Maze,reachable_graph


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args();seed_all(0,1)
    rows=[]
    for checkpoint in sorted(a.root.glob('temporal-*/checkpoint.pt')):
        t=Trainer.load(checkpoint);s=t.distance_spec;tree=t.ancestry
        points,adj,index=reachable_graph(Maze(t.config.maze));n=len(points)
        src=np.repeat(np.arange(n),4);dst=adj[:,:4].flatten();moving=src!=dst
        geo=shortest_path(csr_matrix((np.ones(moving.sum()),(src[moving],dst[moving])),shape=(n,n)),directed=False,unweighted=True)
        rng=np.random.default_rng(99011);indices=rng.integers(tree.size-1,size=20000)
        x,y,k=tree.pairs(indices,rng,s.horizon)
        ids=lambda obs:np.array([index[tuple(np.round((o[:2]+1)*t.config.maze.size/2,5))] for o in obs])
        g=geo[ids(x),ids(y)];positive=k>0
        target=np.minimum(k/s.horizon,1);ideal=np.minimum(g/s.horizon,1)
        rows.append({'run':str(checkpoint.parent),'horizon':s.horizon,'samples':len(k),'positive_lag_samples':int(positive.sum()),
                     'positive_lag_identical_physical_fraction':float((g[positive]==0).mean()),
                     'mean_lag':float(k.mean()),'mean_shortest_distance':float(g.mean()),
                     'median_path_excess_steps':float(np.median(k-g)),
                     'target_mean_excess_over_shortest':float((target-ideal).mean()),
                     'target_excess_over_quarter_fraction':float((target-ideal>.25).mean()),
                     'lag_geodesic_spearman':float(spearmanr(k,g).statistic)})
    a.out.write_text(json.dumps(rows,indent=2)+'\n')
    for r in rows:print(json.dumps(r))

if __name__=='__main__':main()
