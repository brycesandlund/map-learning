"""Post-hoc geometry of learned coordinate encoders; simulator labels never train the agent."""
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from map_learning.training import Trainer, seed_all
from map_learning.maze import Maze, reachable_graph, bfs
from analyze_reward_stability import rank


def rho(x,y):return float(np.corrcoef(rank(x),rank(y))[0,1])


def metrics(z,points,adj,dist,visited):
    centered=z-z.mean(0);u,s,v=np.linalg.svd(centered,full_matrices=False)
    proportions=s*s/(s*s).sum();projection=centered@v[:2].T
    dz=np.linalg.norm(z[:,None]-z[None,:],axis=-1)
    dp=np.linalg.norm(points[:,None]-points[None,:],axis=-1)
    i,j=np.triu_indices(len(z),1);near=np.isclose(dp[i,j],.5)
    barrier=near&(dist[i,j]>=8);openpair=near&(dist[i,j]==2)
    knn=np.argsort(dz,axis=1)[:,1:11];knnp=np.argsort(dp,axis=1)[:,1:11]
    rng=np.random.default_rng(7261);known=rng.permutation(np.flatnonzero(visited));split=int(.8*len(known));fit=known[:split]
    design=np.column_stack([z,np.ones(len(z))]);weights=np.linalg.lstsq(design[fit],points[fit],rcond=None)[0]
    pred=design@weights
    probe={name:float(np.sqrt(np.mean((pred[idx]-points[idx])**2))/.25) if len(idx) else None for name,idx in [('heldout_visited',known[split:]),('unvisited',np.flatnonzero(~visited))]}
    result={'pca_variance':proportions.tolist(),'pca2_variance':float(proportions[:2].sum()),
      'participation_dimension':float(1/(proportions**2).sum()),'latent_physical_spearman':rho(dz[i,j],dp[i,j]),
      'latent_geodesic_spearman':rho(dz[i,j],dist[i,j]),'physical_geodesic_spearman':rho(dp[i,j],dist[i,j]),
      'barrier_pairs':int(barrier.sum()),'open_pairs':int(openpair.sum()),
      'barrier_vs_open_median_distance_ratio':float(np.median(dz[i[barrier],j[barrier]])/np.median(dz[i[openpair],j[openpair]])),
      'knn10_mean_graph_distance':float(dist[np.arange(len(z))[:,None],knn].mean()),
      'physical_knn10_mean_graph_distance':float(dist[np.arange(len(z))[:,None],knnp].mean()),
      'linear_coordinate_probe_rmse_grid_steps':probe,
      'visited':int(visited.sum()),'minimum_distinct_distance':float(dz[i,j].min()),
      'median_adjacent_distance':float(np.median(dz[np.arange(len(z))[:,None],adj][adj!=np.arange(len(z))[:,None]]))}
    return result,projection,proportions,(dz[i,j],dp[i,j],dist[i,j]),(i,j,barrier,openpair)


def main():
    seed_all(0,1);root=Path('runs/latent-geometry-20261110');root.mkdir(exist_ok=False)
    run=Path('runs/horizon-duration-20261110/h16_shared_30k-seed0');t=Trainer.load(run/'checkpoint.pt')
    env=Maze(t.config.maze);points,adj,index=reachable_graph(env);dist=np.stack([bfs(adj,i) for i in range(len(points))])
    obs=np.stack([env.observe(p) for p in points]);rows=[];pictures={}
    for group in ['h16_shared_30k','h8_independent_30k']:
      for seed in range(3):
        run=Path(f'runs/horizon-duration-20261110/{group}-seed{seed}');saved=dict(np.load(run/'replay.npz'))
        for cutoff in [0,6000,18000,30000]:
          t=Trainer.load(run/('checkpoint.pt' if cutoff in [0,30000] else f'checkpoint-{cutoff}.pt'))
          if cutoff==0:seed_all(seed,1);t=Trainer(t.config,tuple(obs.shape[1:]))
          mask=np.zeros(len(points),bool)
          for p in np.concatenate([saved['positions'][:max(cutoff,30000 if cutoff==0 else 0)],saved['next_positions'][:max(cutoff,30000 if cutoff==0 else 0)]]):mask[index[tuple(p)]]=True
          with torch.no_grad():z=t.model.encode(t.tensor(obs)).numpy()
          result,pc,prop,pairs,barriers=metrics(z,points,adj,dist,mask)
          result.update(group=group,seed=seed,cutoff=cutoff);rows.append(result)
          np.savez_compressed(root/f'{group}-seed{seed}-step{cutoff}.npz',points=points,latent=z,visited=mask,projection=pc)
          if seed==0 and group=='h16_shared_30k':pictures[cutoff]=(pc,prop,pairs,barriers,z,mask)
          print(group,seed,cutoff,result['pca2_variance'],result['latent_physical_spearman'],result['barrier_vs_open_median_distance_ratio'],flush=True)
    (root/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
    Path('docs/assets/latent-geometry-results.json').write_text(json.dumps(rows,indent=2)+'\n')
    color=np.column_stack([points[:,0]/12,points[:,1]/12,np.full(len(points),.55)])
    fig,axes=plt.subplots(2,3,figsize=(15,9));env.draw(axes[0,0]);axes[0,0].scatter(*points.T,c=color,s=9);axes[0,0].set_title('Physical maze • color encodes x and y')
    source=np.repeat(np.arange(len(points)),5);target=adj.ravel();edges=np.column_stack([source[source<target],target[source<target]])
    for ax,cutoff in zip([axes[0,1],axes[0,2],axes[1,0]],[0,6000,30000]):
      pc,prop,_,_,_,mask=pictures[cutoff]
      ax.add_collection(LineCollection(pc[edges],colors='gray',alpha=.22,linewidths=.35))
      ax.scatter(*pc.T,c=color,s=9)
      if cutoff:ax.scatter(*pc[~mask].T,s=20,facecolors='none',edgecolors='black',linewidths=.3)
      ax.set_aspect('equal');ax.set_title(f'{"Random initialization" if cutoff==0 else str(cutoff)+" actions"}\nPC1+PC2: {prop[:2].sum():.2%} variance');ax.set_xlabel('PC1');ax.set_ylabel('PC2')
    for cutoff in [0,6000,18000,30000]:axes[1,1].plot(np.arange(1,9),pictures[cutoff][1],'o-',label=f'{cutoff:,} actions')
    axes[1,1].set(yscale='log',xlabel='Principal component',ylabel='Fraction of variance',title='Global latent spectrum');axes[1,1].legend()
    pc,prop,(dz,dp,dg),barriers,_,_=pictures[30000];i,j,barrier,openpair=barriers
    base=np.median(dz[openpair]);axes[1,2].hist(dz[openpair]/base,bins=40,density=True,alpha=.6,label='Open route: 2 moves')
    axes[1,2].hist(dz[barrier]/base,bins=40,density=True,alpha=.6,label='Wall detour: ≥8 moves')
    axes[1,2].set(xlabel='Latent distance / median open-pair distance',ylabel='Density',title='Same physical separation (0.5 units)');axes[1,2].legend()
    fig.suptitle('H16 shared, seed 0 • true adjacency edges • rings mark unvisited states',fontsize=14)
    fig.tight_layout();fig.savefig('docs/assets/latent-geometry.png',dpi=160);plt.close(fig)

if __name__=='__main__':main()
