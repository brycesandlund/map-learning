"""Final-model diagnostic: compare archive rewards on actual vs predicted successors."""
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from map_learning.training import Trainer,seed_all
from map_learning.maze import Maze,reachable_graph
from map_learning.probabilistic import mixture_mean

@torch.no_grad()
def main():
 seed_all(0,1);run=Path('runs/novelty-rewards-20261110/archive_path-seed0');t=Trainer.load(run/'checkpoint.pt');env=Maze(t.config.maze);points,adj,index=reachable_graph(env)
 data=dict(np.load(run/'replay.npz'));visited=np.zeros(len(points),bool)
 for p in np.concatenate([data['positions'],data['next_positions']]):visited[index[tuple(p)]]=True
 z=t.model.encode(t.tensor(np.stack([env.observe(p) for p in points])));archive=z[visited]
 nearest=torch.cdist(z,archive,compute_mode='donot_use_mm_for_euclid_dist').min(-1).values.numpy()
 true=nearest[adj].max(-1);pred=[]
 for start in range(0,len(points),128):
  state=z[start:start+128];mean=mixture_mean(t.dynamics.distribution(state.repeat_interleave(5,0),torch.arange(5).repeat(len(state)))).mean(0)
  pred.extend(torch.cdist(mean,archive,compute_mode='donot_use_mm_for_euclid_dist').min(-1).values.reshape(-1,5).max(-1).values.numpy())
 pred=np.array(pred);scale=max(float(np.quantile(np.r_[true[visited],pred[visited]],.99)),1e-8);norm=SymLogNorm(scale/1000,vmin=0,vmax=scale)
 fig,axes=plt.subplots(1,3,figsize=(14,5))
 for ax in axes:env.draw(ax)
 axes[0].scatter(*points[visited].T,s=6,c='#a0bfce');x=data['positions'][-200:];axes[0].plot(*x.T,color='#e87524',lw=1);axes[0].scatter(*data['next_positions'][-1],c='red',s=25)
 axes[0].set_title('Final visits and last 200 real actions')
 for ax,values,title in zip(axes[1:],[true,pred],['Actual successor: reward only at frontier','Predicted successor: interior reward appears']):
  art=ax.scatter(*points[visited].T,c=values[visited],s=12,marker='s',cmap='magma',norm=norm);ax.set_title(title,fontsize=10)
 fig.colorbar(art,ax=axes[1:].tolist(),fraction=.02,pad=.02,label='Max action distance to observed archive (latent L2)')
 fig.suptitle('Archive/path H16, seed 0 • final checkpoint • visited source states only')
 fig.subplots_adjust(top=.84,bottom=.16,wspace=.25,right=.87)
 fig.text(.5,.035,'Same encoder, archive, and color scale. White regions are unvisited source states. This is a post-hoc one-step diagnostic.',ha='center',fontsize=9)
 fig.savefig('docs/assets/archive-reward-diagnostic.png',dpi=160);plt.close(fig)
if __name__=='__main__':main()
