"""Action structure and latent loop detectability on fixed models; evaluation only."""
from pathlib import Path
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from map_learning.training import Trainer,seed_all
from map_learning.maze import Maze,reachable_graph
from map_learning.probabilistic import mixture_mean
from analyze_exploration_uncertainty import auc

@torch.no_grad()
def main():
 seed_all(0,1);rows=[];hist=[]
 for group in ['h16_shared_30k','h8_independent_30k']:
  for seed in range(3):
   run=Path(f'runs/horizon-duration-20261110/{group}-seed{seed}');t=Trainer.load(run/'checkpoint.pt');env=Maze(t.config.maze)
   points,adj,index=reachable_graph(env);z=t.model.encode(t.tensor(np.stack([env.observe(p) for p in points])))
   source=np.repeat(np.arange(len(points)),4);act=np.tile(np.arange(4),len(points));target=adj[:,:4].ravel();blocked=source==target
   pred=mixture_mean(t.dynamics.distribution(z[source],torch.as_tensor(act))).mean(0)
   stepdist=(z[target]-z[source]).norm(dim=-1).numpy();spacing=float(np.median(stepdist[~blocked]));motion=(pred-z[source]).norm(dim=-1).numpy()/spacing
   saved=dict(np.load(run/'replay.npz'));visited=np.unique([index[tuple(p)] for p in np.concatenate([saved['positions'],saved['next_positions']])])
   rng=np.random.default_rng(6631);real=rng.choice(visited,256);imagined=z[real].clone();previous=rng.integers(5,size=256)
   rh=[real.copy()];zh=[imagined.clone()];ds=[];labels=[];moving=[];depths=[]
   for depth in range(1,33):
    action=np.where(rng.random(256)<.8,previous,rng.integers(5,size=256));previous=action
    nxt=adj[real,action];predicted=mixture_mean(t.dynamics.distribution(imagined,torch.as_tensor(action))).mean(0)
    distance=(predicted[:,None]-torch.stack(zh,dim=1)).norm(dim=-1).min(-1).values.numpy()/spacing
    ds.extend(distance);labels.extend((nxt[:,None]==np.stack(rh,axis=1)).any(1));moving.extend(nxt!=real);depths.extend([depth]*256)
    real=nxt;imagined=predicted;rh.append(real.copy());zh.append(imagined.clone())
   ds=np.array(ds);labels=np.array(labels);moving=np.array(moving);depths=np.array(depths)
   row={'group':group,'seed':seed,'moving_actions':int((~blocked).sum()),'blocked_actions':int(blocked.sum()),
        'median_predicted_moving_displacement_edge_units':float(np.median(motion[~blocked])),
        'median_predicted_blocked_displacement_edge_units':float(np.median(motion[blocked])),
        'blocked_vs_moving_auc':auc(-motion,blocked),'loop':{}}
   for h in [8,16,32]:
    mask=moving&(depths<=h);novel=mask&~labels;threshold=float(np.quantile(ds[novel],.05))
    row['loop'][str(h)]={'moving_steps':int(mask.sum()),'revisit_fraction':float(labels[mask].mean()),
      'auc':auc(-ds,labels,mask),'threshold_at_5pct_false_positive':threshold,
      'true_positive_at_threshold':float((ds[mask&labels]<=threshold).mean())}
   # Remove low-variance input directions without retraining the dynamics.
   centered=z-z.mean(0);_,_,vh=torch.linalg.svd(centered,full_matrices=False)
   row['pca_input_ablation']={}
   for k in [1,2,3,4,8]:
    compressed=(centered@vh[:k].T)@vh[:k]+z.mean(0)
    guesses=[]
    for start in range(0,len(source),256):
     sl=slice(start,start+256)
     predicted=mixture_mean(t.dynamics.distribution(compressed[source[sl]],torch.as_tensor(act[sl]))).mean(0)
     guesses.extend(torch.cdist(predicted,z,compute_mode='donot_use_mm_for_euclid_dist').argmin(-1).numpy())
    correct=np.asarray(guesses)==target
    row['pca_input_ablation'][str(k)]={'all_accuracy':float(correct.mean()),'blocked_accuracy':float(correct[blocked].mean()),'moving_accuracy':float(correct[~blocked].mean())}
   rows.append(row);hist.append((ds,labels,moving))
   if group=='h16_shared_30k' and seed==0:
    zz=z.numpy();_,_,v=np.linalg.svd(zz-zz.mean(0),full_matrices=False);pc=(zz-zz.mean(0))@v[:2].T
    action0=np.arange(0,len(source),4);sel=action0[::6];src=source[sel];true=(zz[target[sel]]-zz[src])@v[:2].T;estimated=(pred.numpy()[sel]-zz[src])@v[:2].T
    fig,ax=plt.subplots(figsize=(8,7));ax.scatter(*pc.T,s=3,c='lightgray');ax.scatter(*pc[src].T,s=9,c=np.where(blocked[sel],'red','blue'))
    ax.quiver(*pc[src].T,*true.T,color='#148740',angles='xy',scale_units='xy',scale=1,width=.002,label='Actual latent displacement')
    ax.quiver(*pc[src].T,*estimated.T,color='#e59324',angles='xy',scale_units='xy',scale=1,width=.002,label='Predicted displacement')
    ax.set_aspect('equal');ax.set_title('Action RIGHT in latent PCA • H16 shared seed 0\nRed: blocked in real maze; blue: movement allowed');ax.legend(loc='lower left');fig.tight_layout();fig.savefig('docs/assets/latent-action-vectors.png',dpi=160);plt.close(fig)
 Path('docs/assets/geometry-dynamics-results.json').write_text(json.dumps(rows,indent=2)+'\n')
 print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
