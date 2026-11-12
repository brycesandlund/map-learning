"""Matched frozen-queue counterfactual, NOT a new pixel-agent performance claim."""
import argparse,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from scipy.spatial import cKDTree
from map_learning.pixel_exploration import PixelMaze,PixelLearner
from map_learning.config import MazeConfig
from map_learning.continuous import sample_controls
p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--out',type=Path,required=True);p.add_argument('--steps',type=int,default=1500);a=p.parse_args();a.out.mkdir(exist_ok=False)
s=SimpleNamespace(**json.loads((a.run/'spec.json').read_text()));assert s.dim==2 and a.steps%5==0;d=dict(np.load(a.run/'trajectory.npz'));ims=np.load(a.run/'images.npy',mmap_mode='r');torch.set_num_threads(1)
N=len(d['positions']);results=[];traces=[]
for mode in ['latent','coordinate']:
 m=PixelLearner(s);m.encoder.load_state_dict(torch.load(a.run/'checkpoint.pt',weights_only=True,map_location='cpu')['encoder'])
 pos=np.empty((N+a.steps,2),np.float32);pos[:N]=d['positions'];z=np.empty((N+a.steps,s.dim),np.float32)
 visited=list(np.unique(np.r_[0,d['parents'][s.warm_steps+1:]]));active=np.zeros(N+a.steps,bool);active[1:N]=True;active[visited]=False
 for j in range(0,N,512):z[j:min(N,j+512)]=m.encode(ims[j:min(N,j+512)].copy()) if mode=='latent' else pos[j:min(N,j+512)]*np.float32(2/s.size)-1
 ids=np.flatnonzero(active);tree=cKDTree(z[visited]);score=np.full(N+a.steps,-np.inf);score[ids]=tree.query(z[ids])[0]
 e=PixelMaze(MazeConfig(size=s.size,step_size=s.step_size,image_size=s.image_size),seed=77291,family=s.family,noise=s.noise,rgb=s.rgb,marker_size=s.marker_size,gate_width=s.gate_width)
 rng=np.random.default_rng(981721);ties=np.random.default_rng(6761);coverage=d['frames'][-1].copy();initial=coverage.sum();parents=[];actions=[];selected=[]
 for k in range(0,a.steps,5):
  ids=np.flatnonzero(active);current=int(ties.choice(ids[score[ids]==score[ids].max()]));active[current]=False;visited.append(current);selected.append(current)
  score[ids]=np.minimum(score[ids],np.linalg.norm(z[ids]-z[current],axis=1))
  controls=sample_controls(rng);controls[:4]=e.ACTIONS[:4]*rng.uniform(.5,1,(4,1)).astype(np.float32);new=[]
  for j,control in enumerate(controls):
   i=N+k+j;e.position=pos[current].copy();im,_=e.step(control);new.append(im);pos[i]=e.position;parents.append(current);actions.append(control)
   x,y=np.floor(pos[i]*4).astype(int);coverage[y,x]=True
   assert np.array_equal(e.transition(pos[current],control),pos[i])
  newids=np.arange(N+k,N+k+5);z[newids]=m.encode(np.stack(new)) if mode=='latent' else pos[newids]*np.float32(2/s.size)-1;active[newids]=True
  score[newids]=cKDTree(z[visited]).query(z[newids])[0]
 actions=np.array(actions);traces.append(actions);np.savez_compressed(a.out/(mode+'.npz'),positions=pos[N:],parents=parents,actions=actions,selected=selected,coverage=coverage)
 row=dict(mode=mode,additional_actions=a.steps,new_bins=int(coverage.sum()-initial),final_coverage=float(coverage.mean()),rare_state_88471_expanded_at=(selected.index(88471)*5+1 if 88471 in selected else None),source_actions_by_chamber=[int(((pos[np.array(parents),0]>=i*3)&(pos[np.array(parents),0]<(i+1)*3)).sum()) for i in range(4)])
 results.append(row)
assert np.array_equal(traces[0],traces[1]);(a.out/'result.json').write_text(json.dumps(dict(source=str(a.run),protocol='Same stored queue/archive, final encoder frozen, same future controls and fresh iid noise stream, only priority representation changes. Coordinates privileged. Additional actions counted separately.',results=results),indent=2));print(json.dumps(results,indent=2))
