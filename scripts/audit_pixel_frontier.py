"""Offline oracle check of whether highly ranked queue entries can discover bins.

No result is fed back to an agent. Twelve cardinal controls are diagnostic probes,
not an exhaustive test of continuous reachability or long-term destination value.
"""
import argparse,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from scipy.spatial import cKDTree
from map_learning.config import MazeConfig
from map_learning.pixel_exploration import PixelMaze,PixelLearner
p=argparse.ArgumentParser();p.add_argument('runs',nargs='+',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args();torch.set_num_threads(1);rows=[]
for path in a.runs:
 s=SimpleNamespace(**json.loads((path/'spec.json').read_text()));m=PixelLearner(s);m.encoder.load_state_dict(torch.load(path/'checkpoint.pt',weights_only=True,map_location='cpu')['encoder'])
 d=dict(np.load(path/'trajectory.npz'));images=np.load(path/'images.npy',mmap_mode='r')
 visited=np.unique(np.r_[0,d['parents'][s.warm_steps+1:]])
 active=np.setdiff1d(np.arange(1,len(images)),visited)
 def enc(ids):return np.concatenate([m.encode(images[ids[i:i+512]]) for i in range(0,len(ids),512)])
 score=cKDTree(enc(visited)).query(enc(active))[0];order=np.argsort(-score)
 rng=np.random.default_rng(59391);random=rng.choice(len(active),min(3000,len(active)),replace=False);sample=np.unique(np.r_[order[:1000],random])
 e=PixelMaze(MazeConfig(size=s.size,seed=s.layout,step_size=s.step_size,image_size=s.image_size),2,s.family,s.noise,s.rgb,s.marker_size,s.noise_mode,getattr(s,'gate_width',1.))
 covered=d['frames'][-1];productive=np.zeros(len(active),bool)
 for i in sample:
  pos=d['positions'][active[i]]
  for direction in e.ACTIONS[:4]:
   for length in [.5,.75,1.]:
    q=e.transition(pos,direction*np.float32(length));x,y=np.floor(q*4).astype(int)
    if not covered[y,x]:productive[i]=True;break
   if productive[i]:break
 rates={f'top{k}_productive_fraction':float(productive[order[:k]].mean()) for k in [10,100,1000]}
 rates['random_productive_fraction']=float(productive[random].mean())
 ranks=np.empty(len(active),int);ranks[order]=np.arange(len(active))
 rows.append(dict(run=str(path),active=len(active),probed=len(sample),productive_found=int(productive.sum()),best_productive_rank=int(ranks[productive].min())+1 if productive.any() else None,**rates))
a.out.write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
