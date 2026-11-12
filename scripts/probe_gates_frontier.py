"""Post-training queue audit for the next undiscovered doorway; no agent feedback."""
import argparse,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from scipy.spatial import cKDTree
from map_learning.pixel_exploration import PixelLearner
p=argparse.ArgumentParser();p.add_argument('runs',nargs='+',type=Path);p.add_argument('--out',required=True,type=Path);a=p.parse_args();torch.set_num_threads(1);rows=[]
for path in a.runs:
 s=SimpleNamespace(**json.loads((path/'spec.json').read_text()));d=dict(np.load(path/'trajectory.npz'));pos=d['positions'];maxx=pos[:,0].max();gates=[i for i in range(1,4) if maxx<=s.size*i/4]
 if not gates:rows.append(dict(run=str(path),all_chambers_reached=True));continue
 gate=gates[0];x=s.size*gate/4;y=s.size*(1.73/12 if gate%2 else 1-1.73/12)
 visited=np.unique(np.r_[0,d['parents'][s.warm_steps+1:]]);active=np.setdiff1d(np.arange(1,len(pos)),visited)
 eligible=(abs(pos[active,1]-y)<s.gate_width/2)&(pos[active,0]<x)&(pos[active,0]>x-s.step_size)
 probability=np.where(eligible,np.clip((1-np.maximum(.5,(x-pos[active,0])/s.step_size))/.5,0,1),0)
 oracle=cKDTree(pos[visited]).query(pos[active])[0]
 m=PixelLearner(s);m.encoder.load_state_dict(torch.load(path/'checkpoint.pt',map_location='cpu',weights_only=True)['encoder']);ims=np.load(path/'images.npy',mmap_mode='r')
 def enc(ids):return np.concatenate([m.encode(ims[ids[j:j+512]]) for j in range(0,len(ids),512)])
 score,nearest=cKDTree(enc(visited)).query(enc(active))
 row=dict(run=str(path),all_chambers_reached=False,next_gate=gate,eligible_occurrences=int(eligible.sum()),active=len(active),distinct_eligible_positions=len(np.unique(pos[active[eligible]],axis=0)))
 for name,values in [('latent',score),('coordinate',oracle)]:
  order=np.argsort(-values);rank=np.empty(len(order),int);rank[order]=np.arange(len(order))+1
  row[name]=dict(best_eligible_rank=int(rank[eligible].min()) if eligible.any() else None,top100_cross_probability_sum=float(probability[order[:100]].sum()),top1000_cross_probability_sum=float(probability[order[:1000]].sum()))
 row['discovered_regions']=[]
 source_positions=pos[d['parents'][1:]]
 for chamber in range(1,4):
  region=(pos[:,0]>s.size*chamber/4)&(pos[:,0]<s.size*(chamber+1)/4)
  ids=np.flatnonzero(region)
  if not len(ids):continue
  pending=region[active];nr=dict(chamber=chamber+1,first_observation=int(ids[0]),observations=len(ids),source_actions=int(((source_positions[:,0]>s.size*chamber/4)&(source_positions[:,0]<s.size*(chamber+1)/4)).sum()),pending=int(pending.sum()))
  for name,values in [('latent',score),('coordinate',oracle)]:
   order=np.argsort(-values);rank=np.empty(len(order),int);rank[order]=np.arange(len(order))+1
   nr[name+'_best_pending_rank']=int(rank[pending].min()) if pending.any() else None
  if pending.sum() and pending.sum()<=10:
   inds=np.flatnonzero(pending)
   nr['rare_pending']=[dict(id=int(active[k]),position=pos[active[k]].tolist(),nearest_latent_archive_position=pos[visited[nearest[k]]].tolist(),nearest_latent_physical_distance=float(np.linalg.norm(pos[active[k]]-pos[visited[nearest[k]]]))) for k in inds]
  row['discovered_regions'].append(nr)
 rows.append(row)
a.out.write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
