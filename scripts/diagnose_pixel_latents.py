"""Offline noise invariance and geometry diagnostics; no feedback to agents."""
import argparse,json
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree
from map_learning.config import MazeConfig
from map_learning.pixel_exploration import PixelMaze,PixelLearner
p=argparse.ArgumentParser();p.add_argument('runs',nargs='+',type=Path);p.add_argument('--out',type=Path,required=True);p.add_argument('--positions',choices=['encountered','uniform'],default='encountered');a=p.parse_args()
torch.set_num_threads(1);rows=[]
for path in a.runs:
 spec=json.loads((path/'spec.json').read_text());cfg=SimpleNamespace(**spec);m=PixelLearner(cfg)
 ck=torch.load(path/'checkpoint.pt',weights_only=True);m.encoder.load_state_dict(ck['encoder'],strict=False);m.heads.load_state_dict(ck['heads'])
 e=PixelMaze(MazeConfig(size=cfg.size,seed=cfg.layout,step_size=cfg.step_size,image_size=cfg.image_size),98221,cfg.family,cfg.noise,getattr(cfg,'rgb',0),getattr(cfg,'marker_size',.85),getattr(cfg,'noise_mode','full'))
 traj=np.load(path/'trajectory.npz');rng=np.random.default_rng(0)
 pts=rng.uniform(.02,cfg.size-.02,(512,2)).astype(np.float32) if a.positions=='uniform' else traj['positions'][rng.choice(len(traj['positions']),512,replace=False)]
 def encode(points):
  return m.encode(np.stack([e.observe(p) for p in points]))
 z=encode(pts);replica=encode(pts)
 nn=cKDTree(z).query(replica)[1];physical_error=np.linalg.norm(pts[nn]-pts,axis=1)
 q=np.stack([e.transition(p,e.ACTIONS[i%4]) for i,p in enumerate(pts)])
 moved=np.linalg.norm(q-pts,axis=1)>1e-6;zq=encode(q)
 noise=np.linalg.norm(z-replica,axis=1);motion=np.linalg.norm(z-zq,axis=1)[moved]
 rows.append(dict(run=str(path),positions=a.positions,latent_std=float(z.std(0).mean()),noise_distance=float(noise.mean()),motion_distance=float(motion.mean()),noise_motion_ratio=float(noise.mean()/max(1e-12,motion.mean())),chance_within_half_cell=float((np.linalg.norm(pts[rng.permutation(len(pts))]-pts,axis=1)<.5).mean()),retrieval_median_physical_error=float(np.median(physical_error)),retrieval_within_half_cell=float((physical_error<.5).mean())))
a.out.write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
