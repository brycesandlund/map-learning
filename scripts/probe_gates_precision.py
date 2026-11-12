"""Offline nuisance sensitivity versus a doorway-width physical displacement."""
import argparse,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from map_learning.pixel_exploration import PixelMaze,PixelLearner
from map_learning.config import MazeConfig
p=argparse.ArgumentParser();p.add_argument('runs',type=Path,nargs='+');p.add_argument('--out',required=True,type=Path);a=p.parse_args();torch.set_num_threads(1);rows=[]
for path in a.runs:
 s=SimpleNamespace(**json.loads((path/'spec.json').read_text()));m=PixelLearner(s);m.encoder.load_state_dict(torch.load(path/'checkpoint.pt',map_location='cpu',weights_only=True)['encoder'])
 e=PixelMaze(MazeConfig(size=s.size,image_size=s.image_size,step_size=s.step_size),seed=4279,family=s.family,noise=s.noise,rgb=s.rgb,marker_size=s.marker_size,gate_width=s.gate_width)
 points=np.random.default_rng(981).uniform([.25,.25],[s.size-.25,s.size-.25],(256,2)).astype(np.float32)
 def encode(pts):
  ims=np.stack([e.observe(q) for q in np.repeat(pts,16,axis=0)])
  return np.concatenate([m.encode(ims[j:j+256]) for j in range(0,len(ims),256)]).reshape(len(pts),16,s.dim)
 z=encode(points);shifted=points+np.array([0,s.gate_width],np.float32);q=encode(shifted)
 # Noise is pairwise RMS; displacement uses independent 16-view centroid estimates.
 noise=np.sqrt(np.mean(np.sum((z[:,:8]-z[:,8:])**2,axis=-1),axis=1))
 movement=np.linalg.norm(z.mean(1)-q.mean(1),axis=1)
 # Estimate finite-MC centroid noise floor; this is descriptive, not an exact Jacobian.
 floor=noise/4
 rows.append(dict(run=str(path),dim=s.dim,width=s.gate_width,pair_noise_rms_mean=float(noise.mean()),gate_displacement_mean=float(movement.mean()),ratio_of_means=float(noise.mean()/movement.mean()),median_local_ratio=float(np.median(noise/np.maximum(movement,1e-9))),fraction_noise_exceeds_gate_displacement=float(np.mean(noise>movement)),centroid_noise_floor_mean=float(floor.mean())))
a.out.write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
