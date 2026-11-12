"""Offline attention-temperature sensitivity on one fixed learned model."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from scipy.spatial import cKDTree
from map_learning.pixel_exploration import PixelMaze,PixelLearner
from map_learning.config import MazeConfig
root=Path('runs/pixel-attention-20261111/spatial-rgb0-temp200-seed110')
s=SimpleNamespace(**json.loads((root/'spec.json').read_text()));torch.set_num_threads(1)
m=PixelLearner(s);ck=torch.load(root/'checkpoint.pt',weights_only=True);m.encoder.load_state_dict(ck['encoder'])
e=PixelMaze(MazeConfig(size=s.size,seed=s.layout,step_size=s.step_size,image_size=s.image_size),98222,s.family,s.noise,s.rgb,s.marker_size)
rng=np.random.default_rng(98223);pts=rng.uniform(.1,s.size-.1,(1024,2));im=np.stack([e.observe(p) for p in pts]);rep=np.stack([e.observe(p) for p in pts])
rows=[]
for temp in [25,50,100,200,400]:
 m.encoder.temperature=temp
 z=np.concatenate([m.encode(im[i:i+128]) for i in range(0,len(im),128)])
 y=np.concatenate([m.encode(rep[i:i+128]) for i in range(0,len(im),128)])
 noise=np.linalg.norm(z-y,axis=1);distance=np.linalg.norm(pts[cKDTree(z).query(y)[1]]-pts,axis=1)
 rows.append(dict(temperature=temp,noise_median=float(np.median(noise)),noise_p99=float(np.quantile(noise,.99)),latent_std=float(z.std(0).mean()),retrieval_half_room=float((distance<.5).mean()),retrieval_median=float(np.median(distance))))
Path('docs/assets/pixel-temperature-probe.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
