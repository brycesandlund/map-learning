"""Evaluation-only spatial maps under fresh observation noise."""
import argparse,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from map_learning.config import MazeConfig
from map_learning.pixel_exploration import PixelMaze,PixelLearner
p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
s=SimpleNamespace(**json.loads((a.run/'spec.json').read_text()));torch.set_num_threads(1)
m=PixelLearner(s);ck=torch.load(a.run/'checkpoint.pt',weights_only=True);m.encoder.load_state_dict(ck['encoder'],strict=False)
e=PixelMaze(MazeConfig(size=s.size,seed=s.layout,step_size=s.step_size,image_size=s.image_size),98271,s.family,s.noise,getattr(s,'rgb',0),getattr(s,'marker_size',.85),getattr(s,'noise_mode','full'))
t=dict(np.load(a.run/'trajectory.npz'));images=np.load(a.run/'images.npy',mmap_mode='r')
o=(np.arange(s.size*4)+.5)/4;xx,yy=np.meshgrid(o,o);pts=np.stack([xx.ravel(),yy.ravel()],1)
def encode_images(ims):return np.concatenate([m.encode(ims[i:i+256]) for i in range(0,len(ims),256)])
z=encode_images(np.stack([e.observe(p) for p in pts]));replica=encode_images(np.stack([e.observe(p) for p in pts]))
visited=np.unique(np.r_[0,t['parents'][max(1,s.warm_steps+1):]])
vz=encode_images(images[visited]);nov=cKDTree(vz).query(z)[0]
fig,axes=plt.subplots(2,2,figsize=(10,9))
for ax,data,title in [(axes[0,0],t['frames'][-1],'Actual endpoint coverage'),(axes[1,0],np.linalg.norm(z-replica,axis=1).reshape(xx.shape),'Fresh-noise latent displacement'),(axes[1,1],nov.reshape(xx.shape),'Nearest expanded observation: latent distance')]:
 im=ax.imshow(data,origin='lower',extent=(0,s.size,0,s.size),cmap='viridis')
 ax.add_collection(LineCollection(e.walls,colors='white',linewidths=.5));ax.set_title(title,fontsize=10);fig.colorbar(im,ax=ax,shrink=.7)
mean=(z+replica)/2;u,sv,v=np.linalg.svd(mean-mean.mean(0),full_matrices=False);pc=(mean-mean.mean(0))@v[:2].T
axes[0,1].scatter(pc[:,0],pc[:,1],c=pts[:,0]+pts[:,1]/s.size,s=3,cmap='turbo');axes[0,1].set_title('Latent PCA; color = physical x + y/size',fontsize=10)
fig.suptitle(a.run.name+' — offline full-map evaluation',fontsize=11);fig.tight_layout();fig.savefig(a.out,dpi=140)
