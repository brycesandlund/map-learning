"""Visualize learned feature attention, using only stored observed images."""
import argparse,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from map_learning.pixel_exploration import PixelLearner
p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
s=SimpleNamespace(**json.loads((a.run/'spec.json').read_text()));s.device='cpu';torch.set_num_threads(1)
m=PixelLearner(s);m.encoder.load_state_dict(torch.load(a.run/'checkpoint.pt',weights_only=True,map_location='cpu')['encoder'])
images=np.load(a.run/'images.npy',mmap_mode='r');ids=[0,len(images)//3,2*len(images)//3];raw=np.array(images[ids])
with torch.no_grad():
 x=torch.from_numpy(raw).float()/255-.5-m.encoder.background;x-=x.mean((-2,-1),keepdim=True)
 f=m.encoder.features(x);att=(f.flatten(2)*m.encoder.temperature).softmax(-1).reshape(f.shape).numpy()
fig,axes=plt.subplots(3,9,figsize=(18,6))
for row in range(3):
 axes[row,0].imshow(raw[row].transpose(1,2,0) if raw.shape[1]==3 else raw[row,0],origin='lower',cmap='gray',vmin=0,vmax=255)
 axes[row,0].set_title(f'Observed frame {ids[row]}',fontsize=8)
 for ch in range(8):
  axes[row,ch+1].imshow(att[row,ch],origin='lower',cmap='inferno',interpolation='nearest');axes[row,ch+1].set_title(f'Attention {ch+1}',fontsize=8)
 for ax in axes[row]:ax.axis('off')
fig.suptitle('Learned spatial attention — no supplied masks or position labels');fig.tight_layout();fig.savefig(a.out,dpi=140)
