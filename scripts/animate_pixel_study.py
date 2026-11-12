"""Show actual noisy agent inputs beside actual endpoint coverage."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,PillowWriter
from matplotlib.collections import LineCollection
p=argparse.ArgumentParser();p.add_argument('runs',nargs='+',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
fig,axes=plt.subplots(len(a.runs),2,figsize=(9,4*len(a.runs)),squeeze=False)
items=[]
for i,path in enumerate(a.runs):
 s=json.loads((path/'spec.json').read_text());d=dict(np.load(path/'trajectory.npz'));images=np.load(path/'images.npy',mmap_mode='r')
 ax=axes[i,0];ax.add_collection(LineCollection(d['walls'],colors='black',linewidths=.65));ax.set(xlim=(0,s['size']),ylim=(0,s['size']),aspect='equal')
 cover=ax.imshow(np.zeros_like(d['frames'][0]),origin='lower',extent=(0,s['size'],0,s['size']),vmin=0,vmax=1,cmap='Blues',alpha=.65)
 trace=LineCollection([],colors='red',linewidths=.5);ax.add_collection(trace)
 raw=images[0].transpose(1,2,0) if images.shape[1]==3 else images[0,0]
 frame=axes[i,1].imshow(raw,cmap='gray',vmin=0,vmax=255,origin='lower');axes[i,1].axis('off')
 label=('Frozen encoder + warmup' if s['policy']=='frozen' else ('Learned + warmup' if s['warm_steps'] else 'Learned, cold start')) if s['policy']!='walk' else f"Walk: persistence {s['persistence']:g}, reset probability {s['reset_probability']:g}"
 if s['policy']=='coordinate_oracle':label='Coordinate oracle frontier (privileged diagnostic)'
 if s['policy']=='learned':label+=f" · {s['dim']}D latent"
 items.append((s,d,images,cover,trace,frame,ax,label))
def draw(j):
 for s,d,images,cover,trace,frame,ax,name in items:
    n=max(500,int((j+1)/100*s['steps']//500)*500);k=n//500-1
    cover.set_data(d['frames'][k]);ids=np.arange(max(1,n-150),n+1)
    trace.set_segments(np.stack([d['positions'][d['parents'][ids]],d['positions'][ids]],1))
    frame.set_data(images[n].transpose(1,2,0) if images.shape[1]==3 else images[n,0])
    ax.set_title(f'{name}\n{n:,} actions • {100*d["frames"][k].mean():.1f}% • {int(d["frames"][k].sum()):,} bins',fontsize=10,pad=9)
draw(0)
extra=f" · doorway {items[0][0].get('gate_width',1):g}" if items[0][0]['family']=='gates' else ''
fig.suptitle(f"{items[0][0]['family'].capitalize()}{extra} maze · seed {items[0][0]['seed']} · actual noisy observations at right",fontsize=12,y=.985)
fig.subplots_adjust(left=.07,right=.98,bottom=.04,top=.93,hspace=.30,wspace=.14)
ani=FuncAnimation(fig,draw,frames=100,interval=100)
ani.save(a.out,writer=PillowWriter(fps=10));draw(99);fig.savefig(a.out.with_suffix('.png'),dpi=130)
