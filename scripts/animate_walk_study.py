"""Actual transition segments only: restores never become lines of travel."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation,PillowWriter
from map_learning.maze import Maze
from map_learning.config import load_config
p=argparse.ArgumentParser();p.add_argument('results',type=Path,nargs='+');p.add_argument('--selection',type=Path,required=True);p.add_argument('--step',type=float,required=True);p.add_argument('--seed',type=int,default=92);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
runs={r['name']:r for f in a.results for report in json.loads(f.read_text()) for r in report['runs'] if r['seed']==a.seed and r['layout']==7};choices=[x['arm'] for x in json.loads(a.selection.read_text()) if x['arm']['step_size']==a.step]
walk=next(x['name'] for x in choices if x['objective']=='walk' and not x.get('reset_every',0));reset=next(x['name'] for x in choices if x['objective']=='walk' and x.get('reset_every',0))
hybrid={o:next(x['name'] for x in choices if x['objective']==o) for o in ['dynamics','adjacency']}
names=[walk,reset,f'dynamics-base-s{a.step}',hybrid['dynamics'],f'adjacency-base-s{a.step}',hybrid['adjacency']]
labels=['Persistent walk','Walk + random resets','Dynamics frontier','Dynamics hybrid','Adjacency frontier','Adjacency hybrid']
fig,axes=plt.subplots(2,3,figsize=(12,9));data=[]
for ax,name,label in zip(axes.flat,names,labels):
 r=runs[name];path=Path(r['run']);e=Maze(load_config(path/'config.json').maze);e.draw(ax)
 d=np.load(path/'replay.npz');frames=np.load(path/'frames.npz');scatter=ax.scatter([],[],s=.7,alpha=.4,color='#3679ae');recent=LineCollection([],colors='#e94b50',linewidths=.6,alpha=.7);ax.add_collection(recent)
 tally=ax.text(.5,-.09,'',transform=ax.transAxes,ha='center',fontsize=8);ax.set_title(label)
 ids=np.linspace(0,len(d['actions'])-1,min(12000,len(d['actions'])),dtype=int)
 data.append((r,d,frames,ids,scatter,recent,tally))
title=fig.suptitle('');fig.text(.5,.015,'Blue: sampled observed endpoints • Red: last 200 actual transitions • Simulator restores are never connected',ha='center',fontsize=9)
fig.tight_layout(rect=[0,.04,1,.95])
def update(step):
 title.set_text(f'Maximum move {a.step} • Seed {a.seed} • {step:,} executed actions')
 for r,d,f,ids,scatter,recent,tally in data:
  scatter.set_offsets(d['next_positions'][ids[ids<step]])
  lo=max(0,step-200);recent.set_segments(np.stack([d['positions'][lo:step],d['next_positions'][lo:step]],axis=1))
  i=np.searchsorted(f['interactions'],step,side='right')-1;mask=f['occupied_k4'][i]
  phase='walk' if step<=r['walk_steps'] else 'frontier'
  tally.set_text(f'{mask.sum():,}/{mask.size:,} bins ({mask.mean():.1%}) • {phase}')
animation=FuncAnimation(fig,update,frames=range(1000,100001,1000),interval=100)
animation.save(str(a.out)+'.gif',writer=PillowWriter(fps=10));update(100000);fig.savefig(str(a.out)+'-final.png',dpi=130);plt.close(fig)
