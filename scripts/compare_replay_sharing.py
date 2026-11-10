"""Matched new+20 replay-sharing comparison; explicit new pass stays independent."""
from pathlib import Path
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from matplotlib.animation import FuncAnimation,PillowWriter
from map_learning.config import load_config
from map_learning.maze import Maze
from analyze_reward_stability import analyze,rank

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--mode',choices=['replay','epoch'],default='replay')
args=parser.parse_args()
epoch=args.mode=='epoch'
prefix='epoch-sharing' if epoch else 'replay-sharing'
heading='Full replay epoch every 10 actions' if epoch else 'New transitions once + 20 replay updates'
labels=['Independent epoch minibatches','Shared epoch minibatches'] if epoch else ['Independent replay batches','Shared replay batches']
caption=('Same data and initialization. Only per-member epoch minibatch alignment differs.' if epoch else
         'Same data, new-transition passes and initialization. Only the 20 replay minibatches differ.')
roots=[];stats=[]
for seed in range(3):
 a=Path(f'runs/ensemble-matched-20261109/epoch_independent-seed{seed}' if epoch else f'runs/ensemble-matched-20261109/new_then_replay_independent-seed{seed}')
 b=Path(f'runs/ensemble-matched-20261109/epoch_shared-seed{seed}' if epoch else f'runs/replay-only-sharing-20261110/new20_shared_replay_only-seed{seed}')
 with np.load(a/'replay.npz') as x,np.load(b/'replay.npz') as y:
  for k in x.files:np.testing.assert_array_equal(x[k],y[k])
 stats.append({'seed':seed,'independent':analyze(a),('shared_epoch' if epoch else 'shared_replay_only'):analyze(b)})
 if seed==0:roots=[a,b]
Path(f'docs/assets/{prefix}-results.json').write_text(json.dumps(stats,indent=2)+'\n')
s=[dict(np.load(p/'reward_snapshots.npz')) for p in roots]
np.testing.assert_array_equal(s[0]['step'],s[1]['step'])
steps=s[0]['step'];vmax=np.quantile(np.concatenate([x['reward'][steps>=2000].ravel() for x in s]),.99)
norm=SymLogNorm(vmin=0,vmax=vmax,linthresh=vmax/1000,clip=True)
fig,axes=plt.subplots(2,2,figsize=(9,9));arts=[]
for col,(data,label) in enumerate(zip(s,labels)):
 for row in range(2):
  ax=axes[row,col];Maze(load_config(roots[col]/'config.json').maze).draw(ax)
  ax.set_title(label+('\nCommon absolute reward scale' if row==0 else '\nWithin-frame reward percentile'))
  scatter=ax.scatter(*data['points'].T,c=np.zeros(len(data['points'])),s=15,marker='s',cmap='magma',norm=norm if row==0 else None,vmin=None if row==0 else 0,vmax=None if row==0 else 100)
  agent=ax.scatter([],[],s=35,c='cyan',edgecolors='black',zorder=5)
  fig.colorbar(scatter,ax=ax,fraction=.035,pad=.02)
  arts.append((scatter,agent,col,row))
title=fig.suptitle('');fig.tight_layout(rect=(0,.04,1,.95))
fig.text(.5,.015,caption+'\nAbsolute colors share one fixed scale; percentile colors show spatial ordering, not magnitude.',ha='center',fontsize=9)
def update(i):
 title.set_text(f'{heading} — after {steps[i]:,} actions')
 for scatter,agent,col,row in arts:
  values=s[col]['reward'][i].max(1)
  scatter.set_array(values if row==0 else 100*rank(values)/max(len(values)-1,1))
  agent.set_offsets(s[col]['position'][i:i+1])
update(int(np.argmin(abs(steps-8000))));fig.savefig(f'docs/assets/{prefix}.png',dpi=150)
FuncAnimation(fig,update,frames=range(len(steps)),interval=250).save(f'docs/assets/{prefix}.gif',writer=PillowWriter(fps=4));plt.close(fig)
print(json.dumps(stats,indent=2))
