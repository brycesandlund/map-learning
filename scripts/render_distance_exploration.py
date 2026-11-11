"""Reset-safe frontier coverage animation, using saved physical evaluation frames."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,PillowWriter
from map_learning.config import load_config
from map_learning.maze import Maze


def main():
    p=argparse.ArgumentParser();p.add_argument('runs',type=Path,nargs='+');p.add_argument('--labels',nargs='+',required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--stride',type=int,default=1000);a=p.parse_args()
    if len(a.runs)!=len(a.labels):p.error('Need one label per run')
    if a.out.exists():raise FileExistsError(a.out)
    fig,axes=plt.subplots(1,len(a.runs),figsize=(4*len(a.runs),4.8),squeeze=False)
    data=[];artists=[]
    for ax,path,label in zip(axes[0],a.runs,a.labels):
        cfg=load_config(path/'config.json');d=dict(np.load(path/'frames.npz'));data.append(d)
        Maze(cfg.maze).draw(ax);ax.set_title(label,fontsize=10)
        ax.scatter(*d['points'].T,s=2,c='#e2e5e8')
        visits=ax.scatter([],[],s=5,c='#1992aa',alpha=.8)
        current=ax.scatter([],[],s=35,c='#df502b',zorder=4)
        tally=ax.text(.5,-.09,'',transform=ax.transAxes,ha='center',fontsize=9)
        artists.append((visits,current,tally))
    limit=min(int(d['interactions'][-1]) for d in data);frames=list(range(a.stride,limit+1,a.stride))
    if frames[-1]!=limit:frames.append(limit)
    title=fig.suptitle('')
    fig.text(.5,.02,'Blue: visited positions • Orange: most recent expansion • Simulator restores; no fictitious connecting paths',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.08,1,.9])
    def update(step):
        title.set_text(f'Persistent noisy frontier • {step:,} executed actions')
        for d,(visits,current,tally) in zip(data,artists):
            i=np.searchsorted(d['interactions'],step,side='right')-1
            if i<0:continue
            mask=d['visited'][i];visits.set_offsets(d['points'][mask]);current.set_offsets(d['position'][i:i+1])
            tally.set_text(f'{mask.sum():,} / {len(mask):,} positions ({mask.mean():.1%})')
    anim=FuncAnimation(fig,update,frames=frames,interval=100)
    a.out.parent.mkdir(parents=True,exist_ok=True);anim.save(a.out,writer=PillowWriter(fps=10));plt.close(fig)
    print(a.out)

if __name__=='__main__':main()
