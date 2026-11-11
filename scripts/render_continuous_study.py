"""Scientific coverage plots and reset-safe continuous point animations."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from map_learning.config import load_config
from map_learning.maze import Maze


def main():
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--animate-root',type=Path);p.add_argument('--seed',type=int,default=30)
    p.add_argument('--animation-runs',type=Path,nargs='+')
    p.add_argument('--animation-labels',nargs='+')
    a=p.parse_args();reports=json.loads(a.results.read_text());a.out.parent.mkdir(parents=True,exist_ok=True)
    colors={'dynamics':'#386cb0','adjacency':'#ef8a25','oracle-xy':'#269378','random-queue':'#949494'}
    fig,axes=plt.subplots(2,2,figsize=(11,8))
    base=reports[0]
    for name in [arm['name'] for arm in base['manifest']['arms']]:
        group=[r for r in base['runs'] if r['name']==name]
        rows=[[json.loads(line) for line in (Path(r['run'])/'progress.jsonl').read_text().splitlines()] for r in group]
        for ax,k,label in zip(axes.flat,[1,2,4,8],['Rooms (1-unit bins)','0.5-unit bins','0.25-unit bins','0.125-unit bins']):
            curves=np.array([[row[f'coverage_k{k}']*100 for row in run] for run in rows]);x=np.array([r['interactions'] for r in rows[0]])/1000
            ax.plot(x,curves.mean(0),label=name,color=colors.get(name))
            ax.fill_between(x,curves.min(0),curves.max(0),alpha=.14,color=colors.get(name))
            ax.set(title=label,xlabel='Executed actions (thousands)',ylabel='Endpoint occupancy (%)',ylim=(0,102));ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=9);fig.suptitle('Continuous noisy maze: mean and seed range (initial 3 seeds)')
    fig.tight_layout();fig.savefig(str(a.out)+'.png',dpi=160);plt.close(fig)
    if not a.animate_root and not a.animation_runs:return
    names=a.animation_labels or ['dynamics','adjacency','oracle-xy','random-queue']
    paths=a.animation_runs or [a.animate_root/f'{n}-seed{a.seed}' for n in names]
    if len(paths)!=len(names):p.error('One label required per animation run')
    fig,axes=plt.subplots(1,len(paths),figsize=(4*len(paths),4.8),squeeze=False);axes=axes[0];data=[];artists=[]
    for ax,path,name in zip(axes,paths,names):
        cfg=load_config(path/'config.json');env=Maze(cfg.maze);env.draw(ax)
        replay=np.load(path/'replay.npz');frames=np.load(path/'frames.npz')
        data.append((replay['next_positions'],frames))
        scatter=ax.scatter([],[],s=1.1,c=colors.get(name,'#386cb0'),alpha=.45,zorder=2)
        current=ax.scatter([],[],s=30,c='#cc2525',zorder=5)
        tally=ax.text(.5,-.1,'',transform=ax.transAxes,ha='center',fontsize=9)
        ax.set_title(name);artists.append((scatter,current,tally))
    title=fig.suptitle('');fig.text(.5,.02,'Sample of actual continuous endpoints • Red: latest expansion • Restores are not drawn as travel',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.09,1,.92])
    def update(step):
        title.set_text(f'Continuous coordinates + observation noise • Seed {a.seed} • {step:,} actions')
        for (points,frames),(scatter,current,tally) in zip(data,artists):
            # Deterministic index subsampling affects display only, never metrics.
            take=np.linspace(0,step-1,min(step,12000),dtype=int)
            scatter.set_offsets(points[take]);i=np.searchsorted(frames['interactions'],step,side='right')-1
            current.set_offsets(frames['position'][i:i+1])
            covered=frames['occupied_k4'][i].mean();tally.set_text(f'0.25-bin occupancy: {covered:.1%}')
    limit=min(len(d[0]) for d in data);stride=max(1000,limit//100)
    steps=list(range(stride,limit+1,stride))
    animation=FuncAnimation(fig,update,frames=steps,interval=100)
    animation.save(str(a.out)+'.gif',writer=PillowWriter(fps=10));plt.close(fig)
    print(a.out)

if __name__=='__main__':main()
