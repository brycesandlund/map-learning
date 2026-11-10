"""Replay contemporaneous reward maps, state tallies, and model-search diagnostics."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import SymLogNorm
import numpy as np
from animate_exploration import load_run
from map_learning.maze import Maze


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs',nargs='+',type=Path)
    parser.add_argument('--labels',nargs='+')
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--still',type=Path,help='Save selected frame as a PNG as well')
    parser.add_argument('--still-step',type=int,help='Nearest snapshot; default final frame')
    parser.add_argument('--fps',type=int,default=4)
    parser.add_argument('--trail',type=int,default=200)
    args=parser.parse_args()
    if args.out.exists():parser.error(f'Refusing to overwrite {args.out}')
    if args.labels and len(args.labels)!=len(args.runs):parser.error('One label per run required')
    if args.fps<1 or args.trail<1:parser.error('Positive fps and trail required')
    runs=[]
    for folder,label in zip(args.runs,args.labels or [p.name for p in args.runs]):
        name,cfg,replay=load_run(folder,label)
        with np.load(folder/'reward_snapshots.npz',allow_pickle=False) as saved:
            snapshots={k:saved[k].copy() for k in saved.files}
        if len(snapshots['points'])!=replay['reachable']:raise ValueError('Map size mismatch')
        for step,pos,action in zip(snapshots['step'],snapshots['position'],snapshots['action']):
            np.testing.assert_allclose(pos,replay['positions'][step])
        with np.load(folder/'replay.npz') as raw:
            np.testing.assert_array_equal(snapshots['action'],raw['actions'][snapshots['step']])
        runs.append((name,cfg,replay,snapshots))
    steps=sorted(set.intersection(*(set(r[3]['step'].tolist()) for r in runs)))
    if not steps:parser.error('No shared snapshot times')
    fig,axes=plt.subplots(3,len(runs),figsize=(max(7,4.7*len(runs)),11.5),squeeze=False,
                          gridspec_kw={'height_ratios':[1,1.15,.72]})
    artists=[]
    directions=Maze.ACTIONS
    for col,(label,cfg,data,snap) in enumerate(runs):
        top,heat,bars=axes[:,col]
        env=Maze(cfg.maze);env.draw(top);env.draw(heat)
        for ax in (top,heat):
            ax.set_xticks(np.arange(0,cfg.maze.size+1,3));ax.set_yticks(np.arange(0,cfg.maze.size+1,3))
            ax.tick_params(labelsize=8)
        top.set_title(label,fontsize=12,pad=20)
        tally=top.text(.5,1.015,'',transform=top.transAxes,ha='center',fontsize=9)
        visited=top.scatter([],[],s=8,color='#8eb8cb',alpha=.8,zorder=1)
        path,=top.plot([],[],color='#e87524',lw=1.1,zorder=5)
        agent=top.scatter([],[],s=35,color='#dc2727',edgecolors='white',linewidths=.6,zorder=7)
        heat.set_title('Max immediate reward across actions\nArrows: best action; diamond: global peak',fontsize=9)
        post=snap['step']>=cfg.warmup_steps
        values=snap['reward'][post] if post.any() else snap['reward']
        vmax=max(float(np.quantile(values,.99)),1e-12)
        norm=SymLogNorm(linthresh=vmax/1000,vmin=0,vmax=vmax,clip=True)
        points=snap['points'];score=heat.scatter(*points.T,c=np.zeros(len(points)),s=23,marker='s',
                                               cmap='magma',norm=norm,zorder=1)
        subset=np.flatnonzero(np.isclose(points*2,np.round(points*2)).all(1))
        if not len(subset):subset=np.arange(0,len(points),4)
        arrows=heat.quiver(points[subset,0],points[subset,1],np.zeros(len(subset)),np.zeros(len(subset)),
                           color='#9cdbc8',alpha=.9,angles='xy',scale_units='xy',scale=1,width=.0035,zorder=3)
        stay=heat.scatter([],[],s=9,facecolors='none',edgecolors='#9cdbc8',linewidths=.5,zorder=4)
        current=heat.scatter([],[],s=35,color='#dc2727',edgecolors='white',linewidths=.8,zorder=6)
        chosen=heat.quiver([0],[0],[0],[0],color='#00d7ff',angles='xy',scale_units='xy',scale=1,width=.012,zorder=8)
        chosen_stay=heat.scatter([],[],s=95,facecolors='none',edgecolors='#00d7ff',linewidths=2,zorder=8)
        peak=heat.scatter([],[],marker='D',s=45,facecolors='none',edgecolors='#2df4ff',linewidths=1.4,zorder=5)
        colorbar=fig.colorbar(score,ax=heat,orientation='horizontal',pad=.07,fraction=.045,
                             ticks=[0,vmax/100,vmax/10,vmax],format='%.1g',extend='max')
        colorbar.ax.tick_params(labelsize=7)
        colorbar.set_label('Normalized reward · fixed per-run scale; top 1% clipped',fontsize=7)
        peaktext=heat.text(.5,-.30,'',transform=heat.transAxes,ha='center',fontsize=7)
        x=np.arange(5)
        immediate=bars.bar(x-.24,np.zeros(5),.24,color='#aaaaaa',label='Immediate reward')
        exact=bars.bar(x,np.zeros(5),.24,color='#3d7cb8',label='Exact horizon return')
        search=bars.bar(x+.24,np.zeros(5),.24,color='#e89945',label='MCTS estimate')
        selected=bars.scatter([],[],marker='v',s=45,color='#d52222',zorder=5,label='Chosen action')
        bars.set_xticks(x,['Right','Left','Up','Down','Stay'],fontsize=8)
        bars.ticklabel_format(axis='y',style='sci',scilimits=(0,0));bars.tick_params(axis='y',labelsize=7)
        bars.grid(axis='y',alpha=.2)
        bars.set_title('Action values at the agent',fontsize=10)
        if col==0:bars.legend(fontsize=7,loc='upper left')
        info=bars.text(.5,-.25,'',transform=bars.transAxes,ha='center',fontsize=8)
        artists.append(dict(visited=visited,path=path,agent=agent,tally=tally,score=score,arrows=arrows,stay=stay,
                            subset=subset,current=current,chosen=chosen,chosen_stay=chosen_stay,peak=peak,
                            peaktext=peaktext,immediate=immediate,exact=exact,search=search,selected=selected,bars=bars,info=info))
    title=fig.suptitle('',fontsize=13)
    fig.subplots_adjust(top=.91,bottom=.09,left=.05,right=.98,hspace=.5,wspace=.3)
    fig.text(.5,.025,'Snapshots are BEFORE the shown action. Heatmap evaluates real maze states; exact search uses imagined model states.\n'
             'Blue dots: visits · Orange: recent path · Red: agent · Cyan arrow/ring: chosen move/stay. High reward may lie beyond the search horizon.',
             ha='center',fontsize=8)

    def update(step):
        title.set_text(f'After {step:,} actions — evaluating the choice for action {step+1:,}')
        for (_,cfg,data,snap),art in zip(runs,artists):
            frame=int(np.searchsorted(snap['step'],step));pos=snap['position'][frame];action=int(snap['action'][frame])
            seen=data['unique'][data['first']<2*step]
            seen=np.unique(np.concatenate([seen,pos[None]]),axis=0)
            art['visited'].set_offsets(seen)
            art['tally'].set_text(f"Visited {len(seen):,} / {data['reachable']:,} states ({len(seen)/data['reachable']:.1%})")
            start=max(int(data['episode_start'][step]),step-args.trail)
            trail=np.concatenate([data['positions'][start:step],pos[None]])
            art['path'].set_data(*trail.T);art['agent'].set_offsets(pos[None]);art['current'].set_offsets(pos[None])
            rewards=snap['reward'][frame];maxima=rewards.max(1);best=rewards.argmax(1)
            art['score'].set_array(maxima)
            vectors=.24*directions[best[art['subset']]]
            art['arrows'].set_UVC(vectors[:,0],vectors[:,1])
            art['stay'].set_offsets(snap['points'][art['subset'][best[art['subset']]==4]])
            art['chosen'].set_offsets(pos[None]);art['chosen'].set_UVC([.6*directions[action,0]],[.6*directions[action,1]])
            art['chosen_stay'].set_offsets(pos[None] if action==4 else np.empty((0,2)))
            peak=int(maxima.argmax());art['peak'].set_offsets(snap['points'][peak:peak+1])
            art['peaktext'].set_text(f"Peak {maxima[peak]:.3g} at ({snap['points'][peak,0]:g}, {snap['points'][peak,1]:g})")
            arrays=[snap[k][frame] for k in ['immediate','exact_q','search_q']]
            for key,vals in zip(['immediate','exact','search'],arrays):
                for bar,val in zip(art[key],vals):bar.set_height(val)
            maxq=max(float(np.nanmax(np.concatenate(arrays))),1e-12)
            art['bars'].set_ylim(0,maxq*1.3)
            art['selected'].set_offsets([[action,maxq*1.13]])
            kind=str(snap['decision'][frame]);q=arrays[1]
            counts=snap['root_visits'][frame]
            art['bars'].set_xticks(np.arange(5),[f'{s}\nn={v}' if kind=='mcts' else s for s,v in zip(['Right','Left','Up','Down','Stay'],counts)],fontsize=8)
            if kind=='mcts' and np.isfinite(q).all():
                gap=max(0.,1-float(q[action])/float(q.max())) if q.max()>0 else 0.
                art['info'].set_text(f'MCTS choice: {gap:.1%} below exact best H={cfg.planner.horizon} return')
            elif kind=='mcts':art['info'].set_text(f'MCTS choice — exact H={cfg.planner.horizon} enumeration omitted')
            else:art['info'].set_text(f'{kind} choice — not a MCTS decision')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    if args.still:
        frame=steps[-1] if args.still_step is None else min(steps,key=lambda s:abs(s-args.still_step))
        update(frame);args.still.parent.mkdir(parents=True,exist_ok=True);fig.savefig(args.still,dpi=130)
    animation=FuncAnimation(fig,update,frames=steps,interval=1000/args.fps)
    animation.save(args.out,writer=PillowWriter(fps=args.fps));plt.close(fig)
    print(args.out)


if __name__=='__main__':main()
