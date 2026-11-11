"""Joint sweep plots and full tables, preserving screening/confirmation separation."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.animation import FuncAnimation,PillowWriter
from map_learning.maze import Maze
from map_learning.config import load_config


def label(key):return f'α={key[0]:g}, β={key[1]:g}'


def read(root):
    rows=[]
    for j in json.loads((root/'manifest.json').read_text())['jobs']:
        d=Path(j['path']);r=json.loads((d/'result.json').read_text())
        pr=[json.loads(x) for x in (d/'progress.jsonl').read_text().splitlines()]
        auc=float(np.trapezoid([1/1784]+[x['coverage'] for x in pr],[0]+[x['interactions'] for x in pr])/100000)
        rows.append({**j,'auc':auc,'first_full':r['first_full'],'visited':r['visited'],'final':r['blocks'][-1]})
    return rows


def groups(rows):
    g=defaultdict(list)
    for r in rows:g[r['repulsion'],r['beta']].append(r)
    for rr in g.values():rr.sort(key=lambda x:x['seed'])
    return g


def curves(g,keys,path,title):
    seeds=sorted({r['seed'] for rr in g.values() for r in rr})
    fig,axes=plt.subplots(1,len(seeds),figsize=(14,4),squeeze=False,constrained_layout=True)
    for ax,seed in zip(axes.flat,seeds):
        for ki,k in enumerate(keys):
            for r in g[k]:
                if r['seed']!=seed:continue
                pr=[json.loads(x) for x in (Path(r['path'])/'progress.jsonl').read_text().splitlines()]
                ax.plot([x['interactions']/1000 for x in pr],[100*x['coverage'] for x in pr],label=label(k),color=['#888888','#377eb8','#57a65a','#e27f24'][ki])
        ax.set(title=f'Seed {seed}',xlabel='Real actions (thousands)',ylabel='Physical coverage (%)',ylim=(0,101));ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8);fig.suptitle(title);fig.savefig(path,dpi=150);plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--selected',default='0.03:1e-5');p.add_argument('--confirmation',type=Path);p.add_argument('--extra',type=Path);p.add_argument('--skip-animation',action='store_true')
    a=p.parse_args();assets=Path('docs/assets');primary=read(a.root);rows=primary+(read(a.extra) if a.extra else []);g=groups(rows);grid_groups=groups(primary)
    selected=tuple(map(float,a.selected.split(':')))
    keys=list(dict.fromkeys([(.01,0.),(.01,1e-6),(selected[0],0.),selected]))
    reps=sorted({k[0] for k in grid_groups});betas=sorted({k[1] for k in grid_groups})
    metrics=[('Mean coverage over 100k (%)',lambda r:100*r['auc'],False,'viridis'),
             ('Final physical coverage (%)',lambda r:100*r['visited']/1784,False,'viridis'),
             ('Noise / movement distance (lower better)',lambda r:r['final']['noise_to_move_ratio'],True,'viridis_r'),
             ('Normalized mean-input prediction MSE',lambda r:r['final']['seen_normalized_mse'],True,'viridis_r'),
             ('Unweighted hinge repulsion',lambda r:r['final']['repulsion'],True,'viridis_r'),
             ('Mean latent variance',lambda r:r['final']['bottleneck']['mean_latent_variance'],True,'viridis')]
    fig,axes=plt.subplots(3,2,figsize=(12,11),constrained_layout=True)
    for ax,(title,fun,log,cmap) in zip(axes.flat,metrics):
        val=np.array([[np.mean([fun(r) for r in grid_groups[rep,beta]]) for beta in betas] for rep in reps])
        im=ax.imshow(val,aspect='auto',cmap=cmap,norm=LogNorm(vmin=max(val.min(),1e-10),vmax=val.max()) if log else None)
        ax.set(xticks=range(len(betas)),xticklabels=[f'{b:g}' for b in betas],yticks=range(len(reps)),yticklabels=reps,xlabel='β (KL coefficient)',ylabel='α (repulsion coefficient)',title=title)
        for i in range(len(reps)):
            for j in range(len(betas)):
                ax.text(j,i,f'{val[i,j]:.3g}',ha='center',va='center',color='black',fontsize=11,
                        bbox=dict(facecolor='white',alpha=.75,edgecolor='none'))
        fig.colorbar(im,ax=ax,shrink=.8)
    fig.suptitle('Joint KL × repulsion sweep: means across seeds 0/1/2')
    fig.savefig(assets/'kl-repulsion-grid.png',dpi=160);plt.close(fig)
    curves(g,keys,assets/'kl-repulsion-screen.png','Screening seeds: higher-repulsion candidate selected before confirmation')
    confirm=read(a.confirmation) if a.confirmation else []
    if confirm:curves(groups(confirm),keys,assets/'kl-repulsion-confirm.png','New-seed confirmation: settings fixed before these runs')
    if not a.skip_animation:
        animate_keys=keys
        ag=groups(confirm) if confirm else g
        seeds=sorted({r['seed'] for r in ag[animate_keys[0]]})
        fig,axes=plt.subplots(len(animate_keys),3,figsize=(12,4*len(animate_keys)),constrained_layout=True);frames={};artists={}
        for i,k in enumerate(animate_keys):
            for j,r in enumerate(ag[k]):
                d=Path(r['path']);ax=axes[i,j];Maze(load_config(d/'config.json').maze).draw(ax)
                frames[i,j]=np.load(d/'frames.npz')
                artists[i,j]=(ax.scatter([],[],s=4,color=['#888888','#377eb8','#57a65a','#e27f24'][i],alpha=.7),ax.scatter([],[],s=25,c='red',zorder=6),ax.set_title('',fontsize=10))
                ax.set_xticks([]);ax.set_yticks([])
        def update(idx):
            for i,k in enumerate(animate_keys):
                for j,seed in enumerate(seeds):
                    f=frames[i,j];dots,here,title=artists[i,j]
                    dots.set_offsets(f['points'][f['visited'][idx]]);here.set_offsets(f['position'][idx:idx+1])
                    title.set_text(f'{label(k)} · seed {seed}\n{f["visited"][idx].sum()}/1784 observed')
            fig.suptitle(f'{frames[0,0]["interactions"][idx]:,} real actions · '+('new-seed confirmation' if confirm else 'screening seeds')+'\nRed = simulator-restored expansion; deterministic means in search')
        count=len(frames[0,0]['interactions'])
        anim=FuncAnimation(fig,update,frames=np.unique(np.linspace(0,count-1,100).astype(int)),interval=150)
        anim.save(assets/'kl-repulsion.gif',writer=PillowWriter(fps=7),dpi=85)
        update(count-1);fig.savefig(assets/'kl-repulsion-final.png',dpi=130);plt.close(fig)
    sections=[]
    for title,rr in [('Screening seeds 0/1/2',rows),('Confirmation seeds',confirm)]:
        if not rr:continue
        lines=[f'## {title}','', '| α | β | Mean coverage over run | Final coverage | Completion actions by seed | Noise / movement | Normalized MSE |', '|---:|---:|---:|---:|---|---:|---:|']
        for (rep,beta),gg in sorted(groups(rr).items()):
            times=' / '.join(f'{r["first_full"]:,}' if r['first_full'] else '>100k' for r in gg)
            line=f'| {rep:g} | {beta:g} | {100*np.mean([r["auc"] for r in gg]):.1f}% | {100*np.mean([r["visited"]/1784 for r in gg]):.1f}% | {times} | {np.mean([r["final"]["noise_to_move_ratio"] for r in gg]):.3g} | {np.mean([r["final"]["seen_normalized_mse"] for r in gg]):.3g} |'
            lines.append(line);print(line)
        sections.append('\n'.join(lines))
    (assets/'kl-repulsion-summary.json').write_text(json.dumps({'selected':list(selected),'screen':rows,'confirmation':confirm},indent=2)+'\n')
    Path('docs/kl-repulsion-tables.md').write_text('# Joint sweep: complete results\n\n'+ '\n\n'.join(sections)+'\n')
    print('Selected higher-repulsion arm:',selected)

if __name__=='__main__':main()
