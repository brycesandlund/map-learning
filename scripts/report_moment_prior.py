"""All moment-prior outcomes, old learned controls, and separately selected confirmation."""
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


def read(root):
    rows=[]
    for j in json.loads((root/'manifest.json').read_text())['jobs']:
        if j.get('role')=='zero_reproduction':continue
        d=Path(j['path']);r=json.loads((d/'result.json').read_text())
        pr=[json.loads(x) for x in (d/'progress.jsonl').read_text().splitlines()]
        auc=float(np.trapezoid([1/1784]+[x['coverage'] for x in pr],[0]+[x['interactions'] for x in pr])/100000)
        rows.append({**j,'coverage_auc':auc,'first_full':r['first_full'],'visited':r['visited'],'final':r['blocks'][-1]})
    return rows


def groups(rows):
    g=defaultdict(list)
    for r in rows:
        mode=r['prior_mode']+(f'_warm{r["warmup_updates"]}' if r.get('warmup_updates',0) else '')
        g[mode,r['repulsion'],r['beta']].append(r)
    for rr in g.values():rr.sort(key=lambda r:r['seed'])
    return g


def label(k):
    base=k[0].split('_warm')[0]
    mode={'learned':'Learned','replay_moments':'Replay fit','batch_moments':'Batch fit','update_moments':'Replay fit/update'}[base]
    if '_warm' in k[0]:mode+=', delayed 35k'
    return f'{mode}, α={k[1]:g}, β={k[2]:g}' if k[2] else f'β=0, α={k[1]:g}'


def curves(g,keys,path,title,colors):
    seeds=sorted({r['seed'] for k in keys for r in g[k]});fig,axes=plt.subplots(1,len(seeds),figsize=(14,4),layout='constrained')
    for ax,seed in zip(np.atleast_1d(axes),seeds):
        for k in keys:
            for r in g[k]:
                if r['seed']!=seed:continue
                pr=[json.loads(x) for x in (Path(r['path'])/'progress.jsonl').read_text().splitlines()]
                ax.plot([x['interactions']/1000 for x in pr],[100*x['coverage'] for x in pr],label=label(k),color=colors[k])
        ax.set(title=f'Seed {seed}',xlabel='Real actions (thousands)',ylabel='Physical coverage (%)',ylim=(0,101));ax.grid(alpha=.2)
    np.atleast_1d(axes)[0].legend(fontsize=7);fig.suptitle(title);fig.savefig(path,dpi=155);plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument('--screen-roots',nargs='+',required=True);p.add_argument('--confirmation',type=Path)
    p.add_argument('--selected',required=True);p.add_argument('--skip-animation',action='store_true');a=p.parse_args()
    mode,rep,beta=a.selected.split(':');selected=(mode,float(rep),float(beta))
    rows=[r for root in a.screen_roots for r in read(Path(root))];g=groups(rows)
    old=json.loads(Path('docs/assets/kl-repulsion-audit.json').read_text())
    controls=[{**r,'prior_mode':'learned','role':'reference'} for r in old if r['seed']<3 and r['repulsion']==.03 and r['beta']==1e-5]
    gg=groups(rows+controls)
    keys=list(dict.fromkeys([('learned',.03,1e-5),('replay_moments',.03,1e-5),('replay_moments',selected[1],0),selected]))
    keys=[k for k in keys if k in gg]
    colors={k:'#777777' if k[2]==0 else '#377eb8' if k[0]=='learned' else '#e27f24' if k==selected else '#984ea3' for k in keys}
    assets=Path('docs/assets')
    curves(gg,keys,assets/'moment-prior-screen.png','Screening seeds: moment-fit candidate versus learned-prior control',colors)
    # Primary Cartesian grid only; extra screens are retained in full tables.
    primary=groups(read(Path(a.screen_roots[0])));reps=sorted({k[1] for k in primary});betas=sorted({k[2] for k in primary})
    fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    metrics=[('Mean coverage over 100k (%)',lambda r:100*r['coverage_auc'],False,'viridis'),
             ('Final physical coverage (%)',lambda r:100*r['visited']/1784,False,'viridis'),
             ('Full coverage (of 3 seeds)',lambda r:float(r['first_full'] is not None),False,'viridis'),
             ('Observation-noise / movement distance',lambda r:r['final']['noise_to_move_ratio'],True,'viridis_r'),
             ('Variance of latent means',lambda r:r['final']['bottleneck']['mean_latent_variance'],True,'viridis'),
             ('Mean posterior standard deviation',lambda r:r['final']['bottleneck']['posterior_std_mean'],True,'viridis')]
    for ax,(title,fun,log,cmap) in zip(axes.flat,metrics):
        value=np.array([[np.mean([fun(r) for r in primary['replay_moments',rep,beta]]) for beta in betas] for rep in reps])
        if title.startswith('Full coverage'):value*=3
        im=ax.imshow(value,aspect='auto',cmap=cmap,norm=LogNorm(vmin=max(value.min(),1e-12),vmax=value.max()) if log else None)
        ax.set(xticks=range(len(betas)),xticklabels=[f'{b:g}' for b in betas],yticks=range(len(reps)),yticklabels=reps,xlabel='β',ylabel='Repulsion α',title=title)
        for i in range(len(reps)):
            for j in range(len(betas)):ax.text(j,i,f'{value[i,j]:.3g}',ha='center',va='center',fontsize=9,bbox=dict(facecolor='white',alpha=.75,edgecolor='none'))
        fig.colorbar(im,ax=ax,shrink=.8)
    fig.suptitle('Full-replay moment fits before each training event: primary three-seed grid')
    fig.savefig(assets/'moment-prior-grid.png',dpi=155);plt.close(fig)
    confirm=read(a.confirmation) if a.confirmation else [];cg=groups(confirm)
    if confirm:
        colors.update({k:'#777777' if k[2]==0 else '#377eb8' if k[0]=='learned' else '#e27f24' for k in cg})
        curves(cg,list(cg),assets/'moment-prior-confirm.png','New seeds: settings fixed before confirmation',colors)
    if not a.skip_animation:
        ag=cg if confirm else {k:gg[k] for k in keys};ak=list(ag);seeds=sorted({r['seed'] for rr in ag.values() for r in rr})
        fig,axes=plt.subplots(len(ak),len(seeds),figsize=(12,4*len(ak)),squeeze=False,layout='constrained');frames={};artists={}
        for i,k in enumerate(ak):
            for j,r in enumerate(ag[k]):
                d=Path(r['path']);ax=axes[i,j];Maze(load_config(d/'config.json').maze).draw(ax)
                frames[i,j]=np.load(d/'frames.npz')
                artists[i,j]=(ax.scatter([],[],s=4,color=colors[k],alpha=.7),ax.scatter([],[],s=24,c='red',zorder=6),ax.set_title('',fontsize=9))
                ax.set_xticks([]);ax.set_yticks([])
        def update(idx):
            for i,k in enumerate(ak):
                for j,seed in enumerate(seeds):
                    f=frames[i,j];dots,here,title=artists[i,j]
                    dots.set_offsets(f['points'][f['visited'][idx]]);here.set_offsets(f['position'][idx:idx+1]);title.set_text(f'{label(k)} · seed {seed}\n{f["visited"][idx].sum()}/1784 observed')
            fig.suptitle(f'{frames[0,0]["interactions"][idx]:,} real actions · '+('confirmation seeds' if confirm else 'screening seeds')+'\nRed = simulator-restored expansion; deterministic means in all search')
        count=len(frames[0,0]['interactions']);anim=FuncAnimation(fig,update,frames=np.unique(np.linspace(0,count-1,100).astype(int)),interval=150)
        anim.save(assets/'moment-prior.gif',writer=PillowWriter(fps=7),dpi=85)
        update(count-1);fig.savefig(assets/'moment-prior-final.png',dpi=130);plt.close(fig)
    sections=[]
    for title,rr in [('Screening (seeds 0/1/2)',rows+controls),('Confirmation',confirm)]:
        if not rr:continue
        lines=[f'## {title}','', '| Prior | α | β | Mean coverage over run | Final coverage | Completion actions by seed | Noise / movement | Normalized MSE |', '|---|---:|---:|---:|---:|---|---:|---:|']
        for (mode,rep,beta),rs in sorted(groups(rr).items()):
            times=' / '.join(f'{r["first_full"]:,}' if r['first_full'] else '>100k' for r in rs)
            line=f'| {mode} | {rep:g} | {beta:g} | {100*np.mean([r["coverage_auc"] for r in rs]):.1f}% | {100*np.mean([r["visited"]/1784 for r in rs]):.1f}% | {times} | {np.mean([r["final"]["noise_to_move_ratio"] for r in rs]):.3g} | {np.mean([r["final"]["seen_normalized_mse"] for r in rs]):.3g} |'
            lines.append(line);print(line)
        sections.append('\n'.join(lines))
    Path('docs/moment-prior-tables.md').write_text('# Complete moment-prior results\n\n'+'\n\n'.join(sections)+'\n')
    (assets/'moment-prior-summary.json').write_text(json.dumps({'selected':list(selected),'screen':rows,'references':controls,'confirmation':confirm},indent=2)+'\n')

if __name__=='__main__':main()
