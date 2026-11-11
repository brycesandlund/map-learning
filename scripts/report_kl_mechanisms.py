"""All outcomes of the paired KL mechanisms study, including online animations."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/map-learning-mpl')
os.environ.setdefault('XDG_CACHE_HOME','/tmp/map-learning-cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,PillowWriter
from map_learning.config import load_config
from map_learning.maze import Maze
from kl_mechanisms import ARMS,LABELS,COLORS


def read(root):
    rows=[]
    for j in json.loads((root/'manifest.json').read_text())['jobs']:
        d=Path(j['path']);r=json.loads((d/'result.json').read_text());row={**j,'result':r}
        if j['mode']=='online':
            pr=[json.loads(x) for x in (d/'progress.jsonl').read_text().splitlines()]
            row['coverage_auc']=float(np.trapezoid([1/1784]+[x['coverage'] for x in pr],[0]+[x['interactions'] for x in pr])/100000)
        rows.append(row)
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--fixed',type=Path,required=True);p.add_argument('--online',type=Path);p.add_argument('--confirmation',type=Path);p.add_argument('--skip-animation',action='store_true');a=p.parse_args()
    assets=Path('docs/assets');fixed=read(a.fixed);online=read(a.online) if a.online else []
    groups=defaultdict(list)
    for r in fixed:groups[r['family'],r['arm']].append(r)
    arms=list(ARMS);fig,axes=plt.subplots(2,2,figsize=(15,9),layout='constrained');ft=[]
    for i,family in enumerate(['winner','zero']):
        lines=[f'## Shared {family} data, models at 35k transitions','', '| Arm | Top100 productive (%) | Mean new states /640 actions (seed9 /10 /11) | Mean fitted KL at100k | Prior mismatch at100k |', '|---|---:|---|---:|---:|']
        for j,arm in enumerate(arms):
            rs=sorted(groups[family,arm],key=lambda r:r['seed']);qs=[next(q for q in r['result']['queue'] if q['prefix']==35000) for r in rs]
            yields=np.array([np.mean([v['rows'][-1]['new_states'] for v in q['live']]) for q in qs]);density=np.array([100*q['stats']['top100_productive_fraction'] for q in qs])
            for ax,val in zip(axes[i],[yields,density]):
                ax.bar(j,np.mean(val),color=COLORS[arm],alpha=.7)
                ax.scatter(j+np.linspace(-.15,.15,3),val,c='black',s=17,zorder=5)
                ax.set_xticks(range(len(arms)),[LABELS[x] for x in arms],rotation=45,ha='right');ax.grid(axis='y',alpha=.2)
            checks=[r['result']['queue'][-1]['moments'] for r in rs]
            lines.append(f'| {LABELS[arm]} | {density.mean():.2f} | '+ ' / '.join(f'{v:.1f}' for v in yields)+f' | {np.mean([c["matched_kl"] for c in checks]):.2f} | {np.mean([c["prior_mismatch"] for c in checks]):.2f} |')
        axes[i,0].set(title=f'{family} archive: frozen continuation',ylabel='New positions in 640 real actions')
        axes[i,1].set(title=f'{family} archive: productive queue entries',ylabel='Top100 productive entries (%)')
        ft.append('\n'.join(lines))
    fig.suptitle('Same observations, batches, visited sets and queues within each source/seed\nDots = model/source seed pairs; frozen yields average 3 environment-noise replicates')
    fig.savefig(assets/'kl-mechanisms-fixed.png',dpi=150);plt.close(fig)
    # Factorial contrasts calculated within source/model pair, not pooled snapshots.
    contrasts=[]
    for family in ['winner','zero']:
        for seed in [9,10,11]:
            values={}
            for arm in arms:
                r=next(r for r in groups[family,arm] if r['seed']==seed);q=next(q for q in r['result']['queue'] if q['prefix']==35000)
                values[arm]=float(np.mean([v['rows'][-1]['new_states'] for v in q['live']]))
            contrasts.append({'family':family,'seed':seed,'values':values,
                'noise_without_kl':values['noise_zero']-values['mean_zero'],
                'kl_without_noise':values['mean_kl']-values['mean_zero'],
                'interaction':values['noise_kl']-values['mean_kl']-values['noise_zero']+values['mean_zero']})
    # All models use equal data; logs can be compared within each source family.
    fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    metrics=[('Noise / movement','noise_to_move_ratio'),('Normalized MSE','seen_normalized_mse'),('Posterior std','posterior_std_mean')]
    for i,family in enumerate(['winner','zero']):
        for arm in ['mean_zero','noise_zero','mean_kl','noise_kl','adaptive_kl']:
            rs=groups[family,arm]
            for ax,(title,key) in zip(axes[i],metrics):
                v=np.array([[b[key] if key in b else b['bottleneck'][key] for b in r['result']['blocks']] for r in rs])
                ax.plot(np.arange(1,21)*5,v.mean(0),label=LABELS[arm],color=COLORS[arm]);ax.set(title=f'{family}: {title}',xlabel='Replay transitions (thousands)',yscale='log');ax.grid(alpha=.2)
    h,l=axes[0,0].get_legend_handles_labels();fig.legend(h,l,loc='outside lower center',ncol=5,fontsize=8)
    fig.suptitle('Shared-data training trajectories; posterior std is NOT injected noise in mean/fixed-noise arms')
    fig.savefig(assets/'kl-mechanisms-trajectories.png',dpi=150);plt.close(fig)
    if online:
        og=defaultdict(list)
        for r in online:og[r['arm']].append(r)
        fig,axes=plt.subplots(2,3,figsize=(15,9),layout='constrained')
        sets=[['mean_zero','noise_zero','mean_kl','noise_kl','adaptive_zero','adaptive_kl'],['adaptive_kl','prior_lr3','prior_lr10','prior_steps5']]
        for i,aa in enumerate(sets):
            for k,seed in enumerate([9,10,11]):
                ax=axes[i,k]
                for arm in aa:
                    r=next(r for r in og[arm] if r['seed']==seed);pr=[json.loads(x) for x in (Path(r['path'])/'progress.jsonl').read_text().splitlines()]
                    ax.plot([v['interactions']/1000 for v in pr],[100*v['coverage'] for v in pr],label=LABELS[arm],color=COLORS[arm])
                ax.set(title=f'Seed {seed} · '+('noise/KL ablations' if i==0 else 'prior adaptation'),xlabel='Real actions (thousands)',ylabel='Physical coverage (%)',ylim=(0,101));ax.grid(alpha=.2)
            axes[i,0].legend(fontsize=8)
        fig.suptitle('Online exploration: equal encoder/dynamics updates; repulsion α=.03 throughout')
        fig.savefig(assets/'kl-mechanisms-online.png',dpi=150);plt.close(fig)
        lines=['## Online, seeds9/10/11','', '| Arm | Mean coverage over run | Final coverage by seed | Full-coverage actions by seed |', '|---|---:|---|---|']
        for arm in arms:
            rs=sorted(og[arm],key=lambda r:r['seed'])
            lines.append(f'| {LABELS[arm]} | {100*np.mean([r["coverage_auc"] for r in rs]):.1f}% | '+' / '.join(f'{100*r["result"]["coverage"]:.1f}%' for r in rs)+' | '+' / '.join(str(r['result']['first_full']) if r['result']['first_full'] else '>100k' for r in rs)+' |')
        ft.append('\n'.join(lines))
        if not a.skip_animation:
            # Predeclared central contrast, all seeds; no best-seed selection.
            aa=['adaptive_kl','mean_zero','noise_zero','mean_kl','noise_kl'];fig,axes=plt.subplots(5,3,figsize=(10,16),layout='constrained');frames={};artists={}
            for i,arm in enumerate(aa):
                for k,seed in enumerate([9,10,11]):
                    r=next(r for r in og[arm] if r['seed']==seed);d=Path(r['path']);ax=axes[i,k];Maze(load_config(d/'config.json').maze).draw(ax)
                    frames[i,k]=np.load(d/'frames.npz');artists[i,k]=(ax.scatter([],[],s=3,color=COLORS[arm],alpha=.8),ax.scatter([],[],s=20,c='red',zorder=6),ax.set_title('',fontsize=8));ax.set_xticks([]);ax.set_yticks([])
            def update(idx):
                for i,arm in enumerate(aa):
                    for k,seed in enumerate([9,10,11]):
                        f=frames[i,k];dots,here,title=artists[i,k];dots.set_offsets(f['points'][f['visited'][idx]]);here.set_offsets(f['position'][idx:idx+1]);title.set_text(f'{LABELS[arm]} · seed {seed}\n{f["visited"][idx].sum()}/1784 observed')
                fig.suptitle(f'{frames[0,0]["interactions"][idx]:,} actions · deterministic means in all queues\nRed = simulator-restored expansion; α=.03 in all arms')
            n=len(frames[0,0]['interactions']);anim=FuncAnimation(fig,update,frames=np.unique(np.linspace(0,n-1,100).astype(int)),interval=150)
            anim.save(assets/'kl-mechanisms.gif',writer=PillowWriter(fps=7),dpi=90);update(n-1);fig.savefig(assets/'kl-mechanisms-final.png',dpi=140);plt.close(fig)
    confirmation=read(a.confirmation) if a.confirmation else []
    if confirmation:
        fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
        lines=['## Fresh confirmation, seeds12/13/14','', '| Arm | Mean coverage over run | Final coverage by seed | Full-coverage actions by seed |', '|---|---:|---|---|']
        for arm in ['mean_zero','mean_kl','adaptive_kl']:
            rs=sorted([r for r in confirmation if r['arm']==arm],key=lambda r:r['seed'])
            lines.append(f'| {LABELS[arm]} | {100*np.mean([r["coverage_auc"] for r in rs]):.1f}% | '+' / '.join(f'{100*r["result"]["coverage"]:.1f}%' for r in rs)+' | '+' / '.join(str(r['result']['first_full']) if r['result']['first_full'] else '>100k' for r in rs)+' |')
            for ax,r in zip(axes,rs):
                pr=[json.loads(x) for x in (Path(r['path'])/'progress.jsonl').read_text().splitlines()]
                ax.plot([x['interactions']/1000 for x in pr],[100*x['coverage'] for x in pr],label=LABELS[arm],color=COLORS[arm]);ax.set(title=f'Seed {r["seed"]}',xlabel='Real actions (thousands)',ylabel='Coverage (%)',ylim=(0,101));ax.grid(alpha=.2)
        axes[0].legend(fontsize=8);fig.suptitle('Fresh confirmation of the no-sampling hypothesis; α=.03 throughout');fig.savefig(assets/'kl-mechanisms-confirm.png',dpi=150);plt.close(fig);ft.append('\n'.join(lines))
    Path('docs/kl-mechanisms-tables.md').write_text('# All KL mechanism results\n\nPrior mismatch is inactive in β=0 arms. In deterministic-input arms, q is an auxiliary distribution: its fitted KL does not bound information passed to dynamics through the noiseless means. See [the report](kl-mechanisms.md) for definitions and interpretation.\n\n'+'\n\n'.join(ft)+'\n')
    summary={'fixed':[{k:v for k,v in r.items() if k!='result'}|{'final':r['result']['blocks'][-1],'queue':r['result']['queue']} for r in fixed],
             'contrasts':contrasts,'online':[{k:v for k,v in r.items() if k!='result'}|{k:r['result'][k] for k in ['first_full','coverage','visited']}|{'final':r['result']['blocks'][-1]} for r in online],
             'confirmation':[{k:v for k,v in r.items() if k!='result'}|{k:r['result'][k] for k in ['first_full','coverage','visited']}|{'final':r['result']['blocks'][-1]} for r in confirmation]}
    (assets/'kl-mechanisms-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('\n\n'.join(ft))

if __name__=='__main__':main()
