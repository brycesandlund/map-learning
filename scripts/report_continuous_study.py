"""Validation and ablation figures, accounting, and paired uncertainty summaries."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    reports=json.loads(a.results.read_text());by_stage={Path(r['root']).name:r for r in reports}
    confirm=by_stage['continuous-confirm-20261111'];layouts=by_stage['continuous-layouts-20261111'];ablation=by_stage['continuous-ablation-20261111']
    accounting={'runs':sum(len(r['runs']) for r in reports),'interactions':sum(x['interactions'] for r in reports for x in r['runs']),
                'updates':sum(x['updates'] for r in reports for x in r['runs']),
                'presentations_per_head':sum(x['samples_per_head'] for r in reports for x in r['runs'])}
    paired=[]
    for seed in confirm['manifest']['seeds']:
        runs={r['objective']:r for r in confirm['runs'] if r['seed']==seed}
        if all('k4_95' in r['milestones'] for r in runs.values()):
            paired.append(runs['adjacency']['milestones']['k4_95']-runs['dynamics']['milestones']['k4_95'])
    if len(paired)==len(confirm['manifest']['seeds']):
        d=np.array(paired);rng=np.random.default_rng(3911)
        boot=d[rng.integers(len(d),size=(20000,len(d)))].mean(1)
        accounting['fresh_seed_paired_adj_minus_dynamics_95']={'differences':paired,'mean':float(d.mean()),
            'bootstrap_interval_95':np.quantile(boot,[.025,.975]).tolist(),'adjacency_wins':int((d<0).sum()),
            'note':'Exploratory paired bootstrap over six seeds on one layout; not independent layout evidence.'}
    (Path(str(a.out)+'-accounting.json')).write_text(json.dumps(accounting,indent=2)+'\n')
    fig,axes=plt.subplots(1,3,figsize=(13.5,4.6))
    for ax,report,title in [(axes[0],confirm,'Original layout: 6 fresh seeds'),(axes[1],layouts,'3 new layouts × 3 seeds')]:
        names=[arm['name'] for arm in report['manifest']['arms']]
        for i,name in enumerate(names):
            group=[r for r in report['runs'] if r['name']==name];color='#386cb0' if 'dynamics' in name else '#ef8a25'
            times=np.array([r['milestones'].get('k4_95',r['interactions']) for r in group])/1000
            ax.scatter(i+np.linspace(-.1,.1,len(times)),times,color=color,s=30)
            ax.plot([i-.2,i+.2],[times.mean()]*2,color=color,lw=2)
        ax.set_xticks(range(len(names)),[n.replace('-layout','\nlayout ') for n in names],rotation=35,ha='right',fontsize=8)
        ax.set(title=title,ylabel='Actions to 95% of 0.25 bins (thousands)',ylim=(0,105));ax.grid(axis='y',alpha=.2)
    names=[arm['name'] for arm in ablation['manifest']['arms']]
    for i,name in enumerate(names):
        group=[r for r in ablation['runs'] if r['name']==name]
        values=np.array([r['coverage_k4'] for r in group])*100
        color='#386cb0' if name.startswith('dynamics') else '#ef8a25' if name.startswith('adjacency') else '#777777'
        axes[2].scatter(i+np.linspace(-.12,.12,len(values)),values,color=color,s=30)
        axes[2].plot([i-.2,i+.2],[values.mean()]*2,color=color,lw=2)
    axes[2].set_xticks(range(len(names)),[n.replace('-','\n',1) for n in names],rotation=35,ha='right',fontsize=8)
    axes[2].set(title='Ablations: 3 paired seeds each',ylabel='0.25-bin occupancy at 100k (%)',ylim=(0,105));axes[2].grid(axis='y',alpha=.2)
    fig.suptitle('Continuous noisy frontier — frozen losses, broader validation')
    fig.tight_layout();fig.savefig(str(a.out)+'.png',dpi=160);plt.close(fig)
    # Show the effects hidden by saturated final coverage.
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    groups=[('clean','No observation noise'),('iid','Independent action directions'),('smallstep','Half-sized moves')]
    screen=by_stage['continuous-screen-20261111']
    for ax,(suffix,title) in zip(axes,groups):
        for source,names,style in [(screen,['dynamics','adjacency'],'--'),(ablation,[f'dynamics-{suffix}',f'adjacency-{suffix}'],'-')]:
            for name in names:
                group=[r for r in source['runs'] if r['name']==name]
                traces=[[json.loads(line) for line in (Path(r['run'])/'progress.jsonl').read_text().splitlines()] for r in group]
                x=np.array([r['interactions'] for r in traces[0]])/1000;y=np.array([[r['coverage_k4']*100 for r in trace] for trace in traces])
                color='#386cb0' if name.startswith('dynamics') else '#ef8a25'
                ax.plot(x,y.mean(0),style,c=color,label=name)
                if style=='-':ax.fill_between(x,y.min(0),y.max(0),color=color,alpha=.12)
        ax.set(title=title,xlabel='Actions (thousands)',ylabel='0.25-bin occupancy (%)',ylim=(0,103));ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    fig.suptitle('Solid: intervention; dashed: original noisy stratified controls (same three seeds)')
    fig.tight_layout();fig.savefig(str(a.out)+'-ablations.png',dpi=160);plt.close(fig)
    print(json.dumps(accounting,indent=2))

if __name__=='__main__':main()
