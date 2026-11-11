"""Audit coverage, initialization and budget for a completed (or partial) study."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def summarize(root):
    manifest=json.loads((root/'manifest.json').read_text());rows=[]
    for arm in manifest['arms']:
        for seed in manifest['seeds']:
            path=root/f"{arm['name']}-seed{seed}"
            if not (path/'result.json').exists():continue
            r=json.loads((path/'result.json').read_text())
            prog=[json.loads(x) for x in (path/'progress.jsonl').read_text().splitlines()]
            assert r['interactions']==manifest['blocks']*5000
            expected_updates=sum(int(np.ceil(5000*k/256)) for k in range(1,manifest['blocks']+1))
            expected_samples=sum(5000*k for k in range(1,manifest['blocks']+1))
            assert (r['updates'],r['samples_per_head'])==(expected_updates,expected_samples)
            x=np.r_[0,[p['interactions'] for p in prog]];y=np.r_[1/r['reachable_states'],[p['coverage'] for p in prog]]
            auc=float(np.trapezoid(y,x)/x[-1])
            rows.append({'arm':arm['name'],'seed':seed,'path':str(path),'first_full':r['first_full'],
                         'coverage_auc':auc,'final_coverage':r['coverage'],'milestones':r['milestones'],
                         'initial_hash':r['initial_parameters_sha256'],'seconds':r['seconds'],
                         'final_diagnostics':r['blocks'][-1], 'interactions':x.tolist(),'coverage':y.tolist()})
    for seed in manifest['seeds']:
        hashes={r['initial_hash'] for r in rows if r['seed']==seed}
        assert len(hashes)<=1, 'Mismatched initialization across arms'
    groups=[]
    for arm in manifest['arms']:
        r=[x for x in rows if x['arm']==arm['name']]
        if r:
            groups.append({'arm':arm['name'],'n':len(r),'completed':sum(x['first_full'] is not None for x in r),
                           'first_full':[x['first_full'] for x in r], 'seeds':[x['seed'] for x in r],
                           'mean_coverage_auc':float(np.mean([x['coverage_auc'] for x in r])),
                           'mean_final_coverage':float(np.mean([x['final_coverage'] for x in r])),
                           'mean_noise_to_move':float(np.mean([x['final_diagnostics']['noise_to_move_ratio'] for x in r]))})
    return {'root':str(root),'manifest':manifest,'groups':groups,'rows':rows}


def plot(result,out):
    fig,axes=plt.subplots(1,2,figsize=(15,6),gridspec_kw={'width_ratios':[1.1,1]})
    groups=result['groups']; colors=plt.cm.tab20(np.linspace(0,1,len(groups)))
    for color,g in zip(colors,groups):
        rows=[r for r in result['rows'] if r['arm']==g['arm']]
        x=np.array(rows[0]['interactions'])/1000;y=np.array([r['coverage'] for r in rows])
        axes[0].plot(x,y.mean(0),label=g['arm'],color=color,lw=1.8)
        axes[0].fill_between(x,y.min(0),y.max(0),alpha=.08,color=color)
    axes[0].set(xlabel='Executed environment actions (thousands)',ylabel='Reachable positions visited',ylim=(0,1.02))
    axes[0].legend(fontsize=7,loc='lower right');axes[0].grid(alpha=.2)
    order=sorted(groups,key=lambda g:g['mean_coverage_auc'],reverse=True)
    for i,g in enumerate(order):
        vals=[1000 if v is None else v/1000 for v in g['first_full']]
        for j,v in enumerate(vals):
            axes[1].scatter(102 if v==1000 else v,i+(j-(len(vals)-1)/2)*.15,
                            marker='x' if v==1000 else 'o',color=f'C{j}',s=30)
    axes[1].set_yticks(range(len(order)),[g['arm'] for g in order]);axes[1].invert_yaxis()
    axes[1].set(xlabel='Actions to 100% coverage (thousands); × = censored',xlim=(0,106))
    axes[1].axvline(100,color='gray',ls=':',lw=1);axes[1].grid(axis='x',alpha=.2)
    fig.suptitle('Distance-objective study — identical search, initialization and training budgets\nCurves: seed mean/range; completion markers: individual seeds',fontsize=12)
    fig.tight_layout();fig.savefig(out,dpi=150);plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--out',type=Path,required=True);p.add_argument('--plot',type=Path);a=p.parse_args()
    result=summarize(a.root);a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n')
    if a.plot:plot(result,a.plot)
    for g in result['groups']:print(json.dumps(g),flush=True)

if __name__=='__main__':main()
