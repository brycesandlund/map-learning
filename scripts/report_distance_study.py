"""Consolidated auditable accounting and validation plots for this distance study."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_distance_study import summarize

ROOTS=['distance-screen-20261110','distance-followup-20261110','distance-dynamics-controls-20261110',
       'distance-qrl-unregularized-20261110','distance-confirm-20261110',
       'distance-layout13-20261110','distance-layout23-20261110','distance-layout37-20261110']
FIXED=['distance-fixed-20261110','distance-tuned-fixed-20261110']
LABELS={'dynamics-l2':'Dynamics: original','dynamics-reg-half':'Dynamics: retuned',
        'adjacency-w0.3-l20.03':'Adjacent attraction','affinity-l20.03':'Student-t affinities'}


def main():
    stages={root:summarize(Path('runs')/root) for root in ROOTS}
    online=[row for study in stages.values() for row in study['rows']]
    fixed=[]
    for root in FIXED:
        manifest=json.loads((Path('runs')/root/'manifest.json').read_text())
        for job in manifest['jobs']:
            path=Path('runs')/root/f"{job['arm']['name']}-seed{job['seed']}"/'result.json'
            r=json.loads(path.read_text());assert r['updates']==4112 and r['samples_per_head']==1050000
            fixed.append({'path':str(path),**r})
    for root,study in stages.items():
        expected=len(study['manifest']['arms'])*len(study['manifest']['seeds'])
        assert len(study['rows'])==expected, f'Incomplete {root}'
        status=list((Path('runs')/root).glob('*.status.json'))
        assert len(status)==expected and all(json.loads(p.read_text())['returncode']==0 for p in status)
    confirm=stages['distance-confirm-20261110'];new_layout_rows=[r for key,s in stages.items() if 'layout' in key for r in s['rows']]
    summaries=[]
    for arm,label in LABELS.items():
        for setting,rows in [('original_maze',confirm['rows']),('new_layouts',new_layout_rows)]:
            selected=[r for r in rows if r['arm']==arm]
            if not selected:continue
            values=[r['first_full'] for r in selected]
            summaries.append({'arm':arm,'label':label,'setting':setting,'n':len(selected),'complete':sum(v is not None for v in values),
                              'completion_actions':values,'mean_completion_actions':float(np.mean(values)) if all(v is not None for v in values) else None,
                              'mean_coverage_auc':float(np.mean([r['coverage_auc'] for r in selected]))})
    count={'online_runs':len(online),'fixed_data_fits':len(fixed),'full_fits':len(online)+len(fixed),
           'main_training_interactions':sum(int(r['interactions'][-1]) for r in online),
           'gradient_updates':(len(online)+len(fixed))*4112,'transition_presentations_per_head':(len(online)+len(fixed))*1050000,
           'frozen_continuation_actions':len(fixed)*3*128*5,
           'additional_smoke_training_actions':2500,'additional_smoke_frozen_continuation_actions':1920,
           'additional_smoke_updates':30}
    result={'accounting':count,'validation':summaries,'stages':stages,'fixed_fits':fixed}
    out=Path('docs/assets/distance-study-results.json');out.write_text(json.dumps(result,indent=2)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(13,5.3))
    for i,(arm,label) in enumerate(LABELS.items()):
        rows=[r for r in confirm['rows'] if r['arm']==arm];y=np.array([r['coverage'] for r in rows]);x=np.array(rows[0]['interactions'])/1000
        axes[0].plot(x,y.mean(0),color=f'C{i}',label=label,lw=2)
        axes[0].fill_between(x,y.min(0),y.max(0),color=f'C{i}',alpha=.09)
        for group,layout in enumerate([7,13,23,37]):
            source=confirm['rows'] if layout==7 else stages[f'distance-layout{layout}-20261110']['rows']
            values=np.array([r['first_full']/1000 for r in source if r['arm']==arm])
            if not len(values):continue
            center=group+(i-1.5)*.15
            axes[1].scatter(center+np.linspace(-.025,.025,len(values)),values,color=f'C{i}',s=20,alpha=.55)
            axes[1].plot([center-.055,center+.055],[values.mean()]*2,color=f'C{i}',lw=3)
    axes[0].set(xlabel='Executed actions (thousands)',ylabel='Visited reachable positions',ylim=(0,1.02),title='Original maze: six fresh seeds')
    axes[0].legend(fontsize=9,loc='lower right');axes[0].grid(alpha=.2)
    axes[1].set_xticks(range(4),['7\n6 seeds','13\n3 seeds','23\n3 seeds','37\n3 seeds'])
    axes[1].set(xlabel='Maze layout',ylabel='Actions to full coverage (thousands)',title='Fixed settings on additional layouts',ylim=(25,85))
    axes[1].grid(axis='y',alpha=.2)
    fig.suptitle('Adjacent attraction explores reliably; its efficiency advantage is not consistent across layouts',fontsize=12)
    fig.text(.5,.015,'Left: means with seed ranges. Right: individual runs and mean bars. All displayed runs reach full coverage.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.05,1,.94]);fig.savefig('docs/assets/distance-validation.png',dpi=160);plt.close(fig)
    print(json.dumps(count));print(json.dumps(summaries))

if __name__=='__main__':main()
