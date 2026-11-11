"""Paired held-out comparisons, retaining censored runs and all seeds."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LABELS={'dynamics':'Dynamics\nm=1','adjacency':'Adjacency\nm=1','mix':'Hybrid A=.03\nm=1',
        'tuned-hybrid':'Hybrid A=.3\nm=.25','tuned-adjacency':'Adjacency\nm=.125',
        'matched-adjacency':'Adjacency\nm=.25','matched-dynamics':'Dynamics\nm=.25'}
COLORS=dict(zip(LABELS,['#386cb0','#ef8a25','#269378','#9063ac','#c9404c','#a77b34','#38a4ad']))

def main():
 p=argparse.ArgumentParser();p.add_argument('results',type=Path,nargs='+');p.add_argument('--out',type=Path,required=True);p.add_argument('--layouts',action='store_true');a=p.parse_args()
 runs=[r for path in a.results for report in json.loads(path.read_text()) for r in report['runs']]
 for r in runs:r['method']=r['name'].rsplit('-',1)[0]
 methods=list(dict.fromkeys(r['method'] for r in runs));splits=[13,23] if a.layouts else [.25,.125]
 fig,axes=plt.subplots(1,2,figsize=(12,5));summary=[]
 for ax,split in zip(axes,splits):
  subset=[r for r in runs if (r['layout'] if a.layouts else r['step_size'])==split]
  for i,method in enumerate(methods):
   group=[r for r in subset if r['method']==method]
   values=np.array([r['coverage_k4']*100 if a.layouts or split==.125 else r['restricted95']/1000 for r in group])
   ax.scatter(i+np.linspace(-.12,.12,len(values)),values,s=30,color=COLORS.get(method))
   ax.plot([i-.22,i+.22],[values.mean()]*2,c='black',lw=1.5)
   summary.append({'split':split,'method':method,'n':len(group),'seeds':[r['seed'] for r in group],
     'mean_final':float(np.mean([r['coverage_k4'] for r in group])), 'final':[r['coverage_k4'] for r in group],
     'mean_fine':float(np.mean([r['coverage_k8'] for r in group])),
     'success95':sum('k4_95' in r['milestones'] for r in group),'times95':[r['milestones'].get('k4_95') for r in group],
     'mean_restricted95':float(np.mean([r['restricted95'] for r in group])),'mean_auc':float(np.mean([r['auc'] for r in group]))})
  ax.set_xticks(range(len(methods)),[LABELS.get(m,m) for m in methods],rotation=25,ha='right',fontsize=9);ax.set_ylim(0,103);ax.grid(axis='y',alpha=.2)
  ax.set_title(f'Layout {split} (max move .125)' if a.layouts else f'Maximum move {split}')
  ax.set_ylabel('Quarter-bin occupancy at 100k (%)' if a.layouts or split==.125 else 'Actions to 95% (thousands; cap 100k)')
 fig.suptitle('Fixed settings on held-out layouts' if a.layouts else 'Frozen candidates on six fresh seeds')
 fig.tight_layout();fig.savefig(str(a.out)+'.png',dpi=170);plt.close(fig)
 paired=[];rng=np.random.default_rng(4371)
 for split in splits:
  subset=[r for r in runs if (r['layout'] if a.layouts else r['step_size'])==split]
  base={r['seed']:r for r in subset if r['method']=='adjacency'}
  for method in methods:
   group={r['seed']:r for r in subset if r['method']==method}
   if method=='adjacency' or set(group)!=set(base):continue
   differences=np.array([(group[s]['coverage_k4']-base[s]['coverage_k4'])*100 if a.layouts or split==.125 else group[s]['restricted95']-base[s]['restricted95'] for s in sorted(base)])
   boot=differences[rng.integers(len(differences),size=(20000,len(differences)))].mean(1)
   paired.append({'split':split,'method':method,'differences':differences.tolist(),'mean':float(differences.mean()),
      'exploratory_paired_bootstrap_95':np.quantile(boot,[.025,.975]).tolist(),
      'units':'percentage points' if a.layouts or split==.125 else 'actions','note':'No multiple-comparison correction; seeds within one layout, not independent environments.'})
 Path(str(a.out)+'.json').write_text(json.dumps({'groups':summary,'paired':paired},indent=2)+'\n')
 if not a.layouts:
  fig,axes=plt.subplots(1,2,figsize=(12,4.5))
  for ax,split in zip(axes,splits):
   for i,method in enumerate(methods):
    group=[r for r in runs if r['step_size']==split and r['method']==method]
    logs=[[json.loads(line) for line in (Path(r['run'])/'progress.jsonl').read_text().splitlines()] for r in group]
    curves=np.array([[row['coverage_k4']*100 for row in log] for log in logs])
    x=np.array([row['interactions'] for row in logs[0]])/1000
    color=COLORS.get(method,f'C{i}')
    for curve in curves:ax.plot(x,curve,color=color,alpha=.10,lw=.7)
    ax.plot(x,curves.mean(0),color=color,label=LABELS.get(method,method).replace('\n','; '),lw=2)
   ax.set(title=f'Maximum move {split}',xlabel='Executed actions (thousands)',ylabel='Quarter-bin occupancy (%)',ylim=(0,102))
   ax.grid(alpha=.2)
  axes[0].legend(fontsize=8)
  fig.suptitle('Fresh seeds: means and individual trajectories')
  fig.tight_layout();fig.savefig(str(a.out)+'-curves.png',dpi=170);plt.close(fig)
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
