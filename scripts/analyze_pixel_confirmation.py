"""Matched-budget tables, paired effects and exact walk-prefix checks."""
import argparse,json,collections
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('roots',nargs='+',type=Path);p.add_argument('--prefix',type=Path,required=True);a=p.parse_args()
rows=[];groups=collections.defaultdict(list)
for root in a.roots:
 for path in root.glob('*/result.json'):
  s=json.loads((path.parent/'spec.json').read_text());r=json.loads(path.read_text())
  if s['policy']=='walk':
   method='Matched-prefix walk' if '-matched-' in path.parent.name else 'Random + resets' if s['reset_probability'] else 'Persistent random' if s['persistence'] else 'IID random'
  elif s['policy']=='learned' and s.get('augmentation',0)>0:method='Learned + augmentation'
  elif s['architecture']=='cnn' and s['policy']=='learned':method='CNN + warmup'
  else:method='Frozen + warmup' if s['policy']=='frozen' else 'Learned + warmup' if s['warm_steps'] else 'Learned cold'
  trajectory=dict(np.load(path.parent/'trajectory.npz'))
  physical_jumps=int(np.any(trajectory['positions'][trajectory['parents'][1:]]!=trajectory['positions'][:-1],axis=1).sum())
  row=dict(**r,spec=s,path=str(path.parent),method=method,family=s['family'],physical_restore_jumps=physical_jumps,random_reset_events=r['resets']);rows.append(row);groups[(s['family'],method)].append(row)
summary=[];table=['| Family | Method | n | Coverage mean ± SD | AUC | Reached95% |','|---|---|---:|---:|---:|---:|']
for (family,method),rs in sorted(groups.items()):
 final=np.array([r['coverage_k4'] for r in rs]);auc=np.array([r['auc'] for r in rs]);times=[r['milestones'].get('k4_95',100000) for r in rs]
 s=dict(family=family,method=method,n=len(rs),final_mean=float(final.mean()),final_sd=float(final.std(ddof=1)) if len(rs)>1 else 0.,auc_mean=float(auc.mean()),time95_capped_mean=float(np.mean(times)),success95=sum('k4_95' in r['milestones'] for r in rs),fine_mean=float(np.mean([r['coverage_k8'] for r in rs])))
 summary.append(s);table.append(f"| {family} | {method} | {len(rs)} | {100*s['final_mean']:.1f} ± {100*s['final_sd']:.1f}% | {100*s['auc_mean']:.1f}% | {s['success95']}/{len(rs)} |")
effects=[];prefix_checks=[]
for family in sorted(set(r['family'] for r in rows)):
 focal='Learned + augmentation' if groups[(family,'Learned + augmentation')] else 'Learned + warmup'
 learned={r['spec']['seed']:r for r in groups[(family,focal)]}
 for comparator in ['Persistent random','Random + resets','Matched-prefix walk','Frozen + warmup','Learned cold','CNN + warmup','Learned + warmup']:
  controls={r['spec']['seed']:r for r in groups[(family,comparator)]};seeds=sorted(learned.keys()&controls.keys())
  if not seeds or comparator==focal:continue
  delta=np.array([learned[s]['auc']-controls[s]['auc'] for s in seeds]);final=np.array([learned[s]['coverage_k4']-controls[s]['coverage_k4'] for s in seeds])
  effects.append(dict(family=family,focal_method=focal,comparator=comparator,n=len(seeds),auc_delta_mean=float(delta.mean()),auc_deltas=delta.tolist(),final_delta_mean=float(final.mean()),final_deltas=final.tolist(),auc_wins=int((delta>0).sum())))
  for seed in seeds:
   x,y=learned[seed],controls[seed]
   if y['spec']['persistence']==.8 and y['spec']['reset_probability']==0 and (y['spec']['policy']=='walk' or y['spec'].get('warm_steps')==5000):
    tx=dict(np.load(Path(x['path'])/'trajectory.npz'));ty=dict(np.load(Path(y['path'])/'trajectory.npz'))
    for key in ['positions','parents','actions']:assert np.array_equal(tx[key][:5001],ty[key][:5001]),(family,seed,key)
    ix=np.load(Path(x['path'])/'images.npy',mmap_mode='r');iy=np.load(Path(y['path'])/'images.npy',mmap_mode='r')
    assert np.array_equal(ix[:5001],iy[:5001]);prefix_checks.append(dict(family=family,seed=seed,comparator=comparator))
a.prefix.with_suffix('.json').write_text(json.dumps(dict(rows=rows,summary=summary,effects=effects,prefix_checks=prefix_checks),indent=2))
a.prefix.with_suffix('.md').write_text('\n'.join(table)+'\n')

details=['| Family | Method | Seed | Quarter-bin coverage | Eighth-bin coverage | AUC | Actions to95% | Physical restore jumps |','|---|---|---:|---:|---:|---:|---:|---:|']
for r in sorted(rows,key=lambda r:(r['family'],r['method'],r['spec']['seed'])):
 details.append(f"| {r['family']} | {r['method']} | {r['spec']['seed']} | {100*r['coverage_k4']:.2f}% | {100*r['coverage_k8']:.2f}% | {100*r['auc']:.2f}% | {r['milestones'].get('k4_95','Not reached')} | {r['physical_restore_jumps']:,} |")
details+=['','Paired differences are focal method minus comparator; points are percentage points. Wins count AUC improvements. Small seed counts are descriptive evidence, not a significance claim.','','| Family | Focal | Comparator | n | Final difference | AUC difference | AUC wins |','|---|---|---|---:|---:|---:|---:|']
for e in effects:
 details.append(f"| {e['family']} | {e['focal_method']} | {e['comparator']} | {e['n']} | {100*e['final_delta_mean']:+.2f} | {100*e['auc_delta_mean']:+.2f} | {e['auc_wins']}/{e['n']} |")
a.prefix.with_name(a.prefix.name+'-details.md').write_text('\n'.join(details)+'\n')

families=[f for f in ['maze','snake','comb','open','rooms'] if any(r['family']==f for r in rows)]
fig,axes=plt.subplots(1,len(families),figsize=(max(6,3.6*len(families)),4),sharey=True);axes=np.atleast_1d(axes)
colors={'Learned + augmentation':'#e759a5','CNN + warmup':'#b28c5a','Learned + warmup':'#ca3c25','Learned cold':'#d59c37','Frozen + warmup':'#999999','Persistent random':'#2466ad','Random + resets':'#43a076','IID random':'#8b66a6','Matched-prefix walk':'#21b7d1'}
for ax,family in zip(axes,families):
 for (fam,method),rs in groups.items():
  if fam!=family or not rs:continue
  traces=[]
  for r in rs:
   progress=[json.loads(line) for line in (Path(r['path'])/'progress.jsonl').read_text().splitlines()]
   x=np.array([0]+[v['interactions'] for v in progress]);y=np.array([1/2304]+[v['coverage_k4'] for v in progress])*100;traces.append(y)
  traces=np.array(traces);mean=traces.mean(0)
  ax.plot(x/1000,mean,label=method,color=colors[method]);ax.fill_between(x/1000,traces.min(0),traces.max(0),color=colors[method],alpha=.10)
 ax.set(title=family,xlabel='Actions (thousands)',ylim=(0,100));ax.grid(alpha=.2)
axes[0].set_ylabel('Quarter-room endpoint coverage (%)')
legend={}
for ax in axes:
 handles,labels=ax.get_legend_handles_labels()
 legend.update(zip(labels,handles))
fig.legend(list(legend.values()),list(legend),loc='lower center',ncol=min(5,len(legend)),fontsize=8)
fig.tight_layout(rect=(0,.12,1,1));fig.savefig(a.prefix.with_suffix('.png'),dpi=140)
print('\n'.join(table));print('Exact matched prefixes',len(prefix_checks))
