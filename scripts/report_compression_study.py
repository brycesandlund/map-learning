"""All-seed plots and machine-readable tables; preserves every failed arm."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('results',type=Path,nargs='+');p.add_argument('--out',type=Path,required=True);p.add_argument('--tuning',action='store_true');a=p.parse_args()
reports=[report for source in a.results for report in json.loads(source.read_text())]
runs=[r for report in reports for r in report['runs']];groups=[]
for name in dict.fromkeys(r['name'] for r in runs):
 group=[r for r in runs if r['name']==name]
 groups.append({'name':name,'objective':group[0]['objective'],'n':len(group),'seeds':[r['seed'] for r in group],
  'mean_final':float(np.mean([r['coverage_k4'] for r in group])),'final':[r['coverage_k4'] for r in group],
  'mean_fine':float(np.mean([r['coverage_k8'] for r in group])),'mean_auc':float(np.mean([r['auc'] for r in group])),
  'spec':group[0].get('compression_spec',{}),'coefficients':group[0]['effective_coefficients'],
  'mean_seconds':float(np.mean([r['seconds'] for r in group])),
  'mean_extra_edges':float(np.mean([r.get('path_edges',0) for r in group]))})
fig,axes=plt.subplots(1,2,figsize=(13,max(5,max(sum(g['objective']==o for g in groups) for o in ['dynamics','adjacency'])*.29+1.5)))
for ax,obj in zip(axes,['dynamics','adjacency']):
 subset=[g for g in groups if g['objective']==obj]
 if a.tuning:subset=sorted(subset,key=lambda r:r['mean_final'])
 labels=[]
 for i,g in enumerate(subset):
  spec=g['spec'];color='#777777'
  if spec.get('latent_l1',0):color='#299c78'
  elif spec.get('noise_std',0):color='#4985bf'
  elif spec.get('multistep_weight',0):color='#a065b4'
  ax.barh(i,g['mean_final']*100,color=color,alpha=.55)
  ax.scatter(np.array(g['final'])*100,np.full(g['n'],i)+np.linspace(-.12,.12,g['n']),color='#222222',s=14,zorder=3)
  labels.append(g['name'].removeprefix(obj+'-'))
 ax.set_yticks(range(len(labels)),labels,fontsize=8);ax.set_xlim(0,102);ax.grid(axis='x',alpha=.2)
 ax.set(title=obj.capitalize(),xlabel='Quarter-bin occupancy at 100k actions (%)')
fig.suptitle(('Tuning: all arms and seeds' if a.tuning else 'Frozen interventions and matched controls on fresh seeds')+'\nGreen: L1 • Blue: fixed noise • Purple: prediction • Grey: controls')
fig.tight_layout();fig.savefig(str(a.out)+'.png',dpi=170);plt.close(fig)
Path(str(a.out)+'.json').write_text(json.dumps({'groups':groups},indent=2)+'\n')
if not a.tuning:
 fig,axes=plt.subplots(1,2,figsize=(13,5))
 for ax,obj in zip(axes,['dynamics','adjacency']):
  for g in [g for g in groups if g['objective']==obj]:
   curves=[]
   for run in [r for r in runs if r['name']==g['name']]:
    rows=[json.loads(s) for s in (Path(run['run'])/'progress.jsonl').read_text().splitlines()]
    curves.append([r['coverage_k4']*100 for r in rows])
   x=np.array([r['interactions'] for r in rows])/1000
   ax.plot(x,np.mean(curves,0),label=g['name'].removeprefix(obj+'-'))
  ax.set(title=obj.capitalize(),xlabel='Actions (thousands)',ylabel='Quarter-bin occupancy (%)',ylim=(0,102));ax.grid(alpha=.2);ax.legend(fontsize=7)
 fig.tight_layout();fig.savefig(str(a.out)+'-curves.png',dpi=170);plt.close(fig)
print(json.dumps(groups,indent=2))
