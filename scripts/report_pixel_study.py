import argparse,json,collections
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('roots',nargs='+',type=Path);p.add_argument('--prefix',type=Path,required=True);p.add_argument('--no-plot',action='store_true');p.add_argument('--qualified-names',action='store_true');a=p.parse_args()
groups=collections.defaultdict(list)
for root in a.roots:
 for path in root.glob('*/result.json'):
    r=json.loads(path.read_text());s=json.loads((path.parent/'spec.json').read_text());r.update(spec=s,path=str(path.parent))
    name=path.parent.name.rsplit('-seed',1)[0]
    if a.qualified_names:name=root.name+'/'+name
    groups[name].append(r)
summary=[]
for name,rs in groups.items():
 summary.append(dict(name=name,n=len(rs),steps=rs[0]['spec']['steps'],warm_steps=rs[0]['spec'].get('warm_steps',0),noise=rs[0]['spec']['noise'],noise_mode=rs[0]['spec'].get('noise_mode','full'),rgb=rs[0]['spec'].get('rgb',0),final=np.mean([r['coverage_k4'] for r in rs]),auc=np.mean([r['auc'] for r in rs]),fine=np.mean([r['coverage_k8'] for r in rs]),seeds=[dict(seed=r['spec']['seed'],final=r['coverage_k4'],auc=r['auc']) for r in rs]))
summary.sort(key=lambda r:-r['auc'])
a.prefix.with_suffix('.json').write_text(json.dumps(summary,indent=2))
lines=['| Arm | n | Actions | Warm start | Final coverage | Coverage AUC | Fine coverage |','|---|---:|---:|---:|---:|---:|---:|']
for r in summary:lines.append(f"| {r['name']} | {r['n']} | {r['steps']} | {r['warm_steps']} | {100*r['final']:.1f}% | {100*r['auc']:.1f}% | {100*r['fine']:.1f}% |")
a.prefix.with_suffix('.md').write_text('\n'.join(lines)+'\n')
if a.no_plot:raise SystemExit(0)
fig,ax=plt.subplots(figsize=(10,max(4,len(summary)*.32)))
for i,r in enumerate(summary):
 ax.barh(i,100*r['auc'],alpha=.45)
 ax.scatter([100*s['auc'] for s in r['seeds']],[i]*r['n'],s=15,color='black')
ax.set_yticks(range(len(summary)),[r['name'] for r in summary]);ax.invert_yaxis();ax.set_xlabel('Coverage AUC (%) — points are individual seeds');ax.set_xlim(0,100)
fig.tight_layout();fig.savefig(a.prefix.with_suffix('.png'),dpi=130);plt.close(fig)
