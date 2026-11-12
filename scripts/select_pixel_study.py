"""Freeze configurations from tuning-only completed runs; never inspect holdout."""
import argparse,json,collections,time
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--roots',nargs='+',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--steps',type=int,default=50000);a=p.parse_args()
groups=collections.defaultdict(list)
for root in a.roots:
 for path in root.glob('*/result.json'):
    s=json.loads((path.parent/'spec.json').read_text())
    if s['seed'] not in [110,111]:raise ValueError('Selection must use tuning seeds only')
    r=json.loads(path.read_text());groups[(str(root),path.parent.name.rsplit('-seed',1)[0])].append((s,r))
candidates=[(key,rs) for key,rs in groups.items() if len(rs)==2 and all(x[0]['steps']==30000 for x in rs)]
if not candidates:raise ValueError('No complete matched30k two-seed candidates')
key,rs=max(candidates,key=lambda kv:np.mean([r['auc'] for s,r in kv[1]]))
s=dict(rs[0][0]);s.pop('seed');s.pop('out');s['steps']=a.steps;s['name']='selected'
a.out.parent.mkdir(parents=True,exist_ok=True)
a.out.write_text(json.dumps(dict(frozen_unix=time.time(),source=key,mean_tuning_auc=np.mean([r['auc'] for _,r in rs]),arm=s),indent=2))
print(json.dumps(s))
