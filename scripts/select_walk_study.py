"""Selection uses tuning seeds only; immutable derived manifests."""
import argparse,json,math
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--phase',choices=['hybrids','walk-confirmation','confirmation'],required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
def load(root):
 root=Path(root);m=json.loads((root/'manifest.json').read_text());assert json.loads((root/'complete.json').read_text())['failed']==0
 rows=[]
 for arm in m['arms']:
  results=[]
  for seed in m['seeds']:
   d=root/f"{arm['name']}-seed{seed}";r=json.loads((d/'result.json').read_text());logs=[json.loads(s) for s in (d/'progress.jsonl').read_text().splitlines()]
   results.append((np.trapezoid(np.r_[0,[x['coverage_k4'] for x in logs]],np.r_[0,[x['interactions'] for x in logs]])/r['interactions'],r['coverage_k4']))
  rows.append(dict(arm=arm,auc=float(np.mean([x[0] for x in results])),final=float(np.mean([x[1] for x in results])),seeds=results))
 return rows
def best(rows):return max(rows,key=lambda r:(r['auc'],r['final']))
walks=load('runs/walk-screen-v2-20261111');a.out.mkdir(exist_ok=False)
selected=[];arms=[]
for step in [.125,.25]:
 pool=[x for x in walks if x['arm']['step_size']==step]
 noreset=best([x for x in pool if x['arm'].get('reset_every',0)==0]);reset=best([x for x in pool if x['arm'].get('reset_every',0)>0]);selected.extend([noreset,reset])
 policy={k:v for k,v in noreset['arm'].items() if k not in ['name','objective','reset_every']}
 if a.phase=='hybrids':
  for obj in ['dynamics','adjacency']:
   for warm in [1000,5000,20000]:
    for resets in [0,2]:
     arms.append(dict(**policy,name=f'{obj}-s{step}-warm{warm}-resets{resets}',objective=obj,repulsion_margin=.25,walk_steps=warm,reset_every=math.ceil(warm/3) if resets else 0))
 else:
  arms.extend([dict(**noreset['arm']),dict(**reset['arm'])])
  iid=next(x['arm'] for x in pool if x['arm']['persistence']==0 and x['arm'].get('reset_every',0)==0)
  if iid not in arms:arms.append(iid)
  arms.append(dict(name=f'random-queue-s{step}',objective='random_queue',step_size=step,repulsion_margin=.25))
  def policy_key(arm):
   return (arm['step_size'],arm.get('persistence',0),arm.get('reset_every',0),arm.get('stay_probability',.2),arm.get('turn_on_block',1))
  # Matched reset/no-reset controls, fixed from tuning choices only.
  for source,interval,label in [(reset['arm'],0,'reset-policy-noreset'),(noreset['arm'],reset['arm']['reset_every'],'noreset-policy-reset')]:
   candidate={**source,'name':f'{label}-s{step}','reset_every':interval}
   if not any(x['objective']=='walk' and policy_key(x)==policy_key(candidate) for x in arms):arms.append(candidate)
  openloop=best([x for x in pool if x['arm'].get('turn_on_block',1)==0])['arm']
  for source in [openloop,{**openloop,'name':f'openloop-policy-blockturn-s{step}','turn_on_block':1}]:
   if not any(x['objective']=='walk' and policy_key(x)==policy_key(source) for x in arms):arms.append(source)
if a.phase=='confirmation':
 hybrids=load('runs/walk-hybrid-tuning-v2-20261111')
 for step in [.125,.25]:
  for obj in ['dynamics','adjacency']:
   choice=best([x for x in hybrids if x['arm']['step_size']==step and x['arm']['objective']==obj]);selected.append(choice);arms.append(choice['arm'])
   if choice['arm']['reset_every']:
    c=choice['arm'];control={k:v for k,v in c.items() if k not in ['name','objective','walk_steps','repulsion_margin']}
    control.update(name=f'continue-s{step}-warm{c["walk_steps"]}-r{c["reset_every"]}',objective='walk',reset_until=c['walk_steps'])
    if not any(x['name']==control['name'] for x in arms):arms.append(control)
 layouts=[]
 walk=best([x for x in walks if x['arm']['step_size']==.125])['arm']
 for layout in [13,23]:
  for arm in [walk]+[dict(name=f'{o}-base-s0.125',objective=o,step_size=.125,repulsion_margin=.25) for o in ['dynamics','adjacency']]+[x for x in arms if x['objective'] in ['dynamics','adjacency'] and x['step_size']==.125]:
   layouts.append(dict(**{**arm,'name':arm['name']+f'-layout{layout}','layout':layout}))
 (a.out/'layouts-arms.json').write_text(json.dumps(layouts,indent=2)+'\n')
(a.out/'arms.json').write_text(json.dumps(arms,indent=2)+'\n');(a.out/'selection.json').write_text(json.dumps(selected,indent=2)+'\n')
for r in selected:print(r['arm']['name'],'AUC',round(r['auc']*100,2),'final',round(r['final']*100,2))
print('new arms',len(arms))
