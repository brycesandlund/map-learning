"""Freeze best intervention per objective/family, with matched ablation controls."""
import argparse,json
from pathlib import Path
from summarize_joint_loss import collect
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--families',nargs='+',default=['l1','noise','multistep']);p.add_argument('--stages',nargs='+',default=['regularizers','multistep','followup','delayed-noise','weak-noise']);p.add_argument('--skip-confirmation-arms',type=Path);a=p.parse_args()
reports=[collect(Path(f'runs/compression-{stage}-20261111')) for stage in a.stages]
selected=[];ranking=[]
for obj in ['dynamics','adjacency']:
 for family in a.families:
  options=[]
  for report in reports:
   for g in report['groups']:
    arm=g['arm'];spec=json.loads(Path(arm['experiment_spec']).read_text()) if 'experiment_spec' in arm else {}
    eligible=(spec.get('latent_l1',0)>0 if family=='l1' else spec.get('noise_std',0)>0 if family=='noise' else spec.get('multistep_weight',0)>0 and spec.get('horizon',1)>1)
    if arm['objective']==obj and eligible:options.append({'family':family,'arm':arm,'spec':spec,'mean_final':g['mean_final'],'mean_auc':g['mean_auc'],'final':g['final']})
  best=max(options,key=lambda r:(r['mean_final'],r['mean_auc']));selected.append(best);ranking+=options
# Keep actual coefficients; matched controls turn off only the selected intervention.
a.out.mkdir(parents=True,exist_ok=False)
confirm=[];layouts=[];seen={}
base=lambda obj:{'objective':obj,'step_size':.125,'repulsion_margin':.25}
for obj in ['dynamics','adjacency']:seen[json.dumps(base(obj),sort_keys=True)]=obj+'-base'
def add(row,name,spec):
 arm={k:v for k,v in row['arm'].items() if k not in ['name','experiment_spec']}
 effective={k:v for k,v in spec.items() if v not in [0,False] or k=='noise_repulsion'}
 # Remove schedule metadata when L1 is off; remove noise controls when noise is off.
 if not spec.get('latent_l1',0):
  for k in ['latent_l1','l1_after','l1_ramp']:effective.pop(k,None)
 if not spec.get('noise_std',0):
  for k in ['noise_std','noise_repulsion','noise_after','noise_ramp']:effective.pop(k,None)
 signature=json.dumps({'arm':arm,'spec':effective},sort_keys=True) if effective else json.dumps(arm,sort_keys=True)
 if signature in seen:return seen[signature]
 seen[signature]=name
 if effective:
  sp=a.out/f'{name}.json';sp.write_text(json.dumps(effective,indent=2)+'\n');arm['experiment_spec']=str(sp)
 confirm.append({'name':name,**arm});return name
for row in selected:
 obj=row['arm']['objective'];family=row['family'];name=obj+'-'+family
 row['confirmation_name']=add(row,name,row['spec'])
 control=dict(row['spec'])
 if family=='l1':control['latent_l1']=0
 elif family=='noise':control['noise_std']=0
 else:control['horizon']=1
 row['matched_control']=add(row,obj+'-'+family+'-control',control)
# Layout candidates: best intervention per objective, frozen only from tuning.
layout_names=[]
for obj in ['dynamics','adjacency']:
 winner=max([r for r in selected if r['arm']['objective']==obj],key=lambda r:(r['mean_final'],r['mean_auc']))
 layout_names += [winner['confirmation_name'],winner['matched_control']]
for arm in confirm:
 if arm['name'] in layout_names:
  for layout in [13,23]:layouts.append({**arm,'name':arm['name']+'-layout'+str(layout),'layout':layout})
if a.skip_confirmation_arms:
 prior={r['name']:r for r in json.loads(a.skip_confirmation_arms.read_text())}
 def canonical(arm):
  value={k:v for k,v in arm.items() if k!='experiment_spec'}
  value['spec']=json.loads(Path(arm['experiment_spec']).read_text()) if 'experiment_spec' in arm else {}
  return value
 for arm in confirm:
  if arm['name'] in prior:assert canonical(arm)==canonical(prior[arm['name']]),'Previously frozen candidate changed'
 confirm=[arm for arm in confirm if arm['name'] not in prior]
for name,data in [('selection',{'selected':selected,'ranking':ranking,'layout_names':layout_names}),('confirm-arms',confirm),('layout-arms',layouts)]:
 (a.out/f'{name}.json').write_text(json.dumps(data,indent=2)+'\n')
print(json.dumps({'selected':selected,'confirm_arms':len(confirm),'layout_arms':len(layouts)},indent=2))
