"""Paired effects, conditional on layout; retain all declared controls."""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('results',type=Path,nargs='+');p.add_argument('--selection',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
runs=[r for f in a.results for report in json.loads(f.read_text()) for r in report['runs']];selection=json.loads(a.selection.read_text());rng=np.random.default_rng(553871);rows=[]
def compare(left,right,reason):
 l={r['seed']:r for r in runs if r['name']==left};r={x['seed']:x for x in runs if x['name']==right}
 if not l or set(l)!=set(r):return
 for metric in ['coverage_k4','auc_k4','capped95']:
  val=lambda x:x['milestones'].get('k4_95',x['interactions']) if metric=='capped95' else x[metric]*100
  seeds=sorted(l);d=np.array([val(l[s])-val(r[s]) for s in seeds]);boot=d[rng.integers(len(d),size=(20000,len(d)))].mean(1)
  rows.append(dict(candidate=left,control=right,reason=reason,metric=metric,seeds=seeds,differences=d.tolist(),mean_difference=float(d.mean()),wins=int((d<0 if metric=='capped95' else d>0).sum()),n=len(d),exploratory_bootstrap95=np.quantile(boot,[.025,.975]).tolist()))
for step in [.125,.25]:
 choices=[s for s in selection if s['arm']['step_size']==step]
 walk=next(s['arm']['name'] for s in choices if s['arm']['objective']=='walk' and s['arm'].get('reset_every',0)==0)
 for obj in ['dynamics','adjacency']:
  hybrid_arm=next(s['arm'] for s in choices if s['arm']['objective']==obj)
  hybrid=hybrid_arm['name']
  continuation=f'continue-s{step}-warm{hybrid_arm["walk_steps"]}-r{hybrid_arm["reset_every"]}' if hybrid_arm['reset_every'] else walk
  for layout in sorted({r['layout'] for r in runs}):
   suffix='' if layout==7 else f'-layout{layout}'
   compare(walk+suffix,f'{obj}-base-s{step}'+suffix,'Selected walk versus learned baseline')
   compare(hybrid+suffix,f'{obj}-base-s{step}'+suffix,'Hybrid versus learned baseline')
   compare(hybrid+suffix,walk+suffix,'Hybrid versus selected no-reset walk')
   if continuation!=walk:compare(hybrid+suffix,continuation+suffix,'Hybrid versus exact-prefix continuing walk')
 # matched setting pairs: reset intervention and observed-collision response
 groups={}
 for r in runs:
  if r['objective']=='walk' and r['step_size']==step and r['layout']==7:groups[r['name']]=r
 names=list(groups)
 for i,left in enumerate(names):
  for right in names[i+1:]:
   x,y=groups[left],groups[right]
   if all(x[k]==y[k] for k in ['persistence','stay_probability','turn_on_block']) and (x['reset_every']==0)!=(y['reset_every']==0):
    compare(left if x['reset_every'] else right,right if x['reset_every'] else left,'Matched policy: resets versus none')
   if all(x[k]==y[k] for k in ['persistence','stay_probability','reset_every']) and x['turn_on_block']!=y['turn_on_block']:
    compare(left if x['turn_on_block'] else right,right if x['turn_on_block'] else left,'Matched policy: collision response versus open loop')
a.out.write_text(json.dumps({'note':'Paired seed bootstrap, conditional on layout. No multiplicity correction; repeated model seeds across layouts. Capped time-to95 treats failures as100k, not observed completion.','comparisons':rows},indent=2)+'\n')
for r in rows:
 if r['metric']=='auc_k4':print(r['candidate'],'vs',r['control'],'AUCdelta',round(r['mean_difference'],2),'wins',r['wins'],'/',r['n'])
