"""Freeze candidates from tuning seeds only, before opening confirmation outcomes."""
import argparse,json
from pathlib import Path
from summarize_joint_loss import collect


def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 roots=[Path('runs')/f'joint-loss-{stage}-20261111' for stage in ['mix','scale','margin-extension']]
 groups={}
 for root in roots:
  report=collect(root)
  for run in report['runs']:
   if run['step_size']!=.125 or run['seed'] not in [42,43]:continue
   c=run['effective_coefficients'];key=(run['objective'],*c.values())
   groups.setdefault(key,{})[run['seed']]=run
 candidates=[]
 for group in groups.values():
  assert set(group)=={42,43}
  values=list(group.values());c=values[0]['effective_coefficients']
  candidates.append({'objective':values[0]['objective'],'coefficients':c,'mean_final':sum(r['coverage_k4'] for r in values)/2,
    'mean_auc':sum(r['auc'] for r in values)/2,'final':[r['coverage_k4'] for r in values],'sources':[r['run'] for r in values]})
 best=lambda obj:max([r for r in candidates if r['objective']==obj],key=lambda r:(r['mean_final'],r['mean_auc']))
 hybrid,pure=best('hybrid'),best('adjacency')
 selected=[('tuned-hybrid',hybrid),('tuned-adjacency',pure)]
 counterpart={**hybrid,'objective':'adjacency','coefficients':{**hybrid['coefficients'],'D':0}}
 baseline={'D':0,'A':.3,'R':.03,'L':.03,'m':1.}
 if counterpart['coefficients'] not in [pure['coefficients'],baseline]:selected.append(('matched-adjacency',counterpart))
 dynamics_counterpart={**hybrid,'objective':'dynamics','coefficients':{**hybrid['coefficients'],'A':0}}
 if dynamics_counterpart['coefficients']!={'D':1.,'A':0,'R':.03,'L':.03,'m':1.}:
  selected.append(('matched-dynamics',dynamics_counterpart))
 arms=[];layouts=[]
 for name,r in selected:
  c=r['coefficients'];params={'objective':r['objective'],'adjacency_weight':c['A'],'repulsion_weight':c['R'],'latent_l2':c['L'],'repulsion_margin':c['m']}
  if r['objective'] in {'hybrid','dynamics'}:params['dynamics_weight']=c['D']
  if r['objective']=='dynamics':params.pop('adjacency_weight')
  for label,step in [('normal',.25),('small',.125)]:arms.append({'name':f'{name}-{label}','step_size':step,**params})
  for layout in [13,23]:layouts.append({'name':f'{name}-layout{layout}','step_size':.125,'layout':layout,**params})
 a.out.mkdir(parents=True,exist_ok=False)
 for name,data in [('selection',{'hybrid':hybrid,'pure':pure,'selected':[n for n,_ in selected],'ranking':sorted(candidates,key=lambda r:(r['mean_final'],r['mean_auc']),reverse=True)}),('confirm-arms',arms),('layout-arms',layouts)]:
  (a.out/f'{name}.json').write_text(json.dumps(data,indent=2)+'\n')
 print(json.dumps({'hybrid':hybrid,'pure':pure,'selected':[n for n,_ in selected]},indent=2))
if __name__=='__main__':main()
