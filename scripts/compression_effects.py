"""Prespecified candidate-vs-base and candidate-vs-matched-control contrasts."""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('results',type=Path,nargs='+');p.add_argument('--selection',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
runs=[r for source in a.results for report in json.loads(source.read_text()) for r in report['runs']]
selection=json.loads(a.selection.read_text());rng=np.random.default_rng(91391);rows=[]
for choice in selection['selected']:
 for layout in sorted({r['layout'] for r in runs}):
  suffix='' if layout==7 else '-layout'+str(layout)
  name=choice['confirmation_name']+suffix
  for control in dict.fromkeys([choice['arm']['objective']+'-base',choice['matched_control']]):
   ref=control+suffix
   maps=[{r['seed']:r for r in runs if r['name']==n and r['layout']==layout} for n in [name,ref]]
   if not maps[0] or set(maps[0])!=set(maps[1]):continue
   seeds=sorted(maps[0]);d=np.array([(maps[0][s]['coverage_k4']-maps[1][s]['coverage_k4'])*100 for s in seeds])
   boot=d[rng.integers(len(d),size=(20000,len(d)))].mean(1)
   rows.append({'candidate':name,'control':ref,'layout':layout,'seeds':seeds,'difference_pp':d.tolist(),
    'mean_difference_pp':float(d.mean()),'wins':int((d>0).sum()),'n':len(d),'exploratory_bootstrap95':np.quantile(boot,[.025,.975]).tolist()})
a.out.write_text(json.dumps({'note':'Paired seed bootstrap conditional on layout, no multiple-comparison adjustment. Layouts share model seeds. All candidate/control comparisons retained.','comparisons':rows},indent=2)+'\n')
for r in rows:print(r['candidate'],'vs',r['control'],'delta',round(r['mean_difference_pp'],2),'wins',r['wins'],'/',r['n'],'interval',np.round(r['exploratory_bootstrap95'],2))
