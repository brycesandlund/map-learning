"""Exploratory paired contrasts for frozen candidates; no retuning or hypothesis filtering."""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('results',type=Path,nargs='+');p.add_argument('--out',type=Path,required=True);a=p.parse_args()
runs=[r for source in a.results for report in json.loads(source.read_text()) for r in report['runs']]
for r in runs:r['method']=r['name'].rsplit('-',1)[0]
comparisons=[('mix','adjacency'),('mix','dynamics'),('tuned-hybrid','adjacency'),('tuned-hybrid','dynamics'),('tuned-hybrid','matched-adjacency'),('tuned-hybrid','matched-dynamics'),('tuned-hybrid','tuned-adjacency'),('tuned-adjacency','adjacency'),('matched-adjacency','adjacency'),('matched-dynamics','dynamics')]
rows=[];rng=np.random.default_rng(3912)
for layout,step in sorted({(r['layout'],r['step_size']) for r in runs}):
 for candidate,reference in comparisons:
  groups=[{r['seed']:r for r in runs if r['layout']==layout and r['step_size']==step and r['method']==m} for m in [candidate,reference]]
  if not groups[0] or set(groups[0])!=set(groups[1]):continue
  field='coverage_k4' if step==.125 else 'restricted95';sign=1 if step==.125 else -1;scale=100 if step==.125 else 1
  d=np.array([(groups[0][s][field]-groups[1][s][field])*scale for s in sorted(groups[0])])
  boot=d[rng.integers(len(d),size=(20000,len(d)))].mean(1)
  rows.append({'layout':layout,'step':step,'candidate':candidate,'reference':reference,'n':len(d),'seeds':sorted(groups[0]),'differences':d.tolist(),'mean_difference':float(d.mean()),'wins':int((d*sign>0).sum()),'bootstrap_95':np.quantile(boot,[.025,.975]).tolist(),'units':'coverage percentage points' if step==.125 else 'actions to95 (capped100k)'})
a.out.write_text(json.dumps({'caveat':'Exploratory seed bootstrap, conditional on each layout; no multiple-comparison correction. Positive coverage differences and negative action differences favor candidate. Repeated seeds across layouts are not independent environments.','comparisons':rows},indent=2)+'\n')
for r in rows:print(r['layout'],r['step'],r['candidate'],'vs',r['reference'],'delta',round(r['mean_difference'],2),'wins',r['wins'],'/',r['n'],'interval',np.round(r['bootstrap_95'],2).tolist())
