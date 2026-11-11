"""Fixed off-policy physical/noise probes of frozen small-step checkpoints."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from audit_continuous_representation import audit
p=argparse.ArgumentParser();p.add_argument('results',type=Path,nargs='+');p.add_argument('--out',type=Path,required=True);a=p.parse_args()
torch.set_num_threads(1);rows=[]
for source in a.results:
 for report in json.loads(source.read_text()):
  for run in report['runs']:
   if run['step_size']!=.125:continue
   for checkpoint in ['checkpoint-5.pt','checkpoint-20.pt']:
    row,arrays=audit(Path(run['run']),checkpoint)
    spectrum=np.linalg.eigvalsh(np.cov(arrays['centroids'],rowvar=False)).clip(0)
    row.update(method=run['name'].rsplit('-',1)[0],seed=run['seed'],margin=run['effective_coefficients']['m'],
      centroid_variance=float(spectrum.sum()),centroid_top2_variance_fraction=float(spectrum[-2:].sum()/spectrum.sum()))
    rows.append(row)
summary=[]
for method in dict.fromkeys(r['method'] for r in rows):
 for checkpoint in ['checkpoint-5.pt','checkpoint-20.pt']:
  group=[r for r in rows if r['method']==method and r['checkpoint']==checkpoint]
  keys=['noise_to_open_transition_rms','centroid_distance_spearman_xy','centroid_top2_variance_fraction']
  summary.append({'method':method,'checkpoint':checkpoint,'n':len(group),**{k:float(np.mean([r[k] for r in group])) for k in keys},
    'noise_rms_over_margin':float(np.mean([r['noise_rms']/r['margin'] for r in group]))})
a.out.write_text(json.dumps({'note':'Identical uniform physical probes and independent fresh nuisance draws. Off-policy; replay differs between runs. Not a causal objective-only or memorization comparison. No probe is used by training/search.','summary':summary,'runs':rows},indent=2)+'\n')
print(json.dumps(summary,indent=2))
