"""Margin sweep and hinge saturation; tuning seeds only, without duplicate weighting."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
runs=[r for report in json.loads(a.results.read_text()) for r in report['runs']]
fig,axes=plt.subplots(1,3,figsize=(13,4))
for obj,col in [('hybrid','#9063ac'),('adjacency','#da8628')]:
 filtered={}
 for r in runs:
  c=r['effective_coefficients']
  if r['objective']==obj and r['step_size']==.125 and r['seed'] in [42,43] and c['A']==.3 and c['R']==.03 and c['L']==.03 and c['D']==(1 if obj=='hybrid' else 0):filtered[(c['m'],r['seed'])]=r
 margins=sorted({key[0] for key in filtered})
 for ax,metric in zip(axes,['coverage','norm','hinge']):
  means=[]
  for m in margins:
   group=[filtered[(m,s)] for s in [42,43]]
   values=[r['coverage_k4']*100 if metric=='coverage' else r['tail_metrics']['latent_squared_norm']/m**2 if metric=='norm' else r['tail_metrics']['repulsion_active_fraction']*100 for r in group]
   means.append(np.mean(values));ax.scatter([m]*len(values),values,c=col,s=15,alpha=.6)
  ax.plot(margins,means,color=col,label=obj,marker='o',markersize=4)
  ax.set_xscale('log',base=2);ax.set_xlabel('Distance margin m (hinge cutoff m²)');ax.grid(alpha=.2)
axes[0].set_ylabel('Quarter-bin occupancy at 100k (%)');axes[0].set_ylim(0,100);axes[0].legend()
axes[1].set_ylabel('Mean squared latent norm / m²')
axes[2].set_ylabel('Random pairs inside hinge (%)');axes[2].set_ylim(0,100)
fig.suptitle('Raw-margin tuning: fixed D/A/R/L; dots are seeds 42/43; end metrics use last 10 logged batches')
fig.tight_layout();fig.savefig(a.out,dpi=170)
