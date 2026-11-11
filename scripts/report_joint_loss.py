"""Static, all-seed figures for combined-objective sweeps."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
 p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 reports=json.loads(a.results.read_text());fig,axes=plt.subplots(1,len(reports),figsize=(max(8,len(reports)*5),7),squeeze=False)
 for ax,report in zip(axes[0],reports):
  groups=sorted(report['groups'],key=lambda g:(g['mean_final'],g['mean_auc']))
  labels=[]
  for i,g in enumerate(groups):
   name=g['arm']['name'];labels.append(name.replace('repulsion_margin','m').replace('repulsion_weight','R').replace('latent_l2','L').replace('adjacency_weight','A').replace('dynamics_weight','D'))
   coeff=next(r['effective_coefficients'] for r in report['runs'] if r['name']==name)
   color='#cb6d67' if coeff['L']>=2*coeff['R'] else '#5990bf'
   ax.barh(i,g['mean_final']*100,color=color,alpha=.65)
   ax.scatter(np.array(g['final'])*100,np.full(len(g['final']),i),color='#242f40',s=13,zorder=3)
  ax.set_yticks(range(len(groups)),labels,fontsize=7);ax.set(xlabel='Quarter-bin occupancy at 100k (%)',xlim=(0,103),title=Path(report['root']).name.replace('joint-loss-','').replace('-20261111',''))
  ax.grid(axis='x',alpha=.2)
 fig.suptitle('All tuning settings: means and seeds; red = L ≥ 2R (collapse-optimal regime)')
 fig.tight_layout();fig.savefig(str(a.out)+'.png',dpi=170);plt.close(fig)
 # Exact hyperparameters alongside outcomes, suitable for machine auditing.
 rows=[]
 for report in reports:
  for g in report['groups']:
   arm=g['arm'];rows.append({'stage':report['root'],**g})
 Path(str(a.out)+'-table.json').write_text(json.dumps(rows,indent=2)+'\n')

if __name__=='__main__':main()
