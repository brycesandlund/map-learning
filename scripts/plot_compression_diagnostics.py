"""Coordinate use without averaging away per-seed permutations of sparse axes."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
rows=json.loads(a.results.read_text())['runs'];rows=[r for r in rows if r['checkpoint']=='checkpoint-20.pt'];summary=[]
fig,axes=plt.subplots(1,2,figsize=(12,4.5))
for ax,obj in zip(axes,['dynamics','adjacency']):
 for suffix in ['base','l1','noise','multistep']:
  method=obj+'-'+suffix;group=[r for r in rows if r['method']==method]
  if not group:continue
  ordered=np.array([np.sort(r['latent_rms_coordinates'])[::-1] for r in group])/.25
  ax.plot(range(1,9),ordered.mean(0),marker='o',label=suffix)
  ax.set(title=obj.capitalize(),xlabel='Coordinate rank by RMS (sorted within each seed)',ylabel='Coordinate RMS / repulsion margin',yscale='log');ax.grid(alpha=.2)
  keys=['centered_coordinate_participation','pca_participation','centroid_pca_participation','noise_fraction','relative_small_activation_fraction','relative_small_coordinate_fraction','noise_to_open_transition_rms','centroid_distance_spearman_xy']
  summary.append({'method':method,'n':len(group),**{k:float(np.mean([r[k] for r in group])) for k in keys}})
 ax.legend()
fig.suptitle('Common fresh-noise probes at 100k: coordinate use, not a PCA projection')
fig.tight_layout();fig.savefig(str(a.out)+'.png',dpi=170);plt.close(fig)
Path(str(a.out)+'.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
