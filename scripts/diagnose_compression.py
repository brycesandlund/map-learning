"""Common physical probes of noisy encoders. Evaluation only; replays differ."""
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
    latent=arrays['latents'];centroids=arrays['centroids'];by_position=latent.reshape(len(centroids),16,-1)
    within=np.mean((by_position-centroids[:,None,:])**2,axis=(0,1))
    between=np.var(centroids,axis=0);spectrum=np.linalg.eigvalsh(np.cov(latent,rowvar=False)).clip(0)
    clean_spectrum=np.linalg.eigvalsh(np.cov(centroids,rowvar=False)).clip(0)
    rms=np.sqrt(np.mean(latent**2,axis=0));m=run['effective_coefficients']['m']
    pr=lambda x:float(x.sum()**2/max(np.sum(x**2),1e-30))
    row.update(method=run['name'],seed=run['seed'],latent_rms_coordinates=rms.tolist(),
      centered_coordinate_participation=pr(np.var(latent,axis=0)),pca_participation=pr(spectrum),
      centroid_pca_participation=pr(clean_spectrum),relative_small_coordinate_fraction=float(np.mean(rms<.01*m)),
      relative_small_activation_fraction=float(np.mean(abs(latent)<.01*m)),
      noise_fraction=float(within.sum()/max(within.sum()+between.sum(),1e-30)))
    rows.append(row)
a.out.write_text(json.dumps({'note':'Off-policy common physical/noise probes. Different online replays; no causal objective-only attribution. Threshold metrics are evaluation only and do not enter priorities.','runs':rows},indent=2)+'\n')
print('audited',len(rows),'checkpoints')
