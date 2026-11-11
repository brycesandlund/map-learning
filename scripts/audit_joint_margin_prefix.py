"""Compare paired margin-1 and margin-.25 runs without modifying checkpoints."""
import json
from pathlib import Path
import numpy as np
import torch
rows=[]
for method in ['dynamics','adjacency']:
 for seed in range(60,66):
  base=Path('runs/joint-loss-confirm-controls-20261111')/f'{method}-small-seed{seed}'
  changed=Path('runs/joint-loss-confirm-selected-20261111')/f'matched-{method}-small-seed{seed}'
  states=[]
  for checkpoint in [0,1,5,10,20]:
   data=[torch.load(p/f'checkpoint-{checkpoint}.pt',map_location='cpu',weights_only=True) for p in [base,changed]]
   equal={module:all(torch.equal(data[0][module][key],data[1][module][key]) for key in data[0][module]) for module in ['model','dynamics']}
   states.append({'actions':checkpoint*5000,**equal})
  logs=[[json.loads(s) for s in (p/'training.jsonl').read_text().splitlines()] for p in [base,changed]]
  norm_diff=next((x['interactions'] for x,y in zip(*logs) if x['latent_squared_norm']!=y['latent_squared_norm']),None)
  first_inactive=[next((r['interactions'] for r in log if r['repulsion_active_fraction']<1),None) for log in logs]
  replay=[np.load(p/'replay.npz') for p in [base,changed]]
  differences=np.flatnonzero(np.any(replay[0]['positions']!=replay[1]['positions'],axis=1)|np.any(replay[0]['next_positions']!=replay[1]['next_positions'],axis=1))
  rows.append({'method':method,'seed':seed,'checkpoints_exact':states,'first_logged_norm_difference':norm_diff,'first_logged_inactive_hinge_pair_base_changed':first_inactive,'first_different_transition':int(differences[0]+1) if len(differences) else None})
out={'note':'Only saved checkpoints and last batch of each 500-action training event are logged. First logged inactive hinge pair does not identify the very first such minibatch. As long as all sampled squared distances remain below both cutoffs, changing m only adds a constant to the repulsion loss, leaving its gradients identical.','runs':rows}
Path('docs/assets/joint-loss-margin-prefix.json').write_text(json.dumps(out,indent=2)+'\n')
for r in rows:print(r['method'],r['seed'],r['checkpoints_exact'],'norm diff',r['first_logged_norm_difference'],'inactive',r['first_logged_inactive_hinge_pair_base_changed'],'replay diff',r['first_different_transition'])
