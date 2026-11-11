"""Delayed penalties must reproduce corresponding controls through 50k actions."""
import argparse,json
from pathlib import Path
import torch,numpy as np
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();rows=[]
for stage in ['followup','delayed-noise']:
 root=Path(f'runs/compression-{stage}-20261111');m=json.loads((root/'manifest.json').read_text())
 for arm in m['arms']:
  spec=json.loads(Path(arm['experiment_spec']).read_text())
  if not (spec.get('l1_after') or spec.get('noise_after')):continue
  for seed in m['seeds']:
   path=root/f"{arm['name']}-seed{seed}";control=Path('runs/compression-regularizers-20261111')/f"{arm['objective']}-{'bound' if spec.get('noise_after') else 'base'}-seed{seed}"
   models=[torch.load(x/'checkpoint-10.pt',map_location='cpu',weights_only=True) for x in [path,control]]
   for module in ['model','dynamics']:
    for key in models[0][module]:assert torch.equal(models[0][module][key],models[1][module][key]),(path,module,key)
   replay=[np.load(x/'replay.npz') for x in [path,control]]
   for field in ['observations','actions','next_observations','positions','next_positions']:assert np.array_equal(replay[0][field][:50000],replay[1][field][:50000]),(path,field)
   rows.append({'run':str(path),'control':str(control),'exact_prefix_actions':50000,'encoder_and_dynamics_exact':True})
a.out.write_text(json.dumps(rows,indent=2)+'\n');print('exact delayed prefixes:',len(rows))
