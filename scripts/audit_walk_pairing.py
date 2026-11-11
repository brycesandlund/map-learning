"""Cross-root initialization/noise pairing and pure-walk hybrid prefixes."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('roots',type=Path,nargs='+');p.add_argument('--out',type=Path,required=True);a=p.parse_args()
runs=[];pairs={};prefixes=[]
for root in a.roots:
 m=json.loads((root/'manifest.json').read_text());assert json.loads((root/'complete.json').read_text())['failed']==0
 for seed in m['seeds']:
  for arm in m['arms']:
   path=root/f"{arm['name']}-seed{seed}";r=json.loads((path/'result.json').read_text());r.update(run=str(path),name=arm['name']);data=np.load(path/'replay.npz')
   r['physical_restore_jumps']=int(np.any(data['positions'][1:]!=data['next_positions'][:-1],axis=1).sum())
   key=(seed,r['nuisance_dims']);noise=hashlib.sha256(data['next_observations'][:,2:].tobytes()).hexdigest()
   signature=(r['initial_parameters_sha256'],noise)
   if key in pairs:assert pairs[key]==signature,path
   else:pairs[key]=signature
   runs.append(r)
def policy(r):return tuple(r[k] for k in ['seed','layout','step_size','persistence','reset_every','stay_probability','turn_on_block'])
walks={policy(r):r for r in runs if r['objective']=='walk'}
for r in runs:
 if r['objective'] not in ['dynamics','adjacency'] or not r['walk_steps'] or policy(r) not in walks:continue
 ref=walks[policy(r)];prefix=r['walk_steps']
 assert not ref.get('reset_until',0) or ref['reset_until']>=prefix
 with np.load(Path(r['run'])/'replay.npz') as x,np.load(Path(ref['run'])/'replay.npz') as y:
  for key in ['observations','actions','next_observations','positions','next_positions']:assert np.array_equal(x[key][:prefix],y[key][:prefix]),(r['run'],key)
 prefixes.append(dict(hybrid=r['run'],walk=ref['run'],exact_actions=prefix))
totals={k:sum(r[k] for r in runs) for k in ['interactions','updates','samples_per_head','reset_count','physical_restore_jumps']}
a.out.write_text(json.dumps(dict(runs=len(runs),totals=totals,initial_weights_and_successor_noise_matched=True,exact_walk_prefixes=prefixes,per_run=[{k:r[k] for k in ['run','objective','interactions','updates','samples_per_head','reset_count','physical_restore_jumps']} for r in runs]),indent=2)+'\n');print(len(runs),totals,'exact prefixes',len(prefixes))
