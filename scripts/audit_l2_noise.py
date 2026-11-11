"""Run-budget, paired-noise and priority checks for latent L2 experiments."""
import argparse,json
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from scipy.spatial import cKDTree
from map_learning.training import Trainer,seed_all
from audit_frontier_efficiency import reconstruct,encode
from kl_mechanisms import digest_model


def audit(root):
 rows=[];jobs=json.load(open(root/'manifest.json'))['jobs'];byseed=defaultdict(list)
 for j in jobs:byseed[j['seed']].append(j)
 for seed,jj in byseed.items():
  ref=Path(jj[0]['path']);initial=Trainer.load(ref/'checkpoint-0.pt');data0=np.load(ref/'replay.npz')
  for j in jj:
   p=Path(j['path']);r=json.load(open(p/'result.json'));t=Trainer.load(p/'checkpoint.pt');init=Trainer.load(p/'checkpoint-0.pt')
   assert r['updates']==4112 and r['samples_per_head']==1050000 and r['interactions']==100000
   assert t.spec['latent_l2']==j['latent_l2'] and t.spec['beta']==j['beta'] and t.spec['dynamics_input']=='mean' and t.config.train.repulsion_weight==.03
   assert digest_model(init)==digest_model(initial)
   with np.load(p/'replay.npz') as data:
    for key in ['observations','actions','next_observations','positions','next_positions']:np.testing.assert_array_equal(data[key][:500],data0[key][:500])
    np.testing.assert_array_equal(data['next_observations'][:,2:],data0['next_observations'][:,2:])
    obs,pos,vo,vp=reconstruct(data,100000);z=encode(t,obs);v=encode(t,vo);scores=cKDTree(v).query(z)[0]
    with np.load(p/'frontier.npz') as f:
     np.testing.assert_array_equal(obs,f['observations']);np.testing.assert_array_equal(z,f['latents']);np.testing.assert_allclose(scores,f['priorities'],atol=1e-12,rtol=1e-12)
    if j['role']=='reproduction':
     old=Path('runs/kl-mechanisms-online-20261110/online-mean_zero-seed9');oldmodel=Trainer.load(old/'checkpoint.pt');assert digest_model(t)==digest_model(oldmodel)
     with np.load(old/'replay.npz') as original:
      for k in ['observations','actions','next_observations','positions','next_positions']:np.testing.assert_array_equal(original[k],data[k])
   rows.append({**j,'checks_passed':True,'initial_hash':digest_model(init)})
  data0.close()
 return rows


def main():
 p=argparse.ArgumentParser();p.add_argument('--roots',nargs='+',required=True);p.add_argument('--noise-roots',nargs='*',default=[]);p.add_argument('--fixed',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args();seed_all(0,1)
 rows=[r for root in a.roots for r in audit(Path(root))];noise=[r for root in a.noise_roots for r in json.load(open(Path(root)/'results.json'))]
 for r in noise:
  assert len(r['ranking'])==32 and len(r['live'])==15
  for x in r['live']:assert x['rows'][-1]['k']==128 and 0<=x['new_states']<=1784-r['seen_states']
 for seed in [9,10,11]:
  for prefix in [50000,75000]:
   rr=[r for r in noise if r['seed']==seed and r['prefix']==prefix]
   if rr:assert len({(r['seen_states'],r['queue_entries'],r['productive_entries']) for r in rr})==1
 checks=[]
 if a.fixed:
  for seed in [9,10,11]:
   r=json.load(open(a.fixed/f'seed{seed}'/'result.json'));b=json.load(open(f'runs/kl-mechanisms-fixed-20261110/zero-mean_zero-seed{seed}/result.json'))
   assert r['spec']['latent_l2']==json.load(open('docs/assets/l2-noise-selection.json'))['selected']
   for k in ['initial_parameters_sha256','batch_indices_sha256','source_replay_sha256','updates','samples_per_head']:assert r[k]==b[k]
   checks.append(dict(seed=seed,checks_passed=True))
 a.out.write_text(json.dumps(dict(online=rows,noise_cases=len(noise),noise_checks_passed=True,shared_data=checks),indent=2)+'\n');print(len(rows),len(noise),len(checks))

if __name__=='__main__':main()
