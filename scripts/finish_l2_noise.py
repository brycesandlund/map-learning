"""Freeze screen selection, confirm on fresh seeds, fit common-data candidate."""
import json,subprocess,sys,time
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import numpy as np


def main():
 root=Path('runs/l2-noise-screen-20261110');jobs=json.load(open(root/'manifest.json'))['jobs']
 while not all((Path(j['path'])/'result.json').exists() for j in jobs):time.sleep(2)
 groups=defaultdict(list)
 for j in jobs:
  if j['role']!='experiment' or j['kind']!='l2' or not j['latent_l2']:continue
  p=Path(j['path']);pr=[json.loads(x) for x in (p/'progress.jsonl').read_text().splitlines()];auc=float(np.trapezoid([1/1784]+[x['coverage'] for x in pr],[0]+[x['interactions'] for x in pr])/100000)
  groups[j['latent_l2']].append(dict(seed=j['seed'],auc=auc))
 selected=max(groups,key=lambda k:np.mean([r['auc'] for r in groups[k]]))
 selection=dict(selected=selected,criterion='Maximum mean coverage AUC over100k on seeds9/10/11, positive lambda only',candidates=[dict(latent_l2=k,seeds=v) for k,v in groups.items()])
 with Path('docs/assets/l2-noise-selection.json').open('x') as f:json.dump(selection,f,indent=2)
 print(json.dumps(selection),flush=True)
 fixed=Path('runs/l2-noise-fixed-20261110');fixed.mkdir(exist_ok=False)
 schedule='runs/kl-mechanisms-fixed-20261110/noise-schedule.json'
 def fit(seed):
  path=fixed/f'seed{seed}'
  cmd=[sys.executable,'scripts/train_kl_mechanism_fixed.py','--out',str(path),'--arm','mean_zero','--family','zero','--source-seed',str(seed-3),'--seed',str(seed),'--schedule',schedule,'--latent-l2',str(selected)]
  with (fixed/f'seed{seed}.log').open('x') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
 with ThreadPoolExecutor(max_workers=4) as pool:
  fs=[pool.submit(fit,s) for s in [9,10,11]]
  fs.append(pool.submit(subprocess.run,[sys.executable,'scripts/run_l2_sweep.py','--out','runs/l2-noise-confirm-20261110','--lambdas',f'0,{selected}','--seeds','15,16,17','--jobs','3'],check=True))
  for f in fs:f.result()
 subprocess.run([sys.executable,'scripts/noise_search_diagnostics.py','--out','runs/l2-noise-selected-20261110','--l2-root',str(fixed),'--jobs','4'],check=True)

if __name__=='__main__':main()
