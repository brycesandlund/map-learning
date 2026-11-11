import argparse,json,subprocess,sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
p=argparse.ArgumentParser();p.add_argument('--manifest',default='configs/novelty_rewards.json');p.add_argument('--out',default='runs/novelty-rewards-20261110');p.add_argument('--jobs',type=int,default=4);a=p.parse_args()
m=json.loads(Path(a.manifest).read_text());root=Path(a.out);root.mkdir(exist_ok=False);(root/'manifest.json').write_text(json.dumps(m,indent=2))
jobs=([('disagreement',m['control_seed'])] if 'control_seed' in m else [])+[(mode,seed) for mode in m['rewards'] for seed in m['seeds']]
def run(job):
 mode,seed=job;name=f'{mode}-seed{seed}';out=root/name
 command=[sys.executable,'scripts/continue_novelty_reward.py','--source',m['source_template'].format(seed=seed),'--out',str(out),'--reward',mode,'--steps',str(m['steps']),'--simulations',str(m['simulations'])]
 for key in ['horizon','search','candidates']:
  if key in m:command+=['--'+key,str(m[key])]
 with (root/f'{name}.log').open('w') as f:subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,check=True)
 return json.loads((out/'result.json').read_text())
rows=[];failures=[]
with ThreadPoolExecutor(max_workers=a.jobs) as pool:
 pending={pool.submit(run,j):j for j in jobs}
 for future in as_completed(pending):
  try:rows.append(future.result());(root/'summary.json').write_text(json.dumps(rows,indent=2));print(rows[-1],flush=True)
  except Exception as exc:failures.append({'job':pending[future],'error':str(exc)});(root/'failures.json').write_text(json.dumps(failures,indent=2));print(failures[-1],flush=True)
if failures:raise SystemExit('Some runs failed; outputs preserved.')
