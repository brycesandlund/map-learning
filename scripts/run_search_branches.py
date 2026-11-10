"""Run isolated subprocesses, refusing partial or completed output overwrite."""
import subprocess,sys,json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
root=Path('runs/search-budget-20261110');root.mkdir(exist_ok=True)
jobs=[]
for budget in [384,96,24,'random']:
 for seed in range(3):
  name=f'{budget}-seed{seed}';out=root/name
  if out.exists():raise FileExistsError(out)
  command=[sys.executable,'scripts/continue_search_budget.py','--source',f'runs/horizon-duration-20261110/h16_shared_30k-seed{seed}','--out',str(out)]
  command+=['--random'] if budget=='random' else ['--simulations',str(budget)]
  jobs.append((name,command))
(root/'protocol.json').write_text(json.dumps({'steps':1200,'horizon':16,'budgets':[24,96,384],'seeds':[0,1,2],'random_control':True,'training':'epoch every10','source':'h16_shared_30k'},indent=2))
def run(job):
 name,command=job
 with (root/f'{name}.log').open('w') as f:subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,check=True)
 return json.loads((root/name/'result.json').read_text())
rows=[]
with ThreadPoolExecutor(max_workers=4) as pool:
 for future in as_completed([pool.submit(run,j) for j in jobs]):
  rows.append(future.result());(root/'summary.json').write_text(json.dumps(rows,indent=2));print(rows[-1],flush=True)
