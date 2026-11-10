"""Audit completed branch fitting without changing checkpoints/replay."""
import json,time
from pathlib import Path
from analyze_novelty import evaluate
from map_learning.training import seed_all
seed_all(0,1)
root=Path('runs/search-budget-20261110')
expected=[root/f'{budget}-seed{seed}' for budget in [24,96,384,'random'] for seed in range(3)]
while True:
 completed=0
 for run in expected:
  out=run/'fitting-audit.json'
  if out.exists():completed+=1;continue
  if not (run/'result.json').exists():continue
  row=evaluate(run,run/'checkpoint.pt',31200)
  out.write_text(json.dumps(row,indent=2,allow_nan=False)+'\n');completed+=1;print(run.name,flush=True)
 if completed==len(expected):break
 time.sleep(10)
