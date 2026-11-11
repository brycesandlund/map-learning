"""Finish-time fitting audits; read-only with respect to model and replay."""
from pathlib import Path
import json,time
from analyze_novelty import evaluate
from map_learning.training import seed_all
seed_all(0,1)
expected=[Path('runs/novelty-rewards-20261110')/f'{mode}-seed{seed}' for mode in ['root_distance','path_distance','archive_path','disagreement_path'] for seed in range(3)]
expected += [Path('runs/novelty-horizon-20261110')/f'archive_path-seed{seed}' for seed in range(3)]
expected += [Path('runs')/root/f'{mode}-seed{seed}' for root,mode in [('novelty-persistent-20261110','archive_path'),('novelty-recent-20261110','recent_archive')] for seed in range(3)]
while True:
 done=0
 for run in expected:
  out=run/'fitting-audit.json'
  if out.exists():done+=1;continue
  if not (run/'result.json').exists():continue
  result=evaluate(run,run/'checkpoint.pt',31200);out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(run,flush=True);done+=1
 if done==len(expected):break
 time.sleep(10)
