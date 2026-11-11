"""Later shared-archive rollouts after posterior noise has diverged between arms."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from map_learning.training import Trainer,seed_all
from kl_mechanisms import ARMS,queue_audit,data_source
from audit_moment_prior import full_moments


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--arm',choices=ARMS);p.add_argument('--seed',type=int);p.add_argument('--prefix',type=int);p.add_argument('--jobs',type=int,default=6);a=p.parse_args()
    if a.arm:
        seed_all(0,1);d=a.root/f'zero-{a.arm}-seed{a.seed}';t=Trainer.load(d/f'checkpoint-{a.prefix//5000}.pt')
        with np.load(data_source('zero',a.seed-3)/'replay.npz') as data:
            q=queue_audit(t,data,a.prefix,a.seed-3,live=True);q['moments']=full_moments(t,data,a.prefix)
        result={'arm':a.arm,'seed':a.seed,'prefix':a.prefix,'family':'zero','path':str(d),'queue':q}
        with (a.out/f'{a.arm}-seed{a.seed}-at{a.prefix}.json').open('x') as f:json.dump(result,f,indent=2)
        return
    a.out.mkdir(exist_ok=False)
    while not all((a.root/f'zero-{arm}-seed{s}'/'result.json').exists() for arm in ARMS for s in [9,10,11]):time.sleep(2)
    jobs=[(arm,s,prefix) for arm in ARMS for s in [9,10,11] for prefix in [50000,75000]]
    (a.out/'manifest.json').write_text(json.dumps({'jobs':[dict(arm=x,seed=y,prefix=z) for x,y,z in jobs]},indent=2))
    def run(j):
        arm,s,prefix=j
        cmd=[sys.executable,__file__,'--root',str(a.root),'--out',str(a.out),'--arm',arm,'--seed',str(s),'--prefix',str(prefix)]
        with (a.out/f'{arm}-seed{s}-at{prefix}.log').open('x') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
        print(json.dumps({'arm':arm,'seed':s,'prefix':prefix,'exit_code':r.returncode}),flush=True);return r.returncode
    with ThreadPoolExecutor(max_workers=a.jobs) as pool:codes=list(pool.map(run,jobs))
    if any(codes):raise SystemExit(1)
    results=[json.loads((a.out/f'{arm}-seed{s}-at{prefix}.json').read_text()) for arm,s,prefix in jobs]
    (a.out/'results.json').write_text(json.dumps(results,indent=2)+'\n')

if __name__=='__main__':main()
