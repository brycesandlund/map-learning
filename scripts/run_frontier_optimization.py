"""Extra optimization timing: equal additional updates, paired seeds."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    (a.out/'manifest.json').write_text(json.dumps({'seeds':[0,1,2],'placements':['early','distributed'],
        'extra_updates':2000,'early_at_actions':5000,'distributed':'100 extra updates every 5000 actions',
        'base':'persistent interleaved100','actions':100000},indent=2)+'\n')
    def run(job):
        placement,seed=job;name=f'{placement}-seed{seed}'
        with (a.out/f'{name}.log').open('x') as log:
            r=subprocess.run([sys.executable,'scripts/persistent_frontier.py','--arm','interleaved100','--seed',str(seed),
                '--extra-updates','2000','--extra-placement',placement,'--out',str(a.out/name)],stdout=log,stderr=subprocess.STDOUT)
        print(json.dumps({'run':name,'exit_code':r.returncode}),flush=True);return r.returncode
    with ThreadPoolExecutor(max_workers=3) as pool:
        codes=list(pool.map(run,[(p,s) for s in range(3) for p in ['early','distributed']]))
    if any(codes):raise SystemExit(1)

if __name__=='__main__':main()
