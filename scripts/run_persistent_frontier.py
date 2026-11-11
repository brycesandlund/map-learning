"""Matched persistent distance-frontier cadence study; no overwriting."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--jobs',type=int,default=3);p.add_argument('--blocks',type=int,default=20)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    jobs=[(arm,seed) for seed in range(3) for arm in ['epoch1000','interleaved100']]
    (a.out/'manifest.json').write_text(json.dumps({'arms':['epoch1000','interleaved100'],'seeds':[0,1,2],
        'blocks':a.blocks,'expansions_per_block':1000,'epoch_exposure':'each accumulated transition exactly once per block per head',
        'updates_per_block':'ceil(5000 * block_number / 256)','batch_size_note':'epoch arm max256 with tail; frequent arm evenly divides interval samples across allotted updates'},indent=2)+'\n')
    def run(job):
        arm,seed=job;name=f'{arm}-seed{seed}'
        with (a.out/f'{name}.log').open('x') as log:
            result=subprocess.run([sys.executable,'scripts/persistent_frontier.py','--arm',arm,'--seed',str(seed),
                '--blocks',str(a.blocks),'--out',str(a.out/name)],stdout=log,stderr=subprocess.STDOUT)
        print(json.dumps({'run':name,'exit_code':result.returncode}),flush=True)
        return result.returncode
    with ThreadPoolExecutor(max_workers=a.jobs) as pool:codes=list(pool.map(run,jobs))
    if any(codes):raise SystemExit(1)

if __name__=='__main__':main()
