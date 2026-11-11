"""Immutable bounded experiment runner for continuous frontier arms."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--arms',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=[30,31,32])
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--blocks',type=int,default=20)
    a=p.parse_args()
    arms=json.loads(a.arms.read_text());a.out.mkdir(parents=True,exist_ok=False)
    config=Path('configs/continuous_frontier.json').read_text();(a.out/'base-config.json').write_text(config)
    manifest={'arms':arms,'seeds':a.seeds,'blocks':a.blocks,'created_unix':time.time(),'workers':a.workers}
    (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (a.out/'driver.pid').write_text(str(os.getpid()))
    def run(job):
        seed,arm=job;name=f"{arm['name']}-seed{seed}"
        cmd=[sys.executable,'scripts/continuous_frontier.py','--config',str(a.out/'base-config.json'),
             '--objective',arm['objective'],'--seed',str(seed),'--blocks',str(a.blocks),'--out',str(a.out/name)]
        for key,value in arm.items():
            if key not in {'name','objective'}:cmd += ['--'+key.replace('_','-'),str(value)]
        start=time.time()
        with (a.out/f'{name}.log').open('x') as log:
            proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
            (a.out/f'{name}.pid').write_text(str(proc.pid));code=proc.wait()
        row={'name':name,'returncode':code,'seconds':time.time()-start,'command':cmd}
        (a.out/f'{name}.status.json').write_text(json.dumps(row,indent=2)+'\n')
        return row
    jobs=[(seed,arm) for seed in a.seeds for arm in arms]
    results=[]
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for future in as_completed([pool.submit(run,job) for job in jobs]):
            result=future.result();results.append(result);print(json.dumps(result),flush=True)
    (a.out/'complete.json').write_text(json.dumps({'runs':len(results),'failed':sum(x['returncode']!=0 for x in results)}))
    if any(x['returncode'] for x in results):raise SystemExit(1)

if __name__=='__main__':main()
