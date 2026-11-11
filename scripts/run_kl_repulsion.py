"""Matched KL x repulsion sweeps; existing Gaussian controls are explicitly reused."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--repulsions',default='0.01,0.1,1')
    p.add_argument('--betas',default='0,1e-6,1e-5,1e-4,1e-3')
    p.add_argument('--pairs',help='Optional explicit alpha:beta pairs, comma separated')
    p.add_argument('--seeds',default='0,1,2')
    p.add_argument('--jobs',type=int,default=3)
    p.add_argument('--reuse-original',action='store_true')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    jobs=[]
    pairs=([tuple(map(float,x.split(':'))) for x in a.pairs.split(',')] if a.pairs else
           [(r,b) for r in map(float,a.repulsions.split(',')) for b in map(float,a.betas.split(','))])
    for rep,beta in pairs:
        if rep<0 or beta<0:raise ValueError('Coefficients must be nonnegative')
        for seed in map(int,a.seeds.split(',')):
            name=f'rep{rep:g}-beta{beta:g}-seed{seed}'
            old=Path(f'runs/kl-gaussian-20261110/beta{beta:g}-seed{seed}')
            reuse=a.reuse_original and rep==.01 and (old/'result.json').exists()
            path=old if reuse else a.out/name
            if reuse:
                cfg=json.loads((path/'config.json').read_text())
                r=json.loads((path/'result.json').read_text())
                assert cfg['train']['repulsion_weight']==rep and r['bottleneck_spec']['beta']==beta
                assert r['interactions']==100000 and r['updates']==4112
            jobs.append(dict(repulsion=rep,beta=beta,seed=seed,path=str(path),reused=reuse))
    (a.out/'manifest.json').write_text(json.dumps({'jobs':jobs,'args':{**vars(a),'out':str(a.out)}},indent=2)+'\n')
    def run(job):
        if job['reused']:return 0
        path=Path(job['path'])
        command=[sys.executable,'scripts/persistent_frontier.py','--arm','interleaved100','--seed',str(job['seed']),
                 '--out',str(path),'--repulsion-weight',str(job['repulsion']),'--bottleneck-beta',str(job['beta'])]
        with (a.out/(path.name+'.log')).open('x') as log:r=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        print(json.dumps({**job,'exit_code':r.returncode}),flush=True);return r.returncode
    with ThreadPoolExecutor(max_workers=a.jobs) as pool:codes=list(pool.map(run,jobs))
    if any(codes):raise SystemExit(1)

if __name__=='__main__':main()
