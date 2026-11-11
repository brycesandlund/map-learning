"""Moment-prior grids and explicit fixed confirmation arms; immutable run outputs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--repulsions',default='0.01,0.03,0.1');p.add_argument('--betas',default='0,1e-6,1e-5,1e-4,1e-3,1e-2')
    p.add_argument('--prior-mode',choices=['learned','replay_moments','batch_moments'],default='replay_moments')
    p.add_argument('--specs',help='Explicit mode:alpha:beta triples, comma separated')
    p.add_argument('--seeds',default='0,1,2');p.add_argument('--jobs',type=int,default=6)
    p.add_argument('--kl-warmup-updates',type=int,default=0)
    p.add_argument('--reuse-zero',action='store_true');p.add_argument('--reproduce-zero',action='store_true')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    specs=([(s.split(':')[0],float(s.split(':')[1]),float(s.split(':')[2])) for s in a.specs.split(',')] if a.specs else
           [(a.prior_mode,rep,beta) for rep in map(float,a.repulsions.split(',')) for beta in map(float,a.betas.split(','))])
    oldrows=json.loads(Path('docs/assets/kl-repulsion-audit.json').read_text()) if a.reuse_zero else []
    lookup={(r['repulsion'],r['seed']):r['path'] for r in oldrows if r['beta']==0}
    jobs=[]
    for mode,rep,beta in specs:
        for seed in map(int,a.seeds.split(',')):
            reused=bool(a.reuse_zero and beta==0 and (rep,seed) in lookup)
            name=f'{mode}-rep{rep:g}-beta{beta:g}-seed{seed}'
            jobs.append(dict(prior_mode=mode,repulsion=rep,beta=beta,seed=seed,reused=reused,role='experiment',warmup_updates=a.kl_warmup_updates,
                             path=lookup[rep,seed] if reused else str(a.out/name)))
    if a.reproduce_zero:
        jobs.insert(0,dict(prior_mode='replay_moments',repulsion=.03,beta=0.,seed=0,reused=False,role='zero_reproduction',
                           path=str(a.out/'zero-reproduction-seed0')))
    (a.out/'manifest.json').write_text(json.dumps({'jobs':jobs,'args':{**vars(a),'out':str(a.out)}},indent=2)+'\n')
    def run(j):
        if j['reused']:return 0
        d=Path(j['path']);command=[sys.executable,'scripts/persistent_frontier.py','--arm','interleaved100','--seed',str(j['seed']),
            '--out',str(d),'--repulsion-weight',str(j['repulsion']),'--bottleneck-beta',str(j['beta']),'--prior-mode',j['prior_mode'],
            '--kl-warmup-updates',str(j.get('warmup_updates',0))]
        with (a.out/(d.name+'.log')).open('x') as log:r=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        print(json.dumps({**j,'exit_code':r.returncode}),flush=True);return r.returncode
    with ThreadPoolExecutor(max_workers=a.jobs) as pool:codes=list(pool.map(run,jobs))
    if any(codes):raise SystemExit(1)

if __name__=='__main__':main()
