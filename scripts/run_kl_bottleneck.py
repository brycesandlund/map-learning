"""Bottleneck sweep with preserved output directories and bounded concurrency."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--components',type=int,default=1);p.add_argument('--std',type=float,default=.01)
    p.add_argument('--kl-warmup-updates',type=int,default=0)
    p.add_argument('--prior-lr-multiplier',type=float,default=1.)
    p.add_argument('--betas',default='0,1e-6,1e-5,1e-4,1e-3,1e-2');p.add_argument('--seeds',default='0,1,2')
    p.add_argument('--jobs',type=int,default=3);p.add_argument('--deterministic-control',action='store_true')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    seeds=[int(x) for x in a.seeds.split(',')];betas=[float(x) for x in a.betas.split(',')]
    jobs=[(b,s) for s in seeds for b in betas]
    if a.deterministic_control:jobs.insert(0,(None,0))
    manifest=vars(a).copy();manifest['out']=str(a.out)
    (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    def run(job):
        beta,seed=job;name=f'{"deterministic" if beta is None else "beta"+format(beta,"g")}-seed{seed}'
        command=[sys.executable,'scripts/persistent_frontier.py','--arm','interleaved100','--seed',str(seed),'--out',str(a.out/name)]
        if beta is not None:command+=['--bottleneck-beta',str(beta),'--prior-components',str(a.components),'--initial-std',str(a.std),'--prior-lr-multiplier',str(a.prior_lr_multiplier),'--kl-warmup-updates',str(a.kl_warmup_updates)]
        with (a.out/f'{name}.log').open('x') as log:r=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        print(json.dumps({'run':name,'exit_code':r.returncode}),flush=True);return r.returncode
    with ThreadPoolExecutor(max_workers=a.jobs) as pool:codes=list(pool.map(run,jobs))
    if any(codes):raise SystemExit(1)

if __name__=='__main__':main()
