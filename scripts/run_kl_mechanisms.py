"""Immutable, paired mechanism runs. Shared-data training or live exploration."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
from kl_mechanisms import ARMS,make_noise_schedule


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--mode',choices=['fixed','online'],required=True)
    p.add_argument('--arms',default=','.join(ARMS));p.add_argument('--seeds',default='9,10,11');p.add_argument('--families',default='winner,zero');p.add_argument('--jobs',type=int,default=6)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    schedule=make_noise_schedule();(a.out/'noise-schedule.json').write_text(json.dumps(schedule,indent=2)+'\n')
    jobs=[]
    for family in (a.families.split(',') if a.mode=='fixed' else ['online']):
        for arm in a.arms.split(','):
            assert arm in ARMS
            for seed in map(int,a.seeds.split(',')):
                jobs.append({'mode':a.mode,'family':family,'arm':arm,'seed':seed,'source_seed':seed-3,'path':str(a.out/f'{family}-{arm}-seed{seed}')})
    (a.out/'manifest.json').write_text(json.dumps({'jobs':jobs,'args':{**vars(a),'out':str(a.out)},'noise_schedule':schedule},indent=2)+'\n')
    def run(j):
        d=Path(j['path']);s=ARMS[j['arm']]
        if a.mode=='fixed':
            cmd=[sys.executable,'scripts/train_kl_mechanism_fixed.py','--out',str(d),'--arm',j['arm'],'--family',j['family'],
                 '--seed',str(j['seed']),'--source-seed',str(j['source_seed']),'--schedule',str(a.out/'noise-schedule.json')]
        else:
            cmd=[sys.executable,'scripts/persistent_frontier.py','--arm','interleaved100','--out',str(d),'--seed',str(j['seed']),
                 '--repulsion-weight','0.03','--bottleneck-beta',str(s['beta']),'--dynamics-input',s['dynamics_input'],
                 '--prior-lr-multiplier',str(s.get('prior_lr_multiplier',1.)),'--prior-steps',str(s.get('prior_steps',1))]
            if s['dynamics_input']=='scheduled':cmd+=['--noise-schedule',str(a.out/'noise-schedule.json')]
        with (a.out/(d.name+'.log')).open('x') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
        print(json.dumps({**j,'exit_code':r.returncode}),flush=True);return r.returncode
    with ThreadPoolExecutor(max_workers=a.jobs) as pool:codes=list(pool.map(run,jobs))
    if any(codes):raise SystemExit(1)

if __name__=='__main__':main()
