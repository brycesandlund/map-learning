"""Immutable manifest + bounded subprocess scheduler for distance experiments."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from map_learning.distance_learning import DistanceSpec


def screen_arms():
    arms = [{'name':'dynamics-l2', 'spec':None, 'l2':.03, 'repulsion':.03}]
    for w in [.01,.1,1.]:
        arms.append({'name':f'adjacency-w{w:g}', 'spec':{'kind':'adjacency','weight':w}, 'l2':.03,'repulsion':.03})
    for cost in [.05,.1,.25]:
        arms.append({'name':f'qrl-upper-c{cost:g}', 'spec':{'kind':'qrl_upper','weight':.03,'edge_cost':cost},'l2':.03,'repulsion':0.})
    for horizon in [4,16,64]:
        arms.append({'name':f'temporal-h{horizon}', 'spec':{'kind':'temporal','weight':1.,'horizon':horizon},'l2':.03,'repulsion':.03})
    for l2 in [.003,.03]:
        arms.append({'name':f'affinity-l2{l2:g}', 'spec':{'kind':'affinity','weight':.03},'l2':l2,'repulsion':0.})
    return arms


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--config',type=Path,default=Path('configs/deterministic_frontier.json'))
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--seeds',type=int,nargs='+',default=[18,19,20])
    p.add_argument('--blocks',type=int,default=20)
    p.add_argument('--arms',type=Path,help='Optional JSON list, frozen into manifest')
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    arms=json.loads(a.arms.read_text()) if a.arms else screen_arms()
    base_config=json.loads(a.config.read_text())
    (a.out/'base-config.json').write_text(json.dumps(base_config,indent=2)+'\n')
    for arm in arms:
        if arm['spec'] is not None:arm['spec']=asdict(DistanceSpec(**arm['spec']))
    manifest={'seeds':a.seeds,'blocks':a.blocks,'arms':arms,'workers':a.workers,'created_unix':time.time(),'base_config':base_config,
              'protocol':'Native deterministic initialization; interleaved100; Euclidean persistent frontier; no oracle labels. Distance arms train auxiliary dynamics on detached encodings. Temporal samples use actual occurrence ancestry.'}
    (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (a.out/'driver.pid').write_text(str(os.getpid()))
    jobs=[]
    for arm in arms:
        if arm['spec'] is not None:
            (a.out/f"{arm['name']}.json").write_text(json.dumps(arm['spec'],indent=2)+'\n')
    # Each batch starts with all objectives on one seed (early within-seed sanity checks).
    for seed in a.seeds:
        for arm in arms:jobs.append((arm,seed))
    def run(job):
        arm,seed=job; name=f"{arm['name']}-seed{seed}"
        command=[sys.executable,'scripts/persistent_frontier.py','--config',str(a.out/'base-config.json'),
                 '--arm','interleaved100','--seed',str(seed),'--blocks',str(a.blocks),'--out',str(a.out/name),
                 '--latent-l2',str(arm['l2']),'--repulsion-weight',str(arm['repulsion'])]
        if arm['spec'] is not None:command+=['--distance-spec',str(a.out/f"{arm['name']}.json")]
        started=time.time()
        with (a.out/f'{name}.log').open('x') as log:
            proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
            (a.out/f'{name}.pid').write_text(str(proc.pid))
            code=proc.wait()
        record={'name':name,'returncode':code,'seconds':time.time()-started,'command':command}
        (a.out/f'{name}.status.json').write_text(json.dumps(record,indent=2)+'\n')
        return record
    records=[]
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures=[pool.submit(run,job) for job in jobs]
        for future in as_completed(futures):
            record=future.result();records.append(record);print(json.dumps(record),flush=True)
    (a.out/'complete.json').write_text(json.dumps({'finished_unix':time.time(),'runs':len(jobs),'failed':sum(r['returncode']!=0 for r in records)}))
    if any(r['returncode'] for r in records):raise SystemExit(1)

if __name__=='__main__':main()
