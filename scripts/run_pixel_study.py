"""Immutable bounded pixel study; each job runs in an isolated process."""
import argparse,json,subprocess,sys,time,os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
p=argparse.ArgumentParser();p.add_argument('--arms',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--seeds',type=int,nargs='+',required=True);p.add_argument('--workers',type=int,default=4)
a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);arms=json.loads(a.arms.read_text())
(a.out/'manifest.json').write_text(json.dumps(dict(arms=arms,seeds=a.seeds,created=time.time()),indent=2))
(a.out/'driver.pid').write_text(str(os.getpid()))
def run(job):
    seed,arm=job;name=arm['name']+f'-seed{seed}'
    cmd=[sys.executable,'scripts/pixel_frontier.py','--out',str(a.out/name),'--seed',str(seed)]
    for k,v in arm.items():
        if k!='name':cmd+=['--'+k.replace('_','-'),str(v)]
    with (a.out/(name+'.log')).open('x') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
    result=dict(name=name,returncode=r.returncode,command=cmd)
    (a.out/(name+'.status.json')).write_text(json.dumps(result));return result
with ThreadPoolExecutor(a.workers) as ex:
    results=[]
    for f in as_completed([ex.submit(run,(s,arm)) for s in a.seeds for arm in arms]):
        row=f.result();results.append(row);print(json.dumps(row),flush=True)
(a.out/'complete.json').write_text(json.dumps(dict(runs=len(results),failed=sum(r['returncode']!=0 for r in results))))
