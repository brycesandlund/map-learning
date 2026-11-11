"""Resume-safe geometry and common-frontier audits of completed fixed-data fits."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
import os
from pathlib import Path
import subprocess,sys


def main():
    p=argparse.ArgumentParser();p.add_argument('--fixed',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--workers',type=int,default=2);p.add_argument('--prefixes',type=int,nargs='+',default=[25000,50000]);a=p.parse_args()
    manifest=json.loads((a.fixed/'manifest.json').read_text());a.out.mkdir(parents=True,exist_ok=True)
    spec={'fixed':str(a.fixed),'prefixes':a.prefixes,'live_prefixes':[25000]}
    if (a.out/'manifest.json').exists():
        if json.loads((a.out/'manifest.json').read_text())!=spec:raise ValueError('Mismatched audit manifest')
    else:(a.out/'manifest.json').write_text(json.dumps(spec,indent=2)+'\n')
    jobs=[]
    for row in manifest['jobs']:
        name=f"{row['arm']['name']}-seed{row['seed']}";fit=a.fixed/name
        if not (fit/'result.json').exists():continue
        for prefix in a.prefixes:
            dest=a.out/f'{name}-at{prefix}'
            if Path(str(dest)+'.json').exists() and Path(str(dest)+'.npz').exists():continue
            jobs.append((fit,Path(row['source']),prefix,dest))
    def run(job):
        fit,source,prefix,dest=job
        command=[sys.executable,'scripts/audit_distance_geometry.py','--checkpoint',str(fit/f'checkpoint-{prefix//5000}.pt'),
                 '--source',str(source),'--prefix',str(prefix),'--out',str(dest)]
        if prefix==25000:command+=['--live']
        # A retry receives a unique log; completed result artifacts are never overwritten.
        logpath=dest.parent/f'{dest.name}.log'
        if logpath.exists():logpath=dest.parent/f'{dest.name}-retry-{os.getpid()}.log'
        with logpath.open('x') as log:r=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        return {'name':dest.name,'returncode':r.returncode}
    records=[]
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for f in as_completed([pool.submit(run,j) for j in jobs]):
            r=f.result();records.append(r);print(json.dumps(r),flush=True)
    if any(r['returncode'] for r in records):raise SystemExit(1)

if __name__=='__main__':main()
