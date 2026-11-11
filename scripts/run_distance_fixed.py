"""Bounded matched-data fit scheduler; source streams and arm selection frozen."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser();p.add_argument('--screen',type=Path,required=True);p.add_argument('--names',nargs='+',required=True)
    p.add_argument('--arms',type=Path,help='Optional override arm list; source streams still come from --screen')
    p.add_argument('--out',type=Path,required=True);p.add_argument('--workers',type=int,default=3);a=p.parse_args()
    manifest=json.loads((a.screen/'manifest.json').read_text())
    all_arms=json.loads(a.arms.read_text()) if a.arms else manifest['arms']
    arms=[x for x in all_arms if x['name'] in a.names]
    if len(arms)!=len(a.names):raise ValueError('Unknown or duplicate arm names')
    a.out.mkdir(parents=True,exist_ok=False)
    jobs=[]
    for arm in arms:(a.out/f"{arm['name']}.json").write_text(json.dumps(arm,indent=2)+'\n')
    for seed in manifest['seeds']:
        source=a.screen/f'dynamics-l2-seed{seed}'
        if not (source/'result.json').exists():raise ValueError(f'Incomplete source {source}')
        for arm in arms:jobs.append({'seed':seed,'source':str(source),'arm':arm})
    (a.out/'manifest.json').write_text(json.dumps({'jobs':jobs,'blocks':manifest['blocks']},indent=2)+'\n')
    (a.out/'driver.pid').write_text(str(os.getpid()))
    def run(j):
        name=f"{j['arm']['name']}-seed{j['seed']}";dest=a.out/name
        cmd=[sys.executable,'scripts/fit_distance_replay.py','--source',j['source'],'--arm',str(a.out/f"{j['arm']['name']}.json"),
             '--seed',str(j['seed']),'--blocks',str(manifest['blocks']),'--out',str(dest)]
        with (a.out/f'{name}.log').open('x') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
        record={'name':name,'returncode':r.returncode,'command':cmd}
        (a.out/f'{name}.status.json').write_text(json.dumps(record,indent=2)+'\n');return record
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for f in as_completed([pool.submit(run,j) for j in jobs]):print(json.dumps(f.result()),flush=True)
    (a.out/'complete.json').write_text(json.dumps({'jobs':len(jobs)}))

if __name__=='__main__':main()
