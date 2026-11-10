"""Run a bounded, manifest-driven set of fixed-data overfitting diagnostics."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--jobs',type=int,default=3)
    args=p.parse_args()
    manifest=json.loads(args.manifest.read_text())
    args.out.mkdir(parents=True,exist_ok=False)
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    def run(v):
        path=args.out/v['name']
        command=[sys.executable,'scripts/memorize_dynamics.py','--checkpoint',manifest['checkpoint'],
                 '--data',manifest['data'],'--updates',str(manifest['updates']),
                 '--seed',str(manifest['seed']),'--out',str(path),*v['args']]
        with (args.out/(v['name']+'.log')).open('w') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        return {'name':v['name'],'run':str(path.resolve()),
                'experiment':json.loads((path/'experiment.json').read_text()),
                'metrics':json.loads((path/'metrics.json').read_text())}
    rows, failures=[],[]
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        pending={pool.submit(run,v):v['name'] for v in manifest['variants']}
        for future in as_completed(pending):
            try:
                row=future.result();rows.append(row)
                (args.out/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
                print(json.dumps({'done':row['name'],'train':row['metrics']['train'],'seconds':row['metrics']['seconds']}),flush=True)
            except Exception as exc:
                failures.append({'name':pending[future],'error':str(exc)})
                (args.out/'failures.json').write_text(json.dumps(failures,indent=2)+'\n')
    if failures:raise SystemExit(failures)


if __name__=='__main__':main()
