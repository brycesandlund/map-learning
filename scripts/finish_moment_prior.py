"""Wait for fixed screens, freeze selection, run new seeds, then audit/report.

This helper preserves completed runs; use only with the recorded study roots.
"""
import json
import subprocess
import sys
import time
from datetime import datetime,timezone
from pathlib import Path
from collections import defaultdict
import numpy as np

ROOTS=['runs/moment-prior-20261110','runs/moment-prior-batch-20261110','runs/moment-prior-fine-20261110',
       'runs/moment-prior-delayed-20261110','runs/moment-prior-delayed-strong-20261110']

def main():
    jobs=[j for root in ROOTS for j in json.loads((Path(root)/'manifest.json').read_text())['jobs']]
    while not all((Path(j['path'])/'result.json').exists() for j in jobs):time.sleep(2)
    groups=defaultdict(list)
    for j in jobs:
        if j['beta']==0 or j.get('role')=='zero_reproduction':continue
        d=Path(j['path']);r=json.loads((d/'result.json').read_text())
        pr=[json.loads(x) for x in (d/'progress.jsonl').read_text().splitlines()]
        auc=float(np.trapezoid([1/1784]+[x['coverage'] for x in pr],[0]+[x['interactions'] for x in pr])/100000)
        key=j['prior_mode'],j['repulsion'],j['beta'],j.get('warmup_updates',0)
        groups[key].append(dict(seed=j['seed'],auc=auc,first_full=r['first_full'],visited=r['visited']))
    for g in groups.values():assert len(g)==3
    selected=max(groups,key=lambda k:np.mean([r['auc'] for r in groups[k]]))
    mode,rep,beta,warm=selected
    selection={'selected':list(selected),'chosen_before_confirmation_utc':datetime.now(timezone.utc).isoformat(),
               'criterion':'Maximum mean coverage AUC among positive-beta moment variants on seeds 0/1/2 only',
               'candidates':[dict(key=list(k),mean_auc=float(np.mean([r['auc'] for r in rr])),seeds=rr) for k,rr in groups.items()]}
    p=Path('docs/assets/moment-prior-selection.json')
    with p.open('x') as f:json.dump(selection,f,indent=2)
    print(json.dumps({'selected':selected,'screen':groups[selected]}),flush=True)
    with Path('docs/moment-prior.md').open('a') as f:f.write(f'\nFixed confirmation candidate after all screens: {mode}, α={rep:g}, β={beta:g}, warmup={warm} updates; selected by screening mean AUC. Run seeds 6/7/8 against the already-fixed new controls. Selection recorded before candidate runs in docs/assets/moment-prior-selection.json.\n')
    candidate='runs/moment-prior-confirm-candidate-20261110'
    subprocess.run([sys.executable,'scripts/run_moment_prior.py','--out',candidate,'--specs',f'{mode}:{rep}:{beta}',
                    '--kl-warmup-updates',str(warm),'--seeds','6,7,8','--jobs','3'],check=True)
    controls=Path('runs/moment-prior-confirm-controls-20261110')
    cj=json.loads((controls/'manifest.json').read_text())['jobs']
    while not all((Path(j['path'])/'result.json').exists() for j in cj):time.sleep(2)
    combined=Path('runs/moment-prior-confirmation-20261110');combined.mkdir(exist_ok=False)
    cm=json.loads((Path(candidate)/'manifest.json').read_text())['jobs']
    (combined/'manifest.json').write_text(json.dumps({'jobs':cj+cm,'args':{'selection':selection['selected'],'controls':str(controls),'candidate':candidate}},indent=2)+'\n')
    subprocess.run([sys.executable,'scripts/audit_moment_prior.py','--roots',*ROOTS,str(combined),'--resume'],check=True)
    label=mode+(f'_warm{warm}' if warm else '')
    subprocess.run([sys.executable,'scripts/report_moment_prior.py','--screen-roots',*ROOTS,'--confirmation',str(combined),
                    '--selected',f'{label}:{rep}:{beta}'],check=True)

if __name__=='__main__':main()
