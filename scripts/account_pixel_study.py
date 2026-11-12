"""Inventory complete, interrupted and unlaunched jobs without hiding pilots."""
import argparse, json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
roots=[];complete=[];partial=[];unlaunched=[]
for root in sorted(Path('runs').glob('pixel-*-20261111')):
    manifest=root/'manifest.json'
    if not manifest.exists():continue
    m=json.loads(manifest.read_text());done=[];unfinished=[]
    for seed in m['seeds']:
        for arm in m['arms']:
            name=arm['name']+f'-seed{seed}';run=root/name
            if (run/'result.json').exists():
                r=json.loads((run/'result.json').read_text());s=json.loads((run/'spec.json').read_text())
                status=root/(name+'.status.json')
                assert status.exists() and json.loads(status.read_text())['returncode']==0, run
                row=dict(run=str(run),seed=seed,arm=arm['name'],family=s['family'],policy=s['policy'],steps=r['interactions'],updates=r['updates'],samples=r['samples'],coverage=r['coverage_k4'],auc=r['auc'])
                assert row['steps']==s['steps'],run
                complete.append(row);done.append(row)
            elif run.exists():
                progress=run/'progress.jsonl';lines=progress.read_text().splitlines() if progress.exists() else []
                row=dict(run=str(run),recorded_actions=json.loads(lines[-1])['interactions'] if lines else 0,stopped=(root/'stopped.json').exists())
                partial.append(row);unfinished.append(row)
            else:unlaunched.append(str(run))
    roots.append(dict(root=str(root),planned=len(m['seeds'])*len(m['arms']),complete=len(done),partial=len(unfinished),stopped=(root/'stopped.json').exists(),steps=sum(r['steps'] for r in done)))
smokes=[]
for path in sorted(Path('runs').glob('pixel-*smoke*-20261111/result.json')):
    r=json.loads(path.read_text());smokes.append(dict(run=str(path.parent),steps=r['interactions']))
record=dict(complete_runs=len(complete),trained_runs=sum(r['updates']>0 for r in complete),untrained_runs=sum(r['updates']==0 for r in complete),complete_actions=sum(r['steps'] for r in complete),gradient_updates=sum(r['updates'] for r in complete),transition_presentations_per_head=sum(r['samples'] for r in complete),partial_recorded_actions=sum(r['recorded_actions'] for r in partial),roots=roots,completed=complete,partial=partial,unlaunched=unlaunched,smokes=smokes)
a.out.write_text(json.dumps(record,indent=2));print(json.dumps({k:v for k,v in record.items() if not isinstance(v,list)},indent=2));print('Partial',len(partial),'unlaunched',len(unlaunched),'smokes',len(smokes))
