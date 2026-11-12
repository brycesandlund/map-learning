"""Finalize the bounded narrow-door study after its existing jobs complete."""
import json,subprocess,sys,time
from pathlib import Path
roots=[Path('runs')/f'gates-{kind}-20261111' for kind in ['learned','clean','walk','oracle']]
while not all((r/'complete.json').exists() for r in roots):time.sleep(10)
assert all(json.loads((r/'complete.json').read_text())['failed']==0 for r in roots)
def call(script,*args):subprocess.run([sys.executable,'scripts/'+script,*map(str,args)],check=True)
call('audit_pixel_study.py',roots[0],roots[1],'--out','docs/assets/gates-learned-audit.json')
call('analyze_gates.py',*roots,'--prefix','docs/assets/gates-results')
paths=[p for root in roots[:2] for p in sorted(root.iterdir()) if p.is_dir() and json.loads((p/'spec.json').read_text())['gate_width']==.0625]
call('probe_gates_precision.py',*paths,'--out','docs/assets/gates-precision.json')
call('probe_gates_frontier.py',*paths,'--out','docs/assets/gates-frontier.json')
rows=[]
for root in roots:
 for p in sorted(root.glob('*/result.json')):
  r=json.loads(p.read_text());rows.append(dict(run=str(p.parent),steps=r['interactions'],updates=r['updates'],samples=r['samples']))
audit=[]
for name in ['gates-controls-audit.json','gates-learned-audit.json']:audit+=json.loads((Path('docs/assets')/name).read_text())['results']
assert len(rows)==len(audit)==69
assert {r['run'] for r in rows}=={r['run'] for r in audit}
record=dict(counterfactual_extra_actions=3000,smoke_actions=2000,failed_counterfactual_attempt_actions=0,runs=len(rows),actions=sum(r['steps'] for r in rows),updates=sum(r['updates'] for r in rows),presentations_per_head=sum(r['samples'] for r in rows),audited=len(audit),rows=rows)
Path('docs/assets/gates-accounting.json').write_text(json.dumps(record,indent=2));print('Final accounting',json.dumps({k:v for k,v in record.items() if k!='rows'}),flush=True)
