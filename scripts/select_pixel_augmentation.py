"""Apply the prospectively recorded tuning gate; never inspect held-out results."""
import json,time
from pathlib import Path
import numpy as np
base=Path('runs/pixel-accelerated-long-20261111');root=Path('runs/pixel-attention-augmentation-20261111')
baseline=[json.loads((base/f'rgb-attention-seed{s}/result.json').read_text()) for s in [110,111]]
b_auc=np.mean([r['auc'] for r in baseline]);b_final=np.mean([r['coverage_k4'] for r in baseline]);rows=[]
for weight in [.3,3.,30.]:
 paths=[root/f'attention-aug{weight}-seed{s}/result.json' for s in [110,111]]
 if not all(p.exists() for p in paths):raise SystemExit('Wait for all six tuning runs; no partial selection')
 rs=[json.loads(p.read_text()) for p in paths];auc=np.mean([r['auc'] for r in rs]);final=np.mean([r['coverage_k4'] for r in rs])
 passed=(auc>=b_auc+.02 and final>=b_final) or (final>=b_final+.03 and auc>=b_auc)
 rows.append(dict(weight=weight,auc=float(auc),final=float(final),passes=bool(passed)))
passing=[r for r in rows if r['passes']];selected=max(passing,key=lambda r:r['auc']) if passing else None
record=dict(frozen_unix=time.time(),baseline_auc=float(b_auc),baseline_final=float(b_final),candidates=rows,selected=selected)
p=Path('configs/pixel-study-20261111/augmentation-selection.json')
with p.open('x') as f:json.dump(record,f,indent=2)
if selected:
 original=json.loads(Path('configs/pixel-study-20261111/learned-confirm.json').read_text())[0]
 arms=[dict(original,name='original'),dict(original,name='augmentation',augmentation=selected['weight']),dict(name='walk',policy='walk',persistence=.8,controls='cardinal',rgb=1,marker_size=1.3,steps=100000)]
 Path('configs/pixel-study-20261111/augmentation-confirm.json').write_text(json.dumps(arms,indent=2))
print(json.dumps(record,indent=2))
