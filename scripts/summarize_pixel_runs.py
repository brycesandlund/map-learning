import sys,json,collections
from pathlib import Path
import numpy as np
for root in sys.argv[1:]:
 d=collections.defaultdict(list)
 for p in Path(root).glob('*/result.json'):
  r=json.loads(p.read_text());d[p.parent.name.rsplit('-seed',1)[0]].append(r)
 print(root)
 for name,rs in sorted(d.items(),key=lambda kv:-np.mean([r['auc'] for r in kv[1]])):
  print(name,len(rs),'final',round(np.mean([r['coverage_k4']*100 for r in rs]),2),'AUC',round(np.mean([r['auc']*100 for r in rs]),2))
