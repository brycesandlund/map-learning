"""Cross-run controls and exact accounting, including added rollout work."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('roots',nargs='+',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
records=[];pairs={}
def sha(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
for root in a.roots:
 m=json.loads((root/'manifest.json').read_text());assert json.loads((root/'complete.json').read_text())['failed']==0
 for seed in m['seeds']:
  for arm in m['arms']:
   path=root/f"{arm['name']}-seed{seed}";r=json.loads((path/'result.json').read_text());data=np.load(path/'replay.npz')
   key=(seed,r['layout'],r['step_size'],r['nuisance_dims'])
   hashes=(r['initial_parameters_sha256'],sha(data['actions']),sha(data['next_observations'][:,2:]))
   if key in pairs:assert pairs[key]==hashes,(path,'unmatched controls')
   else:pairs[key]=hashes
   logs=[json.loads(s) for s in (path/'training.jsonl').read_text().splitlines()]
   rows={'run':str(path),'actions':r['interactions'],'updates':r['updates'],'one_step_presentations':r['samples_per_head'],
         'path_presentations':logs[-1].get('path_presentations',0),'path_edges':logs[-1].get('path_edges',0)}
   assert rows['updates']==sum((5000*i+255)//256 for i in range(1,m['blocks']+1))
   assert rows['one_step_presentations']==sum(5000*i for i in range(1,m['blocks']+1))
   if (path/'ancestry.npz').exists():
    anc=np.load(path/'ancestry.npz');parents=anc['parents'][1:]
    assert np.all(parents<np.arange(1,len(parents)+1))
    assert np.array_equal(anc['observations'][parents],data['observations'])
    assert np.array_equal(anc['observations'][1:],data['next_observations'])
    assert np.array_equal(anc['depth'][1:],anc['depth'][parents]+1)
   records.append(rows)
result={'jobs':len(records),'totals':{k:sum(r[k] for r in records) for k in ['actions','updates','one_step_presentations','path_presentations','path_edges']},'paired_initial_weights_actions_successor_noise':True,'ancestry_occurrences_verified':True,'runs':records}
a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='runs'},indent=2))
