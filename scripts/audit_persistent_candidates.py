from pathlib import Path
import json
import numpy as np
from map_learning.maze import Maze,reachable_graph
from map_learning.config import load_config
rows=[]
for seed in range(3):
 run=Path(f'runs/horizon-duration-20261110/h16_shared_30k-seed{seed}');env=Maze(load_config(run/'config.json').maze);points,adj,index=reachable_graph(env);visited=np.zeros(len(points),bool)
 with np.load(run/'replay.npz') as data:
  for p in np.concatenate([data['positions'],data['next_positions']]):visited[index[tuple(p)]]=True
 for persistent in [0.,.8]:
  for h in [32,64]:
   for trial in range(5):
    rng=np.random.default_rng(seed*1000+trial);real=np.full(128,index[tuple(env.start)]);previous=rng.integers(5,size=128);found=np.zeros(128,bool)
    for depth in range(h):
     action=np.where(rng.random(128)<persistent,previous,rng.integers(5,size=128));previous=action;real=adj[real,action];found|=~visited[real]
    rows.append({'seed':seed,'persistence':persistent,'horizon':h,'trial':trial,'candidates':128,'novel_candidates':int(found.sum())})
Path('docs/assets/reset-candidate-audit.json').write_text(json.dumps(rows,indent=2)+'\n')
for persistence in [0.,.8]:
 for h in [32,64]:
  rs=[r for r in rows if r['persistence']==persistence and r['horizon']==h];print(persistence,h,sum(r['novel_candidates']>0 for r in rs),'/15',sum(r['novel_candidates'] for r in rs))
