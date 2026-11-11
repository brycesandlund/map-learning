"""Physical identity of nearest latent memory, conditional on a true revisit."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from map_learning.training import Trainer,seed_all
from map_learning.maze import Maze,reachable_graph
from audit_frontier_efficiency import reconstruct,encode


def main():
 seed_all(0,1);rows=[]
 for kind in ['none','kl','l2']:
  for seed in [9,10,11]:
   path=f'runs/l2-noise-fixed-20261110/seed{seed}' if kind=='l2' else f'runs/kl-mechanisms-fixed-20261110/zero-{"mean_kl" if kind=="kl" else "mean_zero"}-seed{seed}'
   for prefix in [50000,75000]:
    t=Trainer.load(Path(path)/f'checkpoint-{prefix//5000}.pt')
    with np.load(f'runs/moment-prior-confirm-controls-20261110/learned-rep0.01-beta0-seed{seed-3}/replay.npz') as d:obs,pos,vo,vp=reconstruct(d,prefix)
    points,adj,index=reachable_graph(Maze(t.config.maze));ids=lambda ps:np.array([index[tuple(np.round(p,5))] for p in ps]);fi=ids(pos);vi=ids(vp);repeat=np.isin(fi,vi)
    near=cKDTree(encode(t,vo)).query(encode(t,obs))[1]
    rows.append({'kind':kind,'seed':seed,'prefix':prefix,'repeated_entries':int(repeat.sum()),'same_physical_nearest_given_repeated':float((vi[near][repeat]==fi[repeat]).mean())})
 Path('docs/assets/l2-noise-neighbors.json').write_text(json.dumps(rows,indent=2)+'\n');print(rows)

if __name__=='__main__':main()
