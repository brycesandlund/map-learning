"""Full replay edge-constraint residuals for completed QRL-inspired runs."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from map_learning.training import Trainer,seed_all


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('roots',type=Path,nargs='+');p.add_argument('--out',type=Path,required=True);a=p.parse_args();seed_all(0,1)
    rows=[]
    for root in a.roots:
        for checkpoint in sorted(root.glob('qrl-*/checkpoint.pt')):
            t=Trainer.load(checkpoint);s=t.distance_spec
            with np.load(checkpoint.parent/'replay.npz') as replay:
                obs=replay['observations'];nxt=replay['next_observations']
            distances=[]
            for i in range(0,len(obs),1024):
                za,zb=t.model.encode(t.tensor(np.concatenate([obs[i:i+1024],nxt[i:i+1024]]))).chunk(2)
                distances.extend((torch.linalg.vector_norm(za-zb,dim=-1)/s.edge_cost).numpy().tolist())
            d=np.asarray(distances);violation=float(np.mean(np.maximum(d-1,0)**2))
            row={'run':str(checkpoint.parent),'spec':vars(s),'latent_l2':t.config.train.latent_l2,'transitions':len(d),
                 'epsilon_squared':s.epsilon**2,'full_replay_squared_upper_violation':violation,
                 'violation_over_budget':violation/s.epsilon**2,'multiplier':float(t.multiplier),
                 'fraction_edges_above_cost':float(np.mean(d>1)),'normalized_edge_quantiles':np.quantile(d,[.5,.9,.99,1]).tolist()}
            rows.append(row);print(json.dumps(row),flush=True)
            a.out.write_text(json.dumps(rows,indent=2)+'\n')

if __name__=='__main__':main()
