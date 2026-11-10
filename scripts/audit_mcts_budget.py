"""Inspect visit-selected MCTS paths at fixed checkpoints; no online feedback."""
from pathlib import Path
import json,math
import numpy as np
import torch
from map_learning.training import Trainer,seed_all
from map_learning.maze import Maze,reachable_graph
from map_learning.uncertainty_planning import UncertaintyPlanner,SearchNode
from map_learning.probabilistic import mixture_mean

class InspectPlanner(UncertaintyPlanner):
    # Exact current MCTS selection/backup inherited; retain root for diagnostics.
    def mcts(self,z):
        root=SearchNode(z,self.trainer.dynamics.actions);self.nodes_created=1;self.uct_scale=1.
        if self.mean_mode:
            self.prepare(root)
            self.uct_scale=max(float(root.rewards.max()),1e-12)*sum(self.config.uncertainty_discount**i for i in range(self.config.horizon))
        for _ in range(self.config.simulations):self.simulate(root,0)
        choices=np.flatnonzero(root.counts==root.counts.max());action=int(self.rng.choice(choices))
        self.root=root
        return action

@torch.no_grad()
def main():
 seed_all(0,1);root=Path('runs/mcts-budget-audit-20261110');root.mkdir(exist_ok=False);rows=[]
 for seed in range(3):
  run=Path(f'runs/horizon-duration-20261110/h16_shared_30k-seed{seed}');t=Trainer.load(run/'checkpoint.pt');env=Maze(t.config.maze)
  points,adj,index=reachable_graph(env);z=t.model.encode(t.tensor(np.stack([env.observe(p) for p in points])))
  data=dict(np.load(run/'replay.npz'));visited=np.zeros(len(points),bool)
  for p in np.concatenate([data['positions'],data['next_positions']]):visited[index[tuple(p)]]=True
  ids=np.random.default_rng(8171).choice(np.flatnonzero(visited),24,replace=False);variance=max(float(z.var(0,unbiased=False).mean()),1e-8)
  actual=[]
  for start in range(0,len(points),128):
   zz=z[start:start+128];means=mixture_mean(t.dynamics.distribution(zz.repeat_interleave(5,0),torch.arange(5).repeat(len(zz))))
   actual.extend(means.var(0,unbiased=False).mean(-1).reshape(-1,5).numpy()/variance)
  actual=np.asarray(actual)
  for budget in [24,96,384]:
   t.config.planner.simulations=budget
   planner=InspectPlanner(t,t.config.planner,seed+45000);planner.reward_variance=variance
   # Verify copied root-selection wrapper against production on one root.
   production=UncertaintyPlanner(t,t.config.planner,seed+45000);production.reward_variance=variance
   a=planner.mcts(z[ids[:1]]);b=production.mcts(z[ids[:1]]);assert a==b
   np.testing.assert_array_equal(planner.root.counts,production.last_metrics['root_visits'])
   np.testing.assert_allclose(planner.root.returns/np.maximum(planner.root.counts,1),production.last_metrics['root_q'])
   for source in ids:
    first=planner.mcts(z[source:source+1]);node=planner.root;real=int(source);seen={real};new=set();ig=rg=0.;repeat=0;stationary=0
    for depth in range(16):
     # Root follows actual selected action. Deeper PV ties use lowest action ID.
     action=first if depth==0 else int(node.counts.argmax());ig+=.99**depth*node.rewards[action];rg+=.99**depth*actual[real,action]
     nxt=int(adj[real,action]);stationary+=nxt==real;repeat+=nxt in seen
     if not visited[nxt]:new.add(nxt)
     seen.add(nxt);real=nxt
     if depth<15:node=node.children[action][0]
    rows.append({'seed':seed,'budget':budget,'start':int(source),'imagined_return':float(ig),'real_state_return':float(rg),
      'new_states':len(new),'repeated_successors':repeat,'stationary_steps':stationary,'unique_path_states':len(seen)})
   (root/'results.json').write_text(json.dumps(rows,indent=2)+'\n');print(seed,budget,flush=True)
 Path('docs/assets/mcts-budget-audit.json').write_text(json.dumps(rows,indent=2)+'\n')
if __name__=='__main__':main()
