"""Privileged perfect-dynamics MCTS diagnostic ONLY. Never used by online agents."""
from pathlib import Path
import json
import numpy as np
import torch
from map_learning.training import Trainer,seed_all
from map_learning.replay import Replay
from map_learning.maze import Maze,reachable_graph
from map_learning.novelty_planning import NoveltyPlanner
from analyze_novelty import frontier_distances

class Node:
 def __init__(self,state,history):
  self.state=state;self.history=history;self.counts=np.zeros(5);self.returns=np.zeros(5);self.children={}


def search(adj,distance,archive_distance,start,horizon,budget,scale,seed):
 rng=np.random.default_rng(seed);root=Node(start,[start]);discoveries=0;novel_simulations=0
 def simulate(node,depth):
  nonlocal discoveries
  if depth==horizon:return 0.
  untried=np.flatnonzero(node.counts==0)
  if len(untried):a=int(rng.choice(untried))
  else:
   scores=node.returns/node.counts+.3*scale*np.sqrt(np.log(node.counts.sum()+1)/node.counts)
   a=int(rng.choice(np.flatnonzero(scores==scores.max())))
  nxt=int(adj[node.state,a]);reward=min(archive_distance[nxt],distance[nxt,node.history].min())
  if archive_distance[nxt]>0:discoveries+=1
  if depth+1<horizon:
   if a not in node.children:node.children[a]=Node(nxt,node.history+[nxt])
   reward+=.99*simulate(node.children[a],depth+1)
  node.counts[a]+=1;node.returns[a]+=reward
  return reward
 for _ in range(budget):
  before=discoveries;simulate(root,0);novel_simulations+=discoveries>before
 action=int(rng.choice(np.flatnonzero(root.counts==root.counts.max())));node=root;found=set()
 for depth in range(horizon):
  a=action if depth==0 else int(node.counts.argmax());nxt=int(adj[node.state,a])
  if archive_distance[nxt]>0:found.add(nxt)
  if depth+1<horizon:node=node.children[a]
 return {'any_simulation_reaches_novelty':novel_simulations>0,'novel_simulations':novel_simulations,'preferred_path_new_states':len(found)}

@torch.no_grad()
def main():
 seed_all(0,1);rows=[]
 for seed in range(3):
  run=Path(f'runs/horizon-duration-20261110/h16_shared_30k-seed{seed}');t=Trainer.load(run/'checkpoint.pt');env=Maze(t.config.maze)
  points,adj,index=reachable_graph(env);z=t.model.encode(t.tensor(np.stack([env.observe(p) for p in points]))).numpy()
  distance=np.linalg.norm(z[:,None]-z[None,:],axis=-1);replay,_=Replay.load(run/'replay.npz');visited=np.zeros(len(points),bool)
  for p in replay.positions+replay.next_positions:visited[index[tuple(p)]]=True
  nearest=distance[:,visited].min(-1);planner=NoveltyPlanner(t,t.config.planner,'archive_path');planner.refresh(replay)
  start=index[tuple(env.start)];true_dist=int(frontier_distances(adj,visited)[start])
  for horizon in [16,32,64]:
   for budget in [24,96,384]:
    for trial in range(5):
     r=search(adj,distance,nearest,start,horizon,budget,planner.step_scale*sum(.99**i for i in range(horizon)),seed*1000+trial)
     r.update(seed=seed,horizon=horizon,simulations=budget,trial=trial,shortest_distance_to_novelty=true_dist);rows.append(r)
    print(seed,horizon,budget,flush=True)
 Path('docs/assets/oracle-novelty-search.json').write_text(json.dumps(rows,indent=2)+'\n')
if __name__=='__main__':main()
