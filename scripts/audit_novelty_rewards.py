"""Same candidate paths scored in imagination and reality, at fixed checkpoints."""
from pathlib import Path
import json
import numpy as np
import torch
from map_learning.training import Trainer,seed_all
from map_learning.maze import Maze,reachable_graph
from map_learning.replay import Replay
from map_learning.novelty_planning import NoveltyPlanner
from map_learning.probabilistic import mixture_mean
from analyze_novelty import top_choice,correlation,frontier_distances

@torch.no_grad()
def main():
 seed_all(0,1);out=Path('runs/novelty-candidate-audit-v2-20261110');out.mkdir(exist_ok=False);results=[]
 modes=['disagreement','root_distance','path_distance','archive_path','disagreement_path']
 for seed in range(3):
  run=Path(f'runs/horizon-duration-20261110/h16_shared_30k-seed{seed}');t=Trainer.load(run/'checkpoint.pt');env=Maze(t.config.maze)
  points,adj,index=reachable_graph(env);z=t.model.encode(t.tensor(np.stack([env.observe(p) for p in points])))
  replay,_=Replay.load(run/'replay.npz');planner=NoveltyPlanner(t,t.config.planner,'archive_path');planner.refresh(replay)
  visited=np.zeros(len(points),bool)
  for p in replay.positions+replay.next_positions:visited[index[tuple(p)]]=True
  archive=z[visited];true_ad=torch.cdist(z,archive,compute_mode='donot_use_mm_for_euclid_dist').min(-1).values
  # Ground-truth catalogue rewards are evaluated only, never used by online agents.
  actual_dis=[]
  for start in range(0,len(points),128):
   state=z[start:start+128];means=mixture_mean(t.dynamics.distribution(state.repeat_interleave(5,0),torch.arange(5).repeat(len(state))))
   actual_dis.extend(means.var(0,unbiased=False).mean(-1).reshape(-1,5).numpy())
  actual_dis=np.asarray(actual_dis);rng=np.random.default_rng(9172);starts=32;candidates=64
  initial=rng.choice(np.flatnonzero(visited),starts,replace=False);initial=np.repeat(initial,candidates)
  real=initial.copy();latent=z[real].clone();origin=latent.clone();ih=[latent];rh=[z[real]]
  seen_path=np.zeros((len(real),len(points)),bool);seen_path[np.arange(len(real)),real]=True
  novelty=np.zeros(len(real));repeats=np.zeros(len(real));stepscale=planner.step_scale
  scores={f'{kind}_{mode}':np.zeros(len(real)) for kind in ['imagined','real'] for mode in modes}
  previous=rng.integers(5,size=len(real));rows=[]
  false_archive_mass=np.zeros(len(real))
  for depth in range(1,65):
   action=np.where(rng.random(len(real))<.8,previous,rng.integers(5,size=len(real)));previous=action
   means=mixture_mean(t.dynamics.distribution(latent,torch.as_tensor(action)));pred=means.mean(0)
   nxt=adj[real,action];actual=z[nxt]
   ip=(pred[:,None]-torch.stack(ih,dim=1)).norm(dim=-1).min(-1).values
   rp=(actual[:,None]-torch.stack(rh,dim=1)).norm(dim=-1).min(-1).values
   ia=torch.cdist(pred,archive,compute_mode='donot_use_mm_for_euclid_dist').min(-1).values
   duplicate=seen_path[np.arange(len(real)),nxt]
   false_archive_mass+=.99**(depth-1)*torch.minimum(ip,ia).numpy()*(visited[nxt]|duplicate)
   dis=means.var(0,unbiased=False).mean(-1);rd=torch.as_tensor(actual_dis[real,action])
   for kind,values in [('imagined',[dis,(pred-origin).norm(dim=-1),ip,torch.minimum(ip,ia),dis*ip/(ip+stepscale)]),
                       ('real',[rd,(actual-origin).norm(dim=-1),rp,torch.minimum(rp,true_ad[nxt]),rd*rp/(rp+stepscale)])]:
    for mode,value in zip(modes,values):scores[f'{kind}_{mode}']+=.99**(depth-1)*value.numpy()
   duplicate=seen_path[np.arange(len(real)),nxt];novelty+=(~visited[nxt])&~duplicate;repeats+=duplicate
   seen_path[np.arange(len(real)),nxt]=True;real=nxt;latent=pred;ih.append(pred);rh.append(actual)
   if depth not in [8,16,32,64]:continue
   n=novelty.reshape(starts,candidates);repeated=repeats.reshape(starts,candidates);eligible=n.max(1)>0
   nn=torch.cdist(pred,z,compute_mode='donot_use_mm_for_euclid_dist');catdist=nn.min(-1).values.numpy().reshape(starts,candidates)/stepscale
   correct=(nn.argmin(-1).numpy()==nxt).reshape(starts,candidates)
   row={'horizon':depth,'starts':starts,'candidates':candidates,'random_new_states':float(n.mean()),'best_new_states':float(n.max(1).mean()),
        'starts_with_novel_candidate':int(eligible.sum()),'scores':{}}
   for key,score in scores.items():
    values=score.reshape(starts,candidates);corr=[correlation(x,y) for x,y in zip(values,n)];corr=[x for x in corr if x is not None]
    mode=key.split('_',1)[1];true=scores[f'real_{mode}'].reshape(starts,candidates)
    row['scores'][key]={'new_states':float(top_choice(values,n).mean()),'repeat_fraction':float(top_choice(values,repeated/depth).mean()),
     'within_start_correlation':float(np.mean(corr)) if corr else None,
     'zero_novelty_given_opportunity':float(top_choice(values[eligible],n[eligible]==0).mean()),
     'selected_real_reward':float(top_choice(values,true).mean()),'selected_imagined_reward':float(top_choice(values,scores[f'imagined_{mode}'].reshape(starts,candidates)).mean()),
     'imagined_archive_mass_on_non_new_steps':float(top_choice(values,false_archive_mass.reshape(starts,candidates)).mean()/max(top_choice(values,scores['imagined_archive_path'].reshape(starts,candidates)).mean(),1e-12)),
     'endpoint_retrieval_accuracy':float(top_choice(values,correct).mean()),'endpoint_catalogue_distance_edge_units':float(top_choice(values,catdist).mean())}
   rows.append(row)
  result={'seed':seed,'source':str(run),'rows':rows};results.append(result);(out/f'seed{seed}.json').write_text(json.dumps(result,indent=2));print(seed,flush=True)
 Path('docs/assets/novelty-candidate-results.json').write_text(json.dumps(results,indent=2)+'\n')
if __name__=='__main__':main()
