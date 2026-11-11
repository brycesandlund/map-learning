"""Fixed-replay fitting interventions on identical saved search archives.

Both branches use frozen models during subsequent exploration. Oracle geometry is
used only for diagnostics. These are causal fixed-archive probes, not online arms.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from map_learning.training import Trainer,seed_all
from map_learning.replay import Replay
from map_learning.maze import Maze,reachable_graph
from map_learning.persistent_frontier import PersistentDistanceFrontier
from audit_frontier_efficiency import reconstruct,encode,priority_stats
from noisy_frontier import diagnostics


def run(source,out,block,updates,steps,zero_noise=False):
    out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    t=Trainer.load(source/f'checkpoint-{block}.pt');cfg=t.config;prefix=5000*block
    data=np.load(source/'replay.npz');obs,pos,vo,vp=reconstruct(data,prefix)
    replay=Replay()
    for name in ['observations','next_observations','actions','episodes','positions','next_positions']:
        setattr(replay,name,list(data[name][:prefix]))
    replay.arrays()
    env=Maze(cfg.maze,cfg.seed+89000+prefix);points,adj,index=reachable_graph(env)
    ids=lambda ps:np.array([index[tuple(np.round(p,5))] for p in ps])
    seen=np.zeros(len(points),bool);seen[0]=True;seen[ids(data['next_positions'][:prefix])]=True
    initial=int(seen.sum());productive=(~seen[adj]).any(1)
    expanded=np.zeros(len(points),bool);expanded[ids(vp)]=True
    before=diagnostics(t,replay,points)
    for _ in range(updates):t.update(replay)
    after=diagnostics(t,replay,points);t.save(out/'fitted.pt')
    q=PersistentDistanceFrontier(len(obs)+steps+len(vo)+1,cfg.model.latent_dim,env.observation_shape,np.random.default_rng(cfg.seed+88000+prefix))
    q.n_visited=len(vo);q.visited_observations[:len(vo)]=vo
    q.entries=list(pos.copy());q.observations[:len(obs)]=obs;q.active[:len(obs)]=True
    def policy_encode(x):
        if zero_noise:
            x=x.copy();x[:,2:]=0
        return encode(t,x)
    q.refresh(policy_encode)
    rank=priority_stats(q.priority[:len(obs)],ids(pos),productive,expanded)
    first_full=None;repeat=0;rows=[];branch=Replay()
    for expansion in range(steps//5):
        o,position,z,score=q.pop();key=index[tuple(np.round(position,5))]
        repeat+=bool(expanded[key]);expanded[key]=True;q.visit(o,z)
        for action in range(5):
            env.position=position.copy();nxt,_=env.step(action)
            q.add(nxt,policy_encode(nxt[None])[0],env.position.copy())
            branch.add(o,action,nxt,len(branch),position,env.position)
            seen[index[tuple(np.round(env.position,5))]]=True
            if seen.all() and first_full is None:first_full=len(branch)
        if (expansion+1)%100==0:
            rows.append({'actions':len(branch),'visited':int(seen.sum()),'repeat_fraction':repeat/(expansion+1)})
    result={'source':str(source),'seed':cfg.seed,'prefix':prefix,'extra_updates':updates,'zero_noise':zero_noise,'actions':len(branch),
        'initial_visited':initial,'final_visited':int(seen.sum()),'new_states':int(seen.sum())-initial,
        'first_full_additional_actions':first_full,'before':before,'after':after,'starting_priority':rank,
        'rows':rows,'seconds':time.monotonic()-started}
    branch.save(out/'branch.npz',cfg.maze)
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['rows','before','after','starting_priority']}),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--zero-only',action='store_true');a=p.parse_args();seed_all(0,1)
    a.out.mkdir(parents=True,exist_ok=False)
    for seed,block in [(0,10),(1,10),(2,10),(2,20)]:
        source=Path(f'runs/persistent-frontier-20261110/interleaved100-seed{seed}')
        for updates in ([0] if a.zero_only else [0,2000]):run(source,a.out/f'seed{seed}-block{block}-extra{updates}',block,updates,10000,a.zero_only)

if __name__=='__main__':main()
