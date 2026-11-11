"""Audit completed joint sweeps against same-seed stochastic controls."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree
from map_learning.training import Trainer,seed_all
from map_learning.maze import Maze,reachable_graph
from audit_frontier_efficiency import reconstruct,encode,priority_stats


def main():
    p=argparse.ArgumentParser();p.add_argument('--roots',nargs='+',required=True)
    p.add_argument('--out',type=Path,default=Path('docs/assets/kl-repulsion-audit.json'))
    p.add_argument('--completed-only',action='store_true');p.add_argument('--resume',action='store_true');a=p.parse_args()
    seed_all(0,1)
    jobs=[j for root in a.roots for j in json.loads((Path(root)/'manifest.json').read_text())['jobs']]
    refs={j['seed']:Path(j['path']) for j in jobs if j['repulsion']==.01 and j['beta']==0}
    cached={r['path']:r for r in json.loads(a.out.read_text())} if a.resume and a.out.exists() else {}
    rows=[];graph=None;done=set()
    for j in jobs:
        d=Path(j['path'])
        if str(d) in done:continue
        done.add(str(d))
        if a.completed_only and not (d/'result.json').exists():continue
        if str(d) in cached:
            assert cached[str(d)]['checks_passed'];rows.append(cached[str(d)]);continue
        r=json.loads((d/'result.json').read_text());cfg=json.loads((d/'config.json').read_text())
        assert cfg['train']['repulsion_weight']==j['repulsion'] and r['bottleneck_spec']['beta']==j['beta']
        assert r['updates']==4112 and r['interactions']==100000 and r['samples_per_head']==1050000
        base=refs[j['seed']]
        original=Trainer.load(base/'checkpoint-0.pt');initial=Trainer.load(d/'checkpoint-0.pt')
        for name,value in original.model.state_dict().items():torch.testing.assert_close(value,initial.model.state_dict()[name],rtol=0,atol=0)
        for name,value in original.dynamics.state_dict().items():torch.testing.assert_close(value,initial.dynamics.state_dict()[name],rtol=0,atol=0)
        data=np.load(d/'replay.npz')
        with np.load(base/'replay.npz') as b:
            np.testing.assert_array_equal(data['next_observations'][:,2:],b['next_observations'][:,2:])
            for key in ['observations','actions','next_observations','positions','next_positions']:
                np.testing.assert_array_equal(data[key][:500],b[key][:500])
        checks=[]
        for block in [10,20]:
            t=Trainer.load(d/f'checkpoint-{block}.pt')
            if graph is None:graph=reachable_graph(Maze(t.config.maze))
            points,adj,index=graph
            ids=lambda ps:np.array([index[tuple(np.round(x,5))] for x in ps])
            obs,pos,vo,vp=reconstruct(data,block*5000)
            seen=np.zeros(len(points),bool);seen[ids(data['positions'][:1])]=True;seen[ids(data['next_positions'][:block*5000])]=True
            expanded=np.zeros(len(points),bool);expanded[ids(vp)]=True
            rng=t.sample_rng.get_state().clone();z=encode(t,obs);v=encode(t,vo)
            assert torch.equal(rng,t.sample_rng.get_state())
            scores=cKDTree(v).query(z)[0]
            if block==20:
                with np.load(d/'frontier.npz') as f:
                    np.testing.assert_array_equal(obs,f['observations']);np.testing.assert_array_equal(z,f['latents'])
                    np.testing.assert_allclose(scores,f['priorities'],rtol=1e-12,atol=1e-12)
            stats=priority_stats(scores,ids(pos),(~seen[adj]).any(1),expanded)
            stats.update(block=block,seen_states=int(seen.sum()));checks.append(stats)
        pr=[json.loads(x) for x in (d/'progress.jsonl').read_text().splitlines()]
        auc=float(np.trapezoid([1/1784]+[x['coverage'] for x in pr],[0]+[x['interactions'] for x in pr])/100000)
        rows.append({**j,'coverage_auc':auc,'visited':r['visited'],'first_full':r['first_full'],
                     'milestones':r['milestones'],'final':r['blocks'][-1],'checkpoints':checks,'checks_passed':True})
        a.out.write_text(json.dumps(rows,indent=2)+'\n')
        print(json.dumps({**j,'auc':auc,'visited':r['visited']}),flush=True)

    a.out.write_text(json.dumps(rows,indent=2)+'\n')

if __name__=='__main__':main()
