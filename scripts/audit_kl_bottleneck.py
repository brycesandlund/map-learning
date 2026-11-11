"""Independent matched-control, inference, and frontier-ranking checks for KL runs."""
import json
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree
from map_learning.training import Trainer,seed_all
from map_learning.maze import Maze,reachable_graph
from audit_frontier_efficiency import reconstruct,encode,priority_stats

ROOTS=['kl-gaussian-20261110','kl-mixture-20261110','kl-fast-prior-20261110','kl-warm-start-20261110','kl-late-start-20261110']


def main():
    seed_all(0,1);rows=[];graph=None
    paths=[p for root in ROOTS for p in sorted(Path('runs',root).glob('*/result.json'))]
    if len(paths)!=40:raise RuntimeError(f'Expected 40 completed new runs, found {len(paths)}')
    for path in paths:
        d=path.parent;r=json.loads(path.read_text());seed=r['seed']
        base=Path(f'runs/persistent-frontier-20261110/interleaved100-seed{seed}')
        original=Trainer.load(base/'checkpoint-0.pt');initial=Trainer.load(d/'checkpoint-0.pt')
        for name,value in original.model.encoder.state_dict().items():torch.testing.assert_close(value,initial.model.encoder.state_dict()[name],rtol=0,atol=0)
        for name,value in original.dynamics.state_dict().items():torch.testing.assert_close(value,initial.dynamics.state_dict()[name],rtol=0,atol=0)
        assert r['updates']==4112 and r['interactions']==100000 and r['samples_per_head']==1050000
        data=np.load(d/'replay.npz')
        with np.load(base/'replay.npz') as b:
            np.testing.assert_array_equal(data['next_observations'][:,2:],b['next_observations'][:,2:])
            for key in ['observations','actions','next_observations','positions','next_positions']:
                np.testing.assert_array_equal(data[key][:500],b[key][:500])
            if r.get('bottleneck_spec') is None:
                for key in data.files:np.testing.assert_array_equal(data[key],b[key])
        if (r.get('bottleneck_spec') or {}).get('kl_warmup_updates',0):
            prefix_check=35000 if r['bottleneck_spec']['kl_warmup_updates']==651 else 25000
            with np.load(Path(f'runs/kl-gaussian-20261110/beta0-seed{seed}/replay.npz')) as zero:
                for key in ['observations','next_observations','actions','positions','next_positions']:
                    np.testing.assert_array_equal(data[key][:prefix_check],zero[key][:prefix_check])
        final=Trainer.load(d/'checkpoint.pt')
        if hasattr(final,'sample_rng'):
            rng=final.sample_rng.get_state().clone()
            x=final.tensor(data['observations'][:100])
            with torch.no_grad():a=final.model.encode(x);b=final.model.encode(x)
            assert torch.equal(a,b) and torch.equal(rng,final.sample_rng.get_state())
        checkpoints=[]
        for block in [5,10,20]:
            t=Trainer.load(d/f'checkpoint-{block}.pt');obs,pos,vo,vp=reconstruct(data,block*5000)
            if graph is None:graph=reachable_graph(Maze(t.config.maze))
            points,adj,index=graph
            ids=lambda ps:np.array([index[tuple(np.round(p,5))] for p in ps])
            seen=np.zeros(len(points),bool);seen[0]=True;seen[ids(data['next_positions'][:block*5000])]=True
            expanded=np.zeros(len(points),bool);expanded[ids(vp)]=True
            z=encode(t,obs);v=encode(t,vo);scores=cKDTree(v).query(z)[0]
            if block==20:
                with np.load(d/'frontier.npz') as f:
                    np.testing.assert_array_equal(obs,f['observations']);np.testing.assert_array_equal(z,f['latents'])
                    np.testing.assert_allclose(scores,f['priorities'],rtol=1e-12,atol=1e-12)
            stats=priority_stats(scores,ids(pos),(~seen[adj]).any(1),expanded)
            stats.update(block=block,seen_states=int(seen.sum()))
            checkpoints.append(stats)
        pr=[json.loads(x) for x in (d/'progress.jsonl').read_text().splitlines()]
        auc=float(np.trapezoid([1/1784]+[x['coverage'] for x in pr],[0]+[x['interactions'] for x in pr])/100000)
        row={'path':str(d),'seed':seed,'spec':r.get('bottleneck_spec'),'coverage_auc':auc,'visited':r['visited'],
             'first_full':r['first_full'],'milestones':r['milestones'],'final':r['blocks'][-1],
             'checkpoints':checkpoints,'checks_passed':True}
        rows.append(row);print(json.dumps({'run':d.name,'root':d.parent.name,'visited':r['visited'],'auc':auc}),flush=True)
    # Two unchanged deterministic references supplement the reproduced seed-zero control.
    for seed in [1,2]:
        d=Path(f'runs/persistent-frontier-20261110/interleaved100-seed{seed}');r=json.loads((d/'result.json').read_text())
        pr=[json.loads(x) for x in (d/'progress.jsonl').read_text().splitlines()]
        auc=float(np.trapezoid([1/1784]+[x['coverage'] for x in pr],[0]+[x['interactions'] for x in pr])/100000)
        rows.append({'path':str(d),'seed':seed,'spec':None,'coverage_auc':auc,'visited':r['visited'],'first_full':r['first_full'],
                     'milestones':r['milestones'],'final':r['blocks'][-1],'checkpoints':[],'checks_passed':True,'reused_reference':True})
    Path('docs/assets/kl-bottleneck-audit.json').write_text(json.dumps(rows,indent=2)+'\n')

if __name__=='__main__':main()
