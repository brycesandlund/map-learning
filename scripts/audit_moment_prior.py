"""Independent run, exact-prior, stochastic-control and frontier checks."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree
from map_learning.training import Trainer,seed_all
from audit_frontier_efficiency import reconstruct,encode


def full_moments(t,data,count):
    # Independent sum/second-moment accumulation; trainer uses central moments.
    sums=np.zeros(8);seconds=np.zeros(8);qsum=np.zeros(8);lssum=np.zeros(8);n=0
    with torch.no_grad():
        for name in ['observations','next_observations']:
            for start in range(0,count,4096):
                mu,ls=t.posterior(t.tensor(data[name][start:min(start+4096,count)]))
                mu=mu.numpy().astype('float64');ls=ls.numpy().astype('float64')
                sums+=mu.sum(0);seconds+=(mu*mu).sum(0);qsum+=np.exp(2*ls).sum(0);lssum+=ls.sum(0);n+=len(mu)
    mean=sums/n;mv=seconds/n-mean*mean;var=mv+qsum/n;fitls=.5*np.log(var)
    pm=t.prior.means[0].detach().numpy().astype('float64');pl=t.prior.log_scales()[0].detach().numpy().astype('float64')
    gap=np.sum(pl-fitls+.5*(np.exp(2*(fitls-pl))+(mean-pm)**2*np.exp(-2*pl)-1))
    matched=np.sum(fitls-lssum/n)
    return dict(mean=mean.tolist(),variance=var.tolist(),mean_variance=float(mv.mean()),posterior_variance=float((qsum/n).mean()),
                matched_kl=float(matched),prior_mismatch=float(gap),actual_kl=float(matched+gap),
                prior_mean_max_error=float(np.max(np.abs(pm-mean))),prior_variance_max_relative_error=float(np.max(np.abs(np.exp(2*pl)/var-1))))


def main():
    p=argparse.ArgumentParser();p.add_argument('--roots',nargs='+',required=True)
    p.add_argument('--out',type=Path,default=Path('docs/assets/moment-prior-audit.json'))
    p.add_argument('--completed-only',action='store_true');p.add_argument('--resume',action='store_true');a=p.parse_args()
    seed_all(0,1);jobs=[j for root in a.roots for j in json.loads((Path(root)/'manifest.json').read_text())['jobs']]
    refs={}
    for j in sorted(jobs,key=lambda j:j['beta']!=0):refs.setdefault(j['seed'],Path(j['path']))
    cache={r['path']:r for r in json.loads(a.out.read_text())} if a.resume and a.out.exists() else {}
    rows=[];done=set()
    for j in jobs:
        d=Path(j['path'])
        if str(d) in done:continue
        done.add(str(d))
        if a.completed_only and not (d/'result.json').exists():continue
        if str(d) in cache:rows.append(cache[str(d)]);continue
        r=json.loads((d/'result.json').read_text());cfg=json.loads((d/'config.json').read_text())
        assert cfg['train']['repulsion_weight']==j['repulsion'] and r['bottleneck_spec']['beta']==j['beta']
        if not j['reused']:
            assert r['bottleneck_spec'].get('prior_mode','learned')==j['prior_mode']
            assert r['bottleneck_spec'].get('kl_warmup_updates',0)==j.get('warmup_updates',0)
        assert r['updates']==4112 and r['interactions']==100000 and r['samples_per_head']==1050000
        base=refs[j['seed']];old=Trainer.load(base/'checkpoint-0.pt');initial=Trainer.load(d/'checkpoint-0.pt')
        for key,v in old.model.state_dict().items():torch.testing.assert_close(v,initial.model.state_dict()[key],rtol=0,atol=0)
        for key,v in old.dynamics.state_dict().items():torch.testing.assert_close(v,initial.dynamics.state_dict()[key],rtol=0,atol=0)
        data=np.load(d/'replay.npz')
        with np.load(base/'replay.npz') as b:
            np.testing.assert_array_equal(data['next_observations'][:,2:],b['next_observations'][:,2:])
            for key in ['observations','actions','next_observations','positions','next_positions']:
                np.testing.assert_array_equal(data[key][:500],b[key][:500])
        if j.get('warmup_updates',0):
            assert j['warmup_updates']==651 and j['repulsion']==.03
            with np.load(f'runs/kl-repulsion-intermediate-20261110/rep0.03-beta0-seed{j["seed"]}/replay.npz') as b:
                for key in ['observations','actions','next_observations','positions','next_positions']:
                    np.testing.assert_array_equal(data[key][:35000],b[key][:35000])
        if j.get('role')=='zero_reproduction':
            oldpath=Path('runs/kl-repulsion-intermediate-20261110/rep0.03-beta0-seed0')
            with np.load(oldpath/'replay.npz') as b:
                for key in ['observations','actions','next_observations','positions','next_positions']:
                    np.testing.assert_array_equal(data[key],b[key])
            original=Trainer.load(oldpath/'checkpoint.pt');fresh=Trainer.load(d/'checkpoint.pt')
            for key,v in original.model.state_dict().items():torch.testing.assert_close(v,fresh.model.state_dict()[key],rtol=0,atol=0)
            for key,v in original.dynamics.state_dict().items():torch.testing.assert_close(v,fresh.dynamics.state_dict()[key],rtol=0,atol=0)
        checks=[]
        for block in [10,20]:
            t=Trainer.load(d/f'checkpoint-{block}.pt');obs,pos,vo,vp=reconstruct(data,block*5000)
            rng=t.sample_rng.get_state().clone();z=encode(t,obs);v=encode(t,vo)
            assert torch.equal(rng,t.sample_rng.get_state());scores=cKDTree(v).query(z)[0]
            if block==20:
                with np.load(d/'frontier.npz') as f:
                    np.testing.assert_array_equal(obs,f['observations']);np.testing.assert_array_equal(z,f['latents'])
                    np.testing.assert_allclose(scores,f['priorities'],rtol=1e-12,atol=1e-12)
            moments=full_moments(t,data,block*5000)
            if t.spec['prior_mode']=='replay_moments':
                assert moments['prior_mean_max_error']<1e-5 and moments['prior_variance_max_relative_error']<1e-5
                assert abs(moments['prior_mismatch'])<1e-8
                assert t.prior_fit_stats['calls']==11*block
                if block==20:assert t.prior_fit_stats['observations']==22200000
            moments.update(block=block);checks.append(moments)
        pr=[json.loads(x) for x in (d/'progress.jsonl').read_text().splitlines()]
        auc=float(np.trapezoid([1/1784]+[x['coverage'] for x in pr],[0]+[x['interactions'] for x in pr])/100000)
        train=[json.loads(x) for x in (d/'training.jsonl').read_text().splitlines()]
        gaps=[x['loss_metrics']['prior_mismatch_batch'] for x in train if 'prior_mismatch_batch' in x.get('loss_metrics',{})]
        gaps_late=[x['loss_metrics']['prior_mismatch_batch'] for x in train if x['interactions']>=35000 and 'prior_mismatch_batch' in x.get('loss_metrics',{})]
        rows.append({**j,'coverage_auc':auc,'visited':r['visited'],'first_full':r['first_full'],'milestones':r['milestones'],
                     'final':r['blocks'][-1],'checkpoints':checks,'checks_passed':True,
                     'last_batch_prior_gap':dict(median=float(np.median(gaps)),maximum=float(max(gaps)),late_maximum=float(max(gaps_late))) if gaps else None})
        a.out.write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps({**j,'auc':auc,'visited':r['visited'],'full_prior_gap':checks[-1]['prior_mismatch']}),flush=True)
    a.out.write_text(json.dumps(rows,indent=2)+'\n')

if __name__=='__main__':main()
