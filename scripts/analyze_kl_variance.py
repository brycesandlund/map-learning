"""Read-only checkpoint analysis of posterior noise, mean spread and prior mismatch."""
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from map_learning.training import Trainer,seed_all


def main():
    seed_all(0,1)
    rows=json.loads(Path('docs/assets/kl-repulsion-audit.json').read_text())
    out=[]
    for r in rows:
        d=Path(r['path']);t=Trainer.load(d/'checkpoint.pt');rng=np.random.default_rng(r['seed']+84000)
        with np.load(d/'replay.npz') as data:
            n=len(data['observations']);idx=rng.integers(2*n,size=2048)
            obs=np.where((idx<n)[:,None],data['observations'][idx%n],data['next_observations'][idx%n])
        with torch.no_grad():
            mu,ls=t.posterior(t.tensor(obs));mu=mu.double();ls=ls.double()
            qvar=(2*ls).exp().mean(0);mvar=mu.var(0,unbiased=False);aggvar=mvar+qvar
            fitmean=mu.mean(0);fitls=.5*aggvar.log()
            klfit=(fitls-ls.mean(0)).sum()
            pm=t.prior.means[0].double();pls=t.prior.log_std[0].double().clamp(-12,2)
            gap=(pls-fitls+.5*((2*(fitls-pls)).exp()+(fitmean-pm).square()*(-2*pls).exp()-1)).sum()
            kl=(pls-ls+.5*((2*(ls-pls)).exp()+(mu-pm).square()*(-2*pls).exp()-1)).sum(-1).mean()
            torch.testing.assert_close(kl,klfit+gap,rtol=1e-9,atol=1e-9)
            np.testing.assert_allclose(float(kl),r['final']['bottleneck']['kl_nats'],rtol=.003,atol=1e-5)
        out.append({**{k:r[k] for k in ['path','repulsion','beta','seed']},'kl_nats':float(kl),
                    'moment_matched_kl':float(klfit),'prior_mismatch_kl':float(gap),
                    'posterior_variance':float(qvar.mean()),'mean_variance':float(mvar.mean()),
                    'posterior_std':float(ls.exp().mean()),'sampling_to_mean_variance':float(qvar.mean()/mvar.mean()),
                    'posterior_variance_per_dim':qvar.tolist(),'mean_variance_per_dim':mvar.tolist(),
                    'prior_std':pls.exp().tolist(),'aggregate_std':aggvar.sqrt().tolist()})
    fig,axes=plt.subplots(1,3,figsize=(12,3.5),layout='constrained')
    for ax,seed in zip(axes,[3,4,5]):
        for rep,beta,label,color in [(.01,1e-6,'Previous best','#984ea3'),(.03,1e-5,'Joint winner','#e27f24')]:
            r=next(r for r in out if r['seed']==seed and r['repulsion']==rep and r['beta']==beta)
            ratio=np.array(r['mean_variance_per_dim'])/np.array(r['posterior_variance_per_dim'])
            ax.plot(range(1,9),np.sort(ratio)[::-1],'-o',label=label,color=color)
        ax.axhline(1,color='gray',linestyle='--')
        ax.set(title=f'Confirmation seed {seed}',xlabel='Coordinate rank (sorted separately)',ylabel='Mean variance / sampling variance',yscale='log',xticks=range(1,9));ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.savefig('docs/assets/kl-coordinate-variance.png',dpi=160);plt.close(fig)
    Path('docs/assets/kl-variance-diagnostics.json').write_text(json.dumps(out,indent=2)+'\n')
    groups=defaultdict(list)
    for r in out:groups[r['repulsion'],r['beta'],'confirm' if r['seed']>=3 else 'screen'].append(r)
    for k,rr in sorted(groups.items()):
        print(k,{f:round(float(np.mean([r[f] for r in rr])),8) for f in ['kl_nats','moment_matched_kl','prior_mismatch_kl','posterior_variance','mean_variance','posterior_std','sampling_to_mean_variance']},flush=True)
    conditions=[(.01,0,'β=0 baseline','#377eb8'),(.01,1e-6,'Previous best','#984ea3'),(.03,1e-5,'Joint winner','#e27f24'),(.01,1e-4,'Collapsed','#b34a42'),(1,1e-4,'Noise-dominated','#666666')]
    metrics=[('KL to learned prior (nats/observation)','kl_nats'),('Mean posterior standard deviation','posterior_std_mean'),('Variance of encoder means','mean_latent_variance'),('Sampling variance / mean variance','sample_noise_to_mean_variance'),('Unweighted repulsion loss','repulsion'),('Observation-noise / movement distance','noise_to_move_ratio')]
    fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    for rep,beta,label,color in conditions:
        rr=[r for r in rows if r['repulsion']==rep and r['beta']==beta and r['seed']<3]
        logs=[[json.loads(x) for x in (Path(r['path'])/'blocks.jsonl').read_text().splitlines()][1:] for r in rr]
        x=np.array([b['interactions']/1000 for b in logs[0]])
        for ax,(title,metric) in zip(axes.flat,metrics):
            if metric=='kl_nats' and beta==0:continue
            values=np.array([[b['bottleneck'][metric] if metric in b['bottleneck'] else b[metric] for b in log] for log in logs])
            ax.plot(x,values.mean(0),label=f'{label}: α={rep:g}, β={beta:g}',color=color)
            ax.fill_between(x,values.min(0),values.max(0),color=color,alpha=.10)
            ax.set(title=title,xlabel='Real actions (thousands)',yscale='log');ax.grid(alpha=.2)
    handles,labels=axes[0,1].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=3,fontsize=9)
    fig.suptitle('KL / variance trajectories: screening seeds 0–2; mean and seed range\nβ=0 prior is inactive and omitted from the KL panel')
    fig.savefig('docs/assets/kl-variance-trajectories.png',dpi=155);plt.close(fig)

if __name__=='__main__':main()
