"""Prior/representation trajectories and early prior-scale diagnostic; no training."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from audit_moment_prior import full_moments
from map_learning.training import Trainer,seed_all


def main():
    seed_all(0,1)
    conditions=[('Learned prior, α=.03 β=1e-5','runs/kl-repulsion-intermediate-20261110/rep0.03-beta1e-05-seed{}','#377eb8'),
      ('Replay fit, same α/β, from start','runs/moment-prior-20261110/replay_moments-rep0.03-beta1e-05-seed{}','#b34a42'),
      ('Replay fit, same α/β, delayed 35k','runs/moment-prior-delayed-20261110/replay_moments-rep0.03-beta1e-05-seed{}','#57a65a'),
      ('Replay fit, α=.03 β=1e-4, delayed 35k','runs/moment-prior-delayed-strong-20261110/replay_moments-rep0.03-beta0.0001-seed{}','#984ea3'),
      ('Replay fit, α=.01 β=1e-7','runs/moment-prior-fine-20261110/replay_moments-rep0.01-beta1e-07-seed{}','#e27f24')]
    metrics=[('KL to current prior (nats)','kl_nats',True),('Mean-latent variance','mean_latent_variance',True),
             ('Mean posterior standard deviation','posterior_std_mean',True),('Noise / movement distance','noise_to_move_ratio',True),
             ('Physical coverage (%)','coverage',False),('Normalized prediction MSE','seen_normalized_mse',True)]
    fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    for label,pattern,color in conditions:
        logs=[[json.loads(x) for x in (Path(pattern.format(s))/'blocks.jsonl').read_text().splitlines()][1:] for s in range(3)]
        assert all(len(x)==20 for x in logs)
        for ax,(title,key,log) in zip(axes.flat,metrics):
            v=np.array([[b['bottleneck'][key] if key in b['bottleneck'] else b[key] for b in bs] for bs in logs])
            if key=='coverage':v*=100
            ax.plot(np.arange(1,21)*5,v.mean(0),label=label,color=color)
            ax.fill_between(np.arange(1,21)*5,v.min(0),v.max(0),color=color,alpha=.10)
            ax.set(title=title,xlabel='Real actions (thousands)',yscale='log' if log else 'linear');ax.grid(alpha=.2)
            ax.axvline(35,color='gray',linestyle=':',linewidth=.7)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=2,fontsize=8)
    fig.suptitle('Prior-fit and timing diagnostics: screening seeds 0–2, mean and seed range\nDotted line = delayed KL activation; prior is fitted throughout those runs')
    fig.savefig('docs/assets/moment-prior-dynamics.png',dpi=155);plt.close(fig)
    initial=[]
    for seed in range(3):
        d=Path(f'runs/kl-gaussian-20261110/beta0-seed{seed}');t=Trainer.load(d/'checkpoint-0.pt')
        with np.load(d/'replay.npz') as data:m=full_moments(t,data,500)
        initial.append(dict(seed=seed,count=1000,learned_prior_initial_std=.05,fitted_prior_std=np.sqrt(m['variance']).tolist(),
                            inverse_variance_factor=(.0025/np.array(m['variance'])).tolist()))
    Path('docs/assets/moment-prior-initial-scale.json').write_text(json.dumps(initial,indent=2)+'\n')

if __name__=='__main__':main()
