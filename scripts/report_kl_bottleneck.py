"""Report all bottleneck arms, including failed settings and adaptive follow-ups."""
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,PillowWriter
from map_learning.maze import Maze
from map_learning.config import load_config


def label(r):
    s=r['spec']
    if s is None:return 'Deterministic'
    if s.get('kl_warmup_updates',0):return 'Gaussian β=1e-4, start '+('35k' if s['kl_warmup_updates']==651 else '25k')
    name='Mixture-4' if s['prior_components']==4 else 'Gaussian'
    if s.get('prior_lr_multiplier',1)>1:name+=' (10× prior LR)'
    return f'{name} β={s["beta"]:g}'


def main():
    rows=json.loads(Path('docs/assets/kl-bottleneck-audit.json').read_text());assets=Path('docs/assets')
    groups=defaultdict(list)
    for r in rows:groups[label(r)].append(r)
    def order(item):
        spec=item[1][0]['spec']
        if spec is None:return (0,0)
        if spec.get('kl_warmup_updates',0):return (4,spec['kl_warmup_updates'])
        return (3 if spec.get('prior_lr_multiplier',1)>1 else 2 if spec['prior_components']==4 else 1,spec['beta'])
    groups=dict(sorted(groups.items(),key=order))
    for g in groups.values():g.sort(key=lambda r:r['seed']);assert len(g)==3
    zero=groups['Gaussian β=0'];det=groups['Deterministic']
    positives={k:g for k,g in groups.items() if g[0]['spec'] and g[0]['spec']['beta']>0}
    best=max(positives,key=lambda k:np.mean([r['coverage_auc'] for r in positives[k]]))
    table=[]
    for name,g in groups.items():
        finished=sum(r['first_full'] is not None for r in g)
        coverage=100*np.mean([r['visited']/1784 for r in g]);auc=100*np.mean([r['coverage_auc'] for r in g])
        times=' / '.join(f'{r["first_full"]:,}' if r['first_full'] is not None else '>100k' for r in g)
        table.append(f'| {name} | {coverage:.1f}% | {auc:.1f}% | {finished}/3 | {times} |')
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for components,fast,color,style,name in [(1,False,'#2775b4','-','Gaussian'),(4,False,'#2c9560','--','Mixture-4'),(1,True,'#b76c31',':','Gaussian, 10× prior LR')]:
        selected=[r for r in rows if r['spec'] and r['spec']['prior_components']==components and (r['spec'].get('prior_lr_multiplier',1)>1)==fast and not r['spec'].get('kl_warmup_updates',0)]
        betas=sorted({r['spec']['beta'] for r in selected})
        for ax,metric in zip(axes.flat,['auc','coverage','posterior_std_mean','mean_latent_variance']):
            means=[]
            for b in betas:
                rr=[r for r in selected if r['spec']['beta']==b]
                y=[100*r['coverage_auc'] if metric=='auc' else 100*r['visited']/1784 if metric=='coverage' else r['final']['bottleneck'][metric] for r in rr]
                ax.scatter([b]*len(y),y,s=18,color=color,alpha=.45);means.append(np.mean(y))
            ax.plot(betas,means,style+'o',color=color,label=name)
    for ax in axes.flat:ax.set_xscale('symlog',linthresh=1e-6);ax.set_xlabel('β (KL is summed across latent dimensions)');ax.grid(alpha=.2)
    axes[0,0].set(title='Interaction efficiency',ylabel='Mean coverage over the 100k-action run (%)',ylim=(0,100))
    axes[0,0].axhline(100*np.mean([r['coverage_auc'] for r in det]),color='gray',label='Deterministic mean',lw=1)
    axes[0,0].legend(fontsize=8)
    axes[0,1].set(title='Final physical coverage',ylabel='Reachable positions observed (%)',ylim=(0,101))
    axes[1,0].set(title='Learned encoder sampling noise',ylabel='Mean posterior standard deviation',yscale='log')
    axes[1,0].axhline(.01,color='gray',lw=.7)
    axes[1,1].set(title='Spread of deterministic encoder means',ylabel='Variance across replay observations',yscale='log')
    fig.savefig(assets/'kl-bottleneck-sweep.png',dpi=160);plt.close(fig)
    selected_groups=['Deterministic','Gaussian β=0',best]
    fig,axes=plt.subplots(1,3,figsize=(14,4.2),constrained_layout=True)
    colors=['#777777','#2875b6','#d4822c']
    for name,color in zip(selected_groups,colors):
        for r in groups[name]:
            pr=[json.loads(x) for x in (Path(r['path'])/'progress.jsonl').read_text().splitlines()]
            axes[r['seed']].plot([x['interactions']/1000 for x in pr],[x['coverage']*100 for x in pr],label=name,color=color)
    for seed,ax in enumerate(axes):ax.set(title=f'Seed {seed}',xlabel='Real actions (thousands)',ylabel='Physical coverage (%)',ylim=(0,101));ax.grid(alpha=.2)
    axes[0].legend(fontsize=7);fig.suptitle('Baseline vs β=0 vs best positive-β mean AUC (selected after the sweep)')
    fig.savefig(assets/'kl-bottleneck-comparison.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(3,3,figsize=(12,12),constrained_layout=True);frames={};artists={}
    for row,name in enumerate(selected_groups):
        for r in groups[name]:
            seed=r['seed'];d=Path(r['path']);ax=axes[row,seed]
            Maze(load_config(d/'config.json').maze).draw(ax);f=np.load(d/'frames.npz');frames[row,seed]=f
            artists[row,seed]=(ax.scatter([],[],s=4,color=colors[row],alpha=.7),ax.scatter([],[],s=25,color='red',zorder=6),ax.set_title('',fontsize=10))
            ax.set_xticks([]);ax.set_yticks([])
    count=len(frames[0,0]['interactions'])
    def update(i):
        for row,name in enumerate(selected_groups):
            for seed in range(3):
                f=frames[row,seed];dots,current,title=artists[row,seed]
                dots.set_offsets(f['points'][f['visited'][i]]);current.set_offsets(f['position'][i:i+1]);title.set_text(f'{name} · seed {seed}\n{f["visited"][i].sum()}/1784 observed states')
        fig.suptitle(f'{frames[0,0]["interactions"][i]:,} real actions · deterministic means throughout search\nRed = simulator-restored expansion; positive-β condition selected post-sweep')
    animation=FuncAnimation(fig,update,frames=np.unique(np.linspace(0,count-1,100).astype(int)),interval=150)
    animation.save(assets/'kl-bottleneck.gif',writer=PillowWriter(fps=7),dpi=85)
    update(count-1);fig.savefig(assets/'kl-bottleneck-final.png',dpi=130);plt.close(fig)
    summary={'groups':{k:{'mean_coverage_auc':float(np.mean([r['coverage_auc'] for r in g])),
        'mean_final_coverage':float(np.mean([r['visited']/1784 for r in g])),
        'completed':sum(r['first_full'] is not None for r in g),'first_full':[r['first_full'] for r in g]} for k,g in groups.items()},'best_positive_auc_post_selection':best}
    (assets/'kl-bottleneck-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    qstd=[r['final']['bottleneck']['posterior_std_mean'] for r in zero]
    doc=f'''# Learned-prior KL bottleneck experiments

Completed November 10, 2026. **40 new full runs** (including one deterministic reproduction), plus two reused deterministic references: 14 conditions × three seeds. Same noisy maze, persistent nearest-visited-mean frontier, 100,000 actions and 4,112 gradient updates per run. No extra replay updates. This study adds 4,000,000 real actions and 164,480 joint updates; the separate small smoke trial is excluded.

## Findings

**A small positive coefficient helps on average, but does not eliminate nuisance sensitivity.** Gaussian β=1e-6 completes all three seeds at 52,472 / 69,202 / 74,798 actions, versus stochastic β=0 at 42,498 / 86,192 / 91,846. Mean completion cost falls from 73,512 to 65,491 actions (10.9%); two seeds improve and one slows down. Mean coverage over the 100k run improves from 54.6% to 60.3%. These three seeds are exploratory, not proof of a broadly optimal coefficient.

**Stochastic training itself matters.** β=0 completes all three seeds, but its mean coverage AUC 54.6% is slightly below the deterministic baseline 55.8%. It is not uniformly better than the old agent. Its posterior standard deviation shrinks about 15× from initialization, consistent with the prediction objective suppressing input sampling noise when there is no KL cost for doing so.

**Stronger from-scratch compression collapses useful mean variation.** At Gaussian β≥1e-4, final mean latent variance is around 1e-7, repulsion is nearly its maximum 1, posterior std remains about 0.04–0.05, and only 23–26% of the physical maze is observed on average. KL becomes nearly zero. This combination supports a collapse interpretation; shrinking scale alone would not establish collapse. β=1e-5 is intermediate: two seeds finish faster than their β=0 counterparts, but the third reaches only 1,337/1,784 positions.

**Additional prior flexibility does not win this sweep.** The four-component prior at β=1e-6 completes all seeds but is slower on average than the single Gaussian; at 1e-5 it succeeds in only one seed. Faster prior learning fails to rescue the stronger penalty and also collapses the 1e-5 runs. The early 25k warm start still prevents useful representation learning. The 35k switch preserves substantial variation but finishes only seed 0; final coverage is 1,784 / 994 / 937. These are separate failed hypotheses retained in the report, not discarded trials.

**The intended noise-removal mechanism is not established.** Final same-position-noise / neighboring-position distance ratios are 0.115 / 0.138 / 0.143 for β=0, versus 0.111 / 0.145 / 0.142 for β=1e-6. That is no consistent reduction. The best small-β arm is a promising optimization/representation change, but it has not “wiped noise from the means.” Noiseless mean search can still exploit distinctions that sampled-input dynamics cannot reliably use. Rank audits at 25k/50k/100k are included in the JSON; different online arms have different archives, so their rank fractions are not identical-data causal comparisons.

## All results

Mean coverage over the run is the area under the coverage-vs-actions curve divided by 100k, evaluated on the common 250-action logging grid. It rewards earlier exploration without pretending censored completion times are observed. Dots in the sweep figure are individual seeds; curves are three-seed means.

| Condition | Mean final coverage | Mean coverage over run | Full coverage | Actions to full coverage, seeds 0 / 1 / 2 |
|---|---:|---:|---|---|
{chr(10).join(table)}

![Sweep](assets/kl-bottleneck-sweep.png)

The best positive-β condition by mean coverage AUC is **{best}**, selected after inspecting this sweep. That selection is exploratory, not held-out validation. The main comparison is against **Gaussian β=0**, not just against the deterministic model: stochastic training itself changes optimization. All settings and unsuccessful runs are included above.

![Paired exploration curves](assets/kl-bottleneck-comparison.png)

![Exploration animation](assets/kl-bottleneck.gif)

## Exact learning setup

The existing encoder's final layer remains the deterministic mean μ(o), with width 256, three hidden layers, latent dimension eight. Add a linear head from its last hidden layer for observation-dependent log σ(o), initialized to log(0.01), weights zero; clamp log σ to [−12, 2] (minimum std about 6.1e-6). Original mean-encoder and dynamics initial weights are preserved exactly. A separate generator samples ε from a standard normal during training only; z=μ(o)+σ(o)⊙ε. The same sampled input latent is given to all three dynamics heads for each shared transition batch.

Loss:

`MSE(g_m(z_t,a_t), μ(o_next)) + 0.01 × mean max(0, 1 − ||μ(o_i)−μ(o_j)||²) + β × E_o KL(q(z|o) || p(z))`, where MSE averages squared coordinate errors over heads, batch entries, and latent dimensions.

Prediction targets and repulsion use means. Both prediction endpoints and both repulsion endpoints retain encoder gradients. KL uses a separate batch of replay observations (the same 512 observations used for repulsion) and is **summed across latent dimensions**, averaged across observations: nats per observation, not nats per dimension. Model/encoder learning rates remain 3e-4/9e-5. Adam jointly fits encoder and dynamics; an independent Adam fits prior parameters with gradients of βKL. β=0 leaves the prior inactive, so its reported KL to the initial reference is not a fitted information estimate.

Single-Gaussian prior: learned diagonal mean and standard deviation, initialized mean 0/std 0.05; analytic KL. Four-component prior: learned logits, means, diagonal scales, initial std 0.05 and small distinct random means; analytic posterior entropy plus four reparameterized samples for expected prior log density. Do not clamp Monte Carlo KL estimates to zero. Diagnostic mixture KL uses 32 isolated samples. Prior log std is clamped to [−12,2]. These are learned priors, not a requirement that the latent population stay standard normal.

**Agent/queue inference always uses μ(o).** No encoder sampling enters queue storage, nearest-neighbor indexing, priorities, or simulator interaction. Physical coordinates are only observation features, restoration payloads, and evaluation labels; no physical-ID deduplication. Observation noise and tie-breaking remain stochastic. All replay remains available; refresh every 100 expansions/500 executed actions, with one accumulated-replay epoch apportioned across every 1,000 expansions exactly as in the previous matched-budget study.

## Arms and adaptive decisions

Primary sweep: learned diagonal Gaussian with β=0,1e-6,1e-5,1e-4,1e-3,1e-2, all three seeds. Four-component mixtures test β=1e-6,1e-5,1e-4 on three seeds. No separate mixture β=0 run is needed to change the loss: an unused prior cannot affect the intended β=0 objective; the Gaussian β=0 is the stochastic baseline.

Two adaptive controls address observed failure modes:

1. **10× prior learning rate**, Gaussian β=1e-5/1e-4, three seeds: selected after early Gaussian runs showed a prior scale much smaller than latent spread. It tests prior adaptation speed, leaving the encoder/dynamics rates and update count unchanged.
2. **Delayed KL**, Gaussian β=1e-4: keep β effectively zero for 296 updates (first 25k actions), then activate it. At activation initialize the prior mean/variance from the current sampled replay aggregate: mean μ and Var(μ)+Eσ², detached. No additional environment data or gradient steps. This combines delayed compression with a causal prior initialization; it does not isolate those two changes separately. The 25k attempt revealed that β=0 had not yet learned strong invariance (probe noise/movement ratio 2.1–8.6), so a second three-seed delay to 651 updates/35k actions was added, when all three baseline ratios were 0.20–0.30. Activation first affects the next training event (25.5k/35.5k actions). The failed earlier switch is retained in the table.

## Diagnostics and limits

The β=0 posterior std shrinks from 0.01 to {min(qstd):.6f}–{max(qstd):.6f} on average across replay samples after training. This supports the expected pressure to remove unused sampling noise, but is not exact zero and does not imply the same optimization trajectory as deterministic training. Per-dimension posterior std and mean variance, prior weights/scales, sampled-input and mean-input prediction MSE, repulsion, observation-noise sensitivity, and KL are saved at every 5k actions.

Checkpoints at 25k/50k/100k are independently audited for productive-entry density, repeat-entry density, and nearest-visited distances. A productive entry is one whose real expansion would reveal an unobserved neighbor; these labels never affect training/search. Completed mazes naturally have zero productive entries, so final rank density is not compared as if all runs still had new states available. Mean-encoding scale alone does not establish collapse; inspect it together with repulsion, sampled noise/mean variance, and exploration.

Compression constrains sampled latent information, while the agent reads noiseless means. Residual distinctions can therefore remain visible to distance search even when they are inaccessible to noisy dynamics inputs. This experiment tests that mismatch empirically rather than assuming information compression guarantees exact state equivalence. Only a coordinate-plus-nuisance discrete maze with simulator restoration is tested. Three paired seeds on one layout are not broad validation; no claim of optimal β or statistical significance.

## Checks and reproduction

**70 tests pass:** analytic KL matches PyTorch distributions; mixture Monte Carlo agrees with identical-component Gaussian KL; gradients reach variance and prior parameters; initial mean/dynamics parameters match deterministic controls; mean inference consumes no sampling RNG; checkpoints reproduce the next update exactly; β=0 leaves the prior unchanged; delayed KL matches β=0 before activation. Existing tests still pass.

Independent run audits verify all initial mean/dynamics parameters, 500-action pre-training prefixes, complete successor-noise streams, action/update/exposure budgets, and every final frontier priority. Delayed runs match their β=0 data prefix through 25k/35k. The fresh deterministic seed 0 run reproduces the original replay exactly. Original deterministic seeds 1/2 are reused references. Legacy checkpoint loading remains supported; stochastic checkpoints dispatch through their saved bottleneck specification.

```sh
.venv/bin/python scripts/run_kl_bottleneck.py --out runs/kl-gaussian-NEW --deterministic-control
.venv/bin/python scripts/run_kl_bottleneck.py --out runs/kl-mixture-NEW --components 4 --betas 1e-6,1e-5,1e-4
.venv/bin/python scripts/run_kl_bottleneck.py --out runs/kl-fast-prior-NEW --prior-lr-multiplier 10 --betas 1e-5,1e-4
.venv/bin/python scripts/run_kl_bottleneck.py --out runs/kl-warm-start-NEW --betas 1e-4 --kl-warmup-updates 296
.venv/bin/python scripts/run_kl_bottleneck.py --out runs/kl-late-start-NEW --betas 1e-4 --kl-warmup-updates 651
.venv/bin/python scripts/audit_kl_bottleneck.py
.venv/bin/python scripts/report_kl_bottleneck.py
```

Audit roots are explicit constants in the audit script; change them for new roots. Every scientific run refuses an existing output directory and retains source snapshots, config/specification, replay, checkpoints, metrics, frontier, and animation data. Recorded roots end in 20261110. Portable [audit/results](assets/kl-bottleneck-audit.json) and [summary](assets/kl-bottleneck-summary.json). No background runs remain.
'''
    Path('docs/kl-bottleneck.md').write_text(doc)
    print('\n'.join(table));print('best positive:',best)

if __name__=='__main__':main()
