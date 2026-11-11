"""Report complete matched noisy-frontier runs and render restoration animations."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from map_learning.config import load_config
from map_learning.maze import Maze


def main():
    p=argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True); a=p.parse_args()
    assets=Path('docs/assets'); assets.mkdir(exist_ok=True)
    rows=[]; progress={}; frames={}
    colors={'distance':'#1876bc', 'disagreement':'#df662c'}
    labels={'distance':'Nearest latent distance', 'disagreement':'Ensemble disagreement'}
    for arm in colors:
        for seed in range(3):
            directory=a.root/f'{arm}-seed{seed}'
            r=json.loads((directory/'result.json').read_text()); rows.append(r)
            progress[arm, seed]=[json.loads(s) for s in (directory/'progress.jsonl').read_text().splitlines()]
            frames[arm, seed]=np.load(directory/'frames.npz')
    for seed in range(3):
        pair=[r for r in rows if r['seed']==seed]
        assert len({r['initial_parameters_sha256'] for r in pair})==1
        assert len({(r['interactions'],r['updates'],r['trained_through']) for r in pair})==1
    audit=[]
    for seed in range(3):
        pair=[np.load(a.root/f'{arm}-seed{seed}'/'replay.npz') for arm in colors]
        assert np.array_equal(pair[0]['next_observations'][:,2:], pair[1]['next_observations'][:,2:])
        for arm,data in zip(colors,pair):
            _,counts=np.unique(data['positions'][-5000::5],axis=0,return_counts=True)
            audit.append({'arm':arm,'seed':seed,'last_phase_expanded_positions':len(counts),
                          'top_position_expansions':int(counts.max()),
                          'top5_position_fraction':float(np.sort(counts)[-5:].sum()/counts.sum())})
        for data in pair: data.close()
    (assets/'noisy-frontier-audit.json').write_text(json.dumps({'paired_noise_streams_identical':True,'rows':audit},indent=2)+'\n')
    (assets/'noisy-frontier-results.json').write_text(json.dumps(rows, indent=2)+'\n')
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for arm in colors:
        for seed in range(3):
            r=next(x for x in rows if x['arm']==arm and x['seed']==seed)
            pr=progress[arm,seed]; ph=r['phases']; c=colors[arm]
            kw=dict(color=c,alpha=.65,linestyle=['-','--',':'][seed],label=labels[arm] if seed==0 else None)
            axes[0,0].plot([x['interactions']/1000 for x in pr],[100*x['coverage'] for x in pr],**kw)
            axes[0,1].plot([x['phase'] for x in ph],[100*x['repeat_fraction'] for x in ph],**kw)
            axes[1,0].plot([x['phase'] for x in ph],[x['noise_to_move_ratio'] for x in ph],**kw)
            axes[1,1].plot([x['phase'] for x in ph],[x['seen_normalized_mse'] for x in ph],**kw)
    axes[0,0].set(xlabel='Real action executions (thousands)',ylabel='Reachable physical states observed (%)',ylim=(0,100),title='Coverage, including generated neighbors')
    axes[0,1].set(xlabel='Search phase',ylabel='Repeated physical-state expansions (%)',title='Duplicates are retained; phase-local repeats',ylim=(0,100))
    axes[1,0].set(xlabel='Training phase',ylabel='Noise distance / one-step spatial distance',yscale='log',title='Fresh noise at fixed positions vs movement')
    axes[1,0].axhline(1,color='gray',lw=.7)
    axes[1,1].set(xlabel='Training phase',ylabel='Replay prediction MSE / latent variance',yscale='log',title='Sampled replay fit (ensemble mean)')
    for ax in axes.flat: ax.grid(alpha=.2)
    axes[0,0].legend(); fig.suptitle('Noisy observations: two frontier priorities, three matched seeds')
    fig.savefig(assets/'noisy-frontier-results.png',dpi=160); plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(12,8),constrained_layout=True)
    artists={}
    for row,arm in enumerate(colors):
        for seed in range(3):
            ax=axes[row,seed]; env=Maze(load_config(a.root/f'{arm}-seed{seed}'/'config.json').maze); env.draw(ax)
            dots=ax.scatter([],[],s=5,c=colors[arm],alpha=.6)
            current=ax.scatter([],[],s=35,c='#e73040',edgecolors='white',zorder=6)
            title=ax.set_title(''); artists[arm,seed]=(dots,current,title)
            ax.set_xticks([]); ax.set_yticks([])
    length=len(frames['distance',0]['interactions'])
    indices=np.unique(np.linspace(0,length-1,min(length,100)).astype(int))
    def update(frame):
        for arm in colors:
            for seed in range(3):
                f=frames[arm,seed]; dots,current,title=artists[arm,seed]
                dots.set_offsets(f['points'][f['visited'][frame]])
                current.set_offsets(f['position'][frame:frame+1])
                title.set_text(f'{labels[arm]} · seed {seed}\n{int(f["visited"][frame].sum())}/1784 states · phase {f["phase"][frame]}')
        fig.suptitle(f'{frames["distance",0]["interactions"][frame]:,} real actions · red = restored expansion state\nEach phase restarts at center; blue/orange = cumulative observed states')
    animation=FuncAnimation(fig,update,frames=indices,interval=150)
    animation.save(assets/'noisy-frontier.gif',writer=PillowWriter(fps=7),dpi=95)
    update(length-1); fig.savefig(assets/'noisy-frontier-final.png',dpi=140); plt.close(fig)
    table=[]
    for arm in colors:
        rr=[r for r in rows if r['arm']==arm]
        table.append(f'| {labels[arm]} | '+ ' / '.join(str(r['visited']) for r in rr)+f' | {100*np.mean([r["coverage"] for r in rr]):.1f}% | '+ ' / '.join(f'{100*r["phases"][-1]["repeat_fraction"]:.1f}%' for r in rr)+' |')
    report=f'''# Noisy-observation frontier search — November 10, 2026

Two arms, three paired seeds, trained from scratch. **All six runs completed.**

| Priority | Observed states, seeds 0 / 1 / 2 | Mean coverage | Final-phase repeated expansions, seeds 0 / 1 / 2 |
|---|---|---|---|
{chr(10).join(table)}

![Results](assets/noisy-frontier-results.png)

![Search animation](assets/noisy-frontier.gif)

## Findings

Nearest-latent distance reaches **73.8% mean coverage**, versus **6.5%** for incoming-edge disagreement. The advantage appears in all three seeds; distance coverage still varies substantially (60.6–97.0%).

The encoder learns strong nuisance invariance under the existing joint loss. At random initialization, independent noise at the same position produces latent distances 92–142 times the mean adjacent-position centroid distance. After training, the ratio is 0.13–0.14 for distance search and 0.10–0.15 for disagreement search. No new invariance loss was needed in this coordinate-input experiment. The contrastive pressure did not force the encoder to preserve nuisance information here.

Disagreement's failure is compatible with accurate fitting and noise suppression. In the final phase it expands only **4 / 12 / 5 distinct physical positions** out of 1,000 pops; one seed expands a single position 966 times. Distance expands **1,000 / 367 / 393 distinct positions**. Frozen incoming-edge scores do not decrease when a transition is collected again during a phase, whereas distance scores immediately decrease near each added visited latent. Thus disagreement has no within-phase mechanism to retire familiar high-scoring branches. These measurements support that failure mode, but do not establish that adding a novelty gate or changing training cadence would fix it.

Distance still has repeats in two seeds, and phase restarts discard search memory. This is neither exhaustive graph search nor evidence of perfect state equivalence. Models trained under the two arms receive different spatial data. There is no frozen-random-encoder arm in this two-arm study, so the before/after invariance measurements are stronger evidence for representation change than coverage alone is for its causal benefit.

All six runs total 600,000 real actions, 120,000 expansions, and 24,672 gradient updates. Paired initial parameter hashes, action/update budgets, and every successor-noise draw were checked identical across arms. All 61 repository tests pass, including queue reprioritization, duplicate retention, brute-force distance agreement, and noise/physics isolation.

## Protocol

Same 12×12-room maze, seed 7, step .25, 1,784 reachable physical positions. Observations are normalized (x,y) plus eight independent standard-normal values, newly sampled on each successor observation. Clean coordinates remain observable features: this tests nuisance rejection, not learning geometry from pixels. Physical transitions are deterministic.

Each run has 20 phases × 1,000 expansions × five executed actions = 100,000 real action executions and 20,000 expansions. Start from a random encoder/dynamics ensemble with no warmup. Every phase freezes the networks, resets to the center and clears the frontier and phase-local visited latents. All raw transitions remain in replay across phases. After each phase, train **one complete accumulated-replay epoch**, batch 256, shared shuffled batches across three heads. This is 4,112 gradient updates per run (including the final training block, which does not affect final coverage).

The encoder and each dynamics head use width 256, three SiLU hidden layers, latent dimension eight, residual mean dynamics. Joint loss = mean per-head next-latent MSE + 0.01 × mean max(0, 1 − squared latent pair distance). Gradients pass through both encoded transition endpoints and both repulsion endpoints. Encoder LR 9e-5, dynamics LR 3e-4. No reconstruction, codebook, compression, or new loss.

Expand every popped state with all five actions, including stay. Simulator snapshots contain only physical position; the previously sampled observation is cached with each frontier entry and used as the source observation when expanded. Restoring does not rewind the observation-noise RNG. Each action generates fresh noise. No physical-state, observation-byte, or latent duplicate removal; no radius cutoff. Physical positions are passed only to restoration and evaluation. Each replay transition receives a separate episode ID because consecutive records are branching edges, not a trajectory.

**Distance:** actual successor latent's Euclidean distance to the nearest expanded latent in this phase. Eagerly decrease all active keys after adding a visited latent. A vector-backed maximum-priority queue implements the exact rule. Newly generated states are in the frontier, not the visited set, until expanded.

**Disagreement:** variance across the three predicted means for the incoming parent observation/action, averaged across latent dimensions. Keys stay fixed within each frozen phase. Raw variance and normalized variance give identical ordering within a phase. This is disagreement, not observed prediction error, and does not promise calibrated epistemic uncertainty. Exact ties use an isolated seeded RNG.

Every phase ends with 4,001 unexpanded frontier entries, which are discarded on restart. Models/optimizer, replay, training and noise RNG streams persist. Within each seed, both arms start with bit-identical weights and have equal interactions, noise-draw counts, and gradient-update counts. Their collected datasets diverge.

## Diagnostics and interpretation

Coverage includes all physically observed successors, even those never popped. Repeat fraction counts repeated physical positions among expansions within a phase; position labels do not influence the queue. It does not measure repeats against lifetime experience. An animation jumps between restored states; it is not an executable navigation path.

Noise probes use 64 fixed evaluation positions with 16 independent noise samples each, separate RNGs, and no training use. Noise/movement ratio compares mean latent distance between independent noisy observations of the same position against mean neighboring-position centroid distance. Neighbor centroids use common nuisance draws, avoiding a finite-sample noise floor in the spatial denominator. Whole-maze probe locations can be unvisited; these are diagnostics, not a clean generalization benchmark. Noise variance fraction is within-position variance divided by within-plus-between-position variance. Representation diagnostics are measured after each phase’s training, while that phase’s coverage was collected before training. Replay MSE is a 2,048-transition sample normalized by latent variance, not discrete-state accuracy.

No claim of statistical significance: three seeds and one maze. The comparison evaluates incoming-edge disagreement specifically, not every uncertainty-based frontier strategy. Nearest-distance search benefits from receiving clean coordinates as part of its input. Continuous position, image observations, stochastic physical transitions, and navigation without restoration remain untested here.

## Reproduction

```sh
.venv/bin/python -m pytest -q
.venv/bin/python scripts/run_noisy_frontier.py --out runs/noisy-frontier-NEW --jobs 3
.venv/bin/python scripts/report_noisy_frontier.py --root runs/noisy-frontier-NEW
```

Existing run directories are never overwritten. Full replay, initial/intermediate/final checkpoints, per-phase diagnostics, source snapshots, and animation data are in `{a.root}`. Portable results: [JSON](assets/noisy-frontier-results.json). Pilot smoke run is separate and excluded from results.
'''
    Path('docs/noisy-frontier.md').write_text(report)
    print('\n'.join(table))

if __name__=='__main__': main()
