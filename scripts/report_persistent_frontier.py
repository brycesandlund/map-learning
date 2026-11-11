"""Validate and visualize the persistent-frontier cadence study."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import scipy
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from map_learning.training import Trainer, seed_all
from map_learning.config import load_config
from map_learning.maze import Maze


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    seed_all(0,1)
    assets=Path('docs/assets');assets.mkdir(exist_ok=True)
    colors={'epoch1000':'#2374b5','interleaved100':'#dd7627'}
    labels={'epoch1000':'Train every 1,000 expansions','interleaved100':'Train every 100 expansions'}
    rows=[];progress={};frames={};audit=[]
    for arm in colors:
        for seed in range(3):
            directory=a.root/f'{arm}-seed{seed}'
            r=json.loads((directory/'result.json').read_text());rows.append(r)
            progress[arm,seed]=[json.loads(x) for x in (directory/'progress.jsonl').read_text().splitlines()]
            frames[arm,seed]=np.load(directory/'frames.npz')
            t=Trainer.load(directory/'checkpoint.pt')
            with np.load(directory/'frontier.npz') as f:
                assert len(f['observations'])==80001 and len(f['visited_observations'])==20000
                with torch.no_grad():
                    encode=lambda x:np.concatenate([t.model.encode(t.tensor(x[i:i+1024])).numpy() for i in range(0,len(x),1024)])
                    z=encode(f['observations']);v=encode(f['visited_observations'])
                np.testing.assert_array_equal(z,f['latents'])
                exact=cKDTree(v).query(z,eps=0,workers=1)[0]
                np.testing.assert_allclose(exact,f['priorities'],atol=1e-12,rtol=1e-12)
                ids=np.random.default_rng(seed).choice(len(z),32,replace=False)
                brute=np.linalg.norm(z[ids,None].astype(float)-v[None].astype(float),axis=-1).min(1)
                np.testing.assert_allclose(exact[ids],brute,rtol=1e-12,atol=1e-12)
                np.testing.assert_array_equal(f['observations'][:,:2],(2*f['positions']/12-1).astype(np.float32))
            audit.append({'arm':arm,'seed':seed,'frontier_entries':r['frontier_entries'],'visited_entries':r['visited_entries'],
                'fresh_encodings_match':True,'exact_priorities_match':True,'brute_force_queries_checked':32})
    for seed in range(3):
        pair=[r for r in rows if r['seed']==seed]
        assert len({r['initial_parameters_sha256'] for r in pair})==1
        assert len({(r['interactions'],r['updates'],r['samples_per_head'],r['trained_through']) for r in pair})==1
        for i in range(20):
            assert pair[0]['blocks'][i]['updates']==pair[1]['blocks'][i]['updates']
            assert pair[0]['blocks'][i]['samples_per_head']==pair[1]['blocks'][i]['samples_per_head']
        with np.load(a.root/f'epoch1000-seed{seed}'/'replay.npz') as x,np.load(a.root/f'interleaved100-seed{seed}'/'replay.npz') as y:
            np.testing.assert_array_equal(x['next_observations'][:,2:],y['next_observations'][:,2:])
    (assets/'persistent-frontier-results.json').write_text(json.dumps(rows,indent=2)+'\n')
    (assets/'persistent-frontier-audit.json').write_text(json.dumps({'scipy':scipy.__version__,'paired_noise_and_budgets_match':True,'rows':audit},indent=2)+'\n')
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for arm in colors:
        for seed in range(3):
            r=next(x for x in rows if x['arm']==arm and x['seed']==seed);pr=progress[arm,seed];bl=r['blocks']
            kw=dict(color=colors[arm],alpha=.75,linestyle=['-','--',':'][seed],label=labels[arm] if seed==0 else None)
            axes[0,0].plot([x['interactions']/1000 for x in pr],[100*x['coverage'] for x in pr],**kw)
            axes[0,1].plot([x['block'] for x in bl],[100*x['block_repeat_fraction'] for x in bl],**kw)
            axes[1,0].plot([x['block'] for x in bl],[x['noise_to_move_ratio'] for x in bl],**kw)
            axes[1,1].plot([x['block'] for x in bl],[x['refresh_seconds'] for x in bl],**kw)
    old=json.loads(Path('docs/assets/noisy-frontier-results.json').read_text())
    for i,r in enumerate(x for x in old if x['arm']=='distance'):
        axes[0,0].plot([x['interactions']/1000 for x in r['phases']],[100*x['coverage'] for x in r['phases']],color='gray',alpha=.4,lw=1,label='Earlier reset-search runs' if i==0 else None)
    axes[0,0].set(xlabel='Real action executions (thousands)',ylabel='Physical states observed (%)',ylim=(0,101),title='Persistent search, three paired seeds')
    axes[0,0].legend(fontsize=8)
    axes[0,1].set(xlabel='1,000-expansion block',ylabel='Repeated physical expansions within block (%)',ylim=(0,101),title='Within-block repeat rate')
    axes[1,0].set(xlabel='1,000-expansion block',ylabel='Noise distance / spatial movement distance',yscale='log',title='After-training representation diagnostics')
    axes[1,0].axhline(1,color='gray',lw=.7)
    axes[1,1].set(xlabel='1,000-expansion block',ylabel='Cumulative seconds',title='Re-encoding and exact priority refresh cost')
    for ax in axes.flat:ax.grid(alpha=.2)
    fig.savefig(assets/'persistent-frontier-results.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(12,8),constrained_layout=True);artists={}
    for row,arm in enumerate(colors):
        for seed in range(3):
            ax=axes[row,seed];Maze(load_config(a.root/f'{arm}-seed{seed}'/'config.json').maze).draw(ax)
            dots=ax.scatter([],[],s=5,c=colors[arm],alpha=.65);current=ax.scatter([],[],s=35,c='#e73040',edgecolors='white',zorder=6)
            title=ax.set_title('');artists[arm,seed]=(dots,current,title);ax.set_xticks([]);ax.set_yticks([])
    count=len(frames['epoch1000',0]['interactions']);indices=np.unique(np.linspace(0,count-1,100).astype(int))
    def update(frame):
        for arm in colors:
            for seed in range(3):
                f=frames[arm,seed];dots,current,title=artists[arm,seed]
                dots.set_offsets(f['points'][f['visited'][frame]]);current.set_offsets(f['position'][frame:frame+1])
                title.set_text(f'{labels[arm]} · seed {seed}\n{int(f["visited"][frame].sum())}/1784 observed states')
        fig.suptitle(f'{frames["epoch1000",0]["interactions"][frame]:,} real actions · persistent frontier and visited archive\nRed = simulator-restored expansion state; no search restarts')
    ani=FuncAnimation(fig,update,frames=indices,interval=150)
    ani.save(assets/'persistent-frontier.gif',writer=PillowWriter(fps=7),dpi=95)
    update(count-1);fig.savefig(assets/'persistent-frontier-final.png',dpi=140);plt.close(fig)
    table=[]
    for arm in colors:
        rr=[r for r in rows if r['arm']==arm]
        full=' / '.join(f'{r["first_full"]:,}' if r['first_full'] is not None else 'not reached' for r in rr)
        table.append(f'| {labels[arm]} | '+ ' / '.join(str(r['visited']) for r in rr)+f' | {100*np.mean([r["coverage"] for r in rr]):.1f}% | {full} |')
    timing=[]
    for arm in colors:
        rr=[r for r in rows if r['arm']==arm]
        timing.append(f'| {labels[arm]} | '+ ' | '.join(f'{np.mean([r[k] for r in rr]):.1f}' for k in ['search_seconds','training_seconds','refresh_seconds','seconds'])+' |')
    report=f'''# Persistent noisy-maze frontier — November 10, 2026

Six completed runs: two distance-priority arms, three paired seeds. Same noisy discrete maze and losses as [the previous study](noisy-frontier.md). No ensemble-disagreement search runs.

| Cadence | Observed states, seeds 0 / 1 / 2 (of 1,784) | Mean coverage | Actions to first 100% coverage, seeds 0 / 1 / 2 |
|---|---|---|---|
{chr(10).join(table)}

![Coverage and diagnostics](assets/persistent-frontier-results.png)

![Exploration animation](assets/persistent-frontier.gif)

## Findings

**Keeping search state improves final coverage across all three seeds.** At the previous cadence, persistent search reaches 97.1% mean coverage (1,782 / 1,784 / 1,632 states), versus 73.8% (1,730 / 1,081 / 1,139) in the earlier restart-after-training runs. Persistence does not uniformly improve the entire learning curve: seed 0 discovers substantial territory later than its restarted counterpart, then overtakes it by the final budget.

**Frequent interleaving improves the matched persistent comparison further:** 99.2% mean coverage (1,784 / 1,784 / 1,741). Two seeds finish the maze at 68,628 and 77,266 actions; the third is 43 states short at 100,000 actions. The experiment does not establish complete exploration in every seed.

| Seed | Actions to 90%, epoch every 1,000 expansions | Actions to 90%, interleaved every 100 |
|---|---:|---:|
| 0 | 88,868 | 61,778 |
| 1 | 76,549 | 64,002 |
| 2 | 96,814 | 94,263 |

Frequent training reaches 90% coverage earlier in every seed: mean 73,348 actions versus 87,410, a 16.1% reduction. This is evidence from three seeds on one maze, not a statistical significance claim. It does not dominate at every earlier milestone (seed 1 reaches 50% later). Both conditions receive the same transition presentations and gradient steps; minibatch partitioning differs as described below.

No loss changes were required. Exact physical-state identity and raw coordinates are not used for duplicate removal or ranking. However, clean coordinates remain observation features, and simulator restoration still supplies navigation. This is a stronger result for noisy-coordinate exploration, not yet a continuous-position or no-restoration result.

## Protocol

Keep the entire frontier and visited observation archive from initialization through all 20,000 expansions (100,000 executed actions). Five real actions per expansion, including stay. No state/observation deduplication, approximate merging, or distance threshold. Exact simulator restoration; observations cached with each frontier entry, new independent standard-normal noise for each successor. Observations still include clean normalized coordinates plus eight nuisance dimensions. All raw replay retained. Models initialized from scratch, unchanged width-256/three-layer encoder and three dynamics heads, latent eight. Joint mean-MSE + 0.01 hinge repulsion, same optimizer and learning rates as before.

After training, re-encode **every visited observation and active frontier observation**, then rebuild all nearest-visited distance keys. An exact SciPy cKDTree index (eps=0, one worker) accelerates this calculation without merging observations. Newly visited points immediately decrease active priorities. Newly generated observations enter the frontier; only expanded observations enter the visited archive. Nothing resets when training occurs.

**Every 1,000 expansions:** one shuffled full epoch over all accumulated replay, batch 256 with a final partial batch. This retains the previous training cadence and changes search persistence.

**Every 100 expansions:** distribute that same full epoch across ten intervals. At each 1,000-expansion block start, shuffle the old replay and partition it into ten disjoint slices. Every interval trains on one old slice plus all 500 new transitions, shuffled together. Allocate exactly ceil(5,000 × block_number / 256) gradient steps across the ten intervals. Both arms therefore train on every accumulated transition **exactly once per block per ensemble head**, and take the same gradient-step count at each block boundary. All new transitions are trained by the next training event. Shared batches across ensemble members.

The frequent arm uses variable minibatch sizes (250–300) to match both exposure and update counts; the epoch arm uses 256 with partial tails. Temporal ordering and batch sizes consequently differ, so this is a matched-budget cadence comparison, not a claim that update timing is the only numerical difference. Repulsion sample count per update remains unchanged. Each run ends with **4,112 updates and 1,050,000 transition presentations per head**. Final training/refresh is included in cost/diagnostics but cannot improve already-collected coverage.

Both arms retain 20,000 visited observation entries and 80,001 frontier entries at the end. Physical labels are evaluation/restoration-only. Coverage includes generated successors; first-full milestones are measured at individual action granularity. Search continues for the full budget after complete coverage, if reached. The animation shows restoration jumps, not a physically executed trajectory.

Initial parameter hashes, successor-noise streams, sample counts, and gradient counts are paired and verified. Unlike the earlier restarted search, there is only one initial environment observation; no extra reset-noise draws between blocks. Earlier reset-search runs are shown as contextual references, not rerun as a third arm. Within the new comparison, each pair receives identical nuisance draws in execution order.

## Computational cost

Mean measured wall-clock seconds per run (three concurrent workers; hardware/runtime dependent):

| Cadence | Expansion/search | Training | Latent/priority refresh | Total including diagnostics/I/O |
|---|---:|---:|---:|---:|
{chr(10).join(timing)}

Exact nearest-neighbor refresh is an acceleration only. Audit re-encodes all saved final frontier entries and visited observations and verifies every priority, plus 32 independent brute-force nearest-distance checks per run. Tests cover archive persistence under changing geometry, incremental priority changes after refresh, and exact causal epoch coverage and gradient budgets across all 20 blocks.

## Scope and reproduction

Three seeds, one maze, coordinate observations, simulator restoration. No change to losses, continuous position, or physical stochasticity. Diagnostic noise/movement ratio uses the same fixed independent probe as before; neighboring-position centroids share nuisance draws. Post-training diagnostics do not describe the weights that collected the preceding block. Full replay and checkpoint/source artifacts remain in `{a.root}`.

```sh
.venv/bin/python -m pytest -q
.venv/bin/python scripts/run_persistent_frontier.py --out runs/persistent-frontier-NEW --jobs 3
.venv/bin/python scripts/report_persistent_frontier.py --root runs/persistent-frontier-NEW
```

SciPy is now an explicit dependency (installed version {scipy.__version__}). Launchers refuse existing output directories. A tiny smoke trial with an insufficient update budget failed before scientific runs; input validation was tightened and a separate valid smoke trial passed. Neither smoke trial contributes to these results.

Portable [results](assets/persistent-frontier-results.json) and [audit](assets/persistent-frontier-audit.json). All 64 tests pass.
'''
    Path('docs/persistent-frontier.md').write_text(report)
    print('\n'.join(table));print('\n'.join(timing))

if __name__=='__main__':main()
