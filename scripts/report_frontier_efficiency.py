"""Combine audits and controlled interventions; produce report, plots, animation."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,PillowWriter
from map_learning.maze import Maze
from map_learning.config import load_config

ROOT=Path('runs/persistent-frontier-20261110')
OPT=Path('runs/frontier-optimization-20261110')
FIT=Path('runs/frontier-fitting-branches-20261110')
ZERO=Path('runs/frontier-zero-noise-oracle-20261110')


def main():
    assets=Path('docs/assets');conditions=['baseline','early','distributed']
    labels={'baseline':'Original frequent training','early':'+2,000 updates at 5k actions','distributed':'+100 updates every 5k actions'}
    colors={'baseline':'#286bad','early':'#28955f','distributed':'#d1782d'}
    dirs={(c,s):(ROOT/f'interleaved100-seed{s}' if c=='baseline' else OPT/f'{c}-seed{s}') for c in conditions for s in range(3)}
    results={(c,s):json.loads((d/'result.json').read_text()) for (c,s),d in dirs.items()}
    progress={(c,s):[json.loads(x) for x in (d/'progress.jsonl').read_text().splitlines()] for (c,s),d in dirs.items()}
    audit=json.loads(Path('runs/frontier-efficiency-audit-20261110/results.json').read_text())
    fit=[json.loads(p.read_text()) for p in sorted(FIT.glob('*/result.json'))]
    zero=[json.loads(p.read_text()) for p in sorted(ZERO.glob('*/result.json'))]
    # Paired initialization, first 5k experience, noise streams, and total budgets.
    for seed in range(3):
        assert len({results[c,seed]['initial_parameters_sha256'] for c in conditions})==1
        with np.load(dirs['baseline',seed]/'replay.npz') as base:
            for c in ['early','distributed']:
                r=results[c,seed];assert r['updates']==6112 and r['samples_per_head']==1562000
                with np.load(dirs[c,seed]/'replay.npz') as data:
                    for key in ['observations','actions','next_observations','positions','next_positions']:
                        np.testing.assert_array_equal(base[key][:5000],data[key][:5000])
                    np.testing.assert_array_equal(base['next_observations'][:,2:],data['next_observations'][:,2:])
    for seed,block in [(0,10),(1,10),(2,10),(2,20)]:
        with np.load(FIT/f'seed{seed}-block{block}-extra0'/'branch.npz') as x,np.load(FIT/f'seed{seed}-block{block}-extra2000'/'branch.npz') as y,np.load(ZERO/f'seed{seed}-block{block}-extra0'/'branch.npz') as z:
            np.testing.assert_array_equal(x['next_observations'][:,2:],y['next_observations'][:,2:])
            np.testing.assert_array_equal(x['next_observations'][:,2:],z['next_observations'][:,2:])
    # Separate early representation acquisition from later inefficiency using a
    # descriptive threshold on isolated probes, never a controller decision.
    decomposition=[]
    for seed in range(3):
        r=results['baseline',seed]
        threshold=next(x['block'] for x in r['blocks'] if x['noise_to_move_ratio']<1)
        diag=next(x for x in audit if x['arm']=='interleaved100' and x['seed']==seed)
        end=diag['expansions_until_completion_or_budget']
        with np.load(dirs['baseline',seed]/'replay.npz') as data:
            seen=set();early=late=0
            for i,pos in enumerate(data['positions'][:end*5:5]):
                if tuple(pos) in seen:
                    if i<threshold*1000:early+=1
                    else:late+=1
                seen.add(tuple(pos))
        decomposition.append({'seed':seed,'first_probe_noise_below_movement_actions':threshold*5000,
            'early_repeat_actions':early*5,'later_repeat_actions':late*5,'repeat_fraction_to_end':diag['repeat_fraction_until_completion_or_budget'],
            'endpoint_actions':r['first_full'] or 100000})
    portable={'online':[{'condition':c,**results[c,s]} for c in conditions for s in range(3)],
        'fixed_archive_branches':fit,'privileged_zero_noise_branches':zero,'archive_audits':audit,'decomposition':decomposition,
        'checks':{'initial_weights_match':True,'first5000_experience_identical':True,'paired_noise_streams_match':True,'extra_budgets_match':True}}
    (assets/'frontier-efficiency-results.json').write_text(json.dumps(portable,indent=2)+'\n')
    fig,axes=plt.subplots(1,3,figsize=(14,4.3),constrained_layout=True)
    for seed,ax in enumerate(axes):
        for c in conditions:
            p=progress[c,seed];ax.plot([x['interactions']/1000 for x in p],[100*x['coverage'] for x in p],color=colors[c],label=labels[c])
        ax.set(title=f'Seed {seed}',xlabel='Real actions (thousands)',ylabel='Physical coverage (%)',ylim=(0,101));ax.grid(alpha=.2)
    axes[0].legend(fontsize=8);fig.suptitle('Extra optimization timing: same +2,000-update budget, mixed outcomes')
    fig.savefig(assets/'frontier-efficiency-online.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4.3),constrained_layout=True)
    x=np.arange(3);w=.25
    for offset,(updates,label,color) in enumerate([(0,'Frozen original','#286bad'),(2000,'Fit existing replay, then freeze','#28955f')]):
        rr=[next(r for r in fit if r['seed']==s and r['prefix']==50000 and r['extra_updates']==updates) for s in range(3)]
        axes[0].bar(x+(offset-1)*w,[r['new_states'] for r in rr],width=w,color=color,label=label)
        axes[1].bar(x+(offset-.5)*.35,[100*r['starting_priority']['top100_repeat_fraction'] for r in rr],width=.35,color=color,label=label)
    rr=[next(r for r in zero if r['seed']==s and r['prefix']==50000) for s in range(3)]
    axes[0].bar(x+w,[r['new_states'] for r in rr],width=w,color='#888888',label='Known nuisance removed (oracle)')
    axes[0].set(title='Same 50k archive: next 10k actions',ylabel='New physical states',xticks=x,xticklabels=['Seed 0','Seed 1','Seed 2']);axes[0].legend(fontsize=7)
    axes[1].set(title='Initial queue priorities after fitting',ylabel='Top 100 entries that repeat an expanded state (%)',xticks=x,xticklabels=['Seed 0','Seed 1','Seed 2'],ylim=(0,101))
    axes[2].bar(x,[r['early_repeat_actions']/1000 for r in decomposition],label='Before noise probe ratio < 1',color='#999999')
    axes[2].bar(x,[r['later_repeat_actions']/1000 for r in decomposition],bottom=[r['early_repeat_actions']/1000 for r in decomposition],label='After ratio < 1',color='#ca7b4b')
    axes[2].set(title='Repeat actions before completion/budget',ylabel='Actions with no new physical coverage (thousands)',xticks=x,xticklabels=['Seed 0','Seed 1','Seed 2']);axes[2].legend(fontsize=7)
    for ax in axes: ax.title.set_fontsize(11)
    fig.savefig(assets/'frontier-efficiency-diagnostics.png',dpi=160);plt.close(fig)
    # All three online conditions, same action clock. Baseline reused, not rerun.
    frames={(c,s):np.load(d/'frames.npz') for (c,s),d in dirs.items()}
    fig,axes=plt.subplots(3,3,figsize=(12,12),constrained_layout=True);artists={}
    for row,c in enumerate(conditions):
        for seed in range(3):
            ax=axes[row,seed];Maze(load_config(dirs[c,seed]/'config.json').maze).draw(ax)
            dots=ax.scatter([],[],s=4,c=colors[c],alpha=.7);current=ax.scatter([],[],s=25,c='red',zorder=6)
            artists[c,seed]=(dots,current,ax.set_title(''));ax.set_xticks([]);ax.set_yticks([])
    count=len(frames['baseline',0]['interactions'])
    def update(i):
        for c in conditions:
            for s in range(3):
                f=frames[c,s];dots,current,title=artists[c,s]
                dots.set_offsets(f['points'][f['visited'][i]]);current.set_offsets(f['position'][i:i+1])
                title.set_text(f'{labels[c]} · seed {s}\n{f["visited"][i].sum()}/1784 states');title.set_fontsize(10)
        fig.suptitle(f'{frames["baseline",0]["interactions"][i]:,} real actions · persistent noisy frontier\nRed marks restored expansion state; jumps are simulator restores')
    ani=FuncAnimation(fig,update,frames=np.unique(np.linspace(0,count-1,100).astype(int)),interval=150)
    ani.save(assets/'frontier-efficiency.gif',writer=PillowWriter(fps=7),dpi=85)
    update(count-1);fig.savefig(assets/'frontier-efficiency-final.png',dpi=130);plt.close(fig)
    def completion(r):return f'{r["first_full"]:,}' if r['first_full'] else '>100,000 (not reached)'
    table=[]
    for c in conditions:
        table.append('| '+labels[c]+' | '+' / '.join(completion(results[c,s]) for s in range(3))+' | '+' / '.join(str(results[c,s]['visited']) for s in range(3))+' |')
    branch_table=[]
    for seed in range(3):
        rr=[next(r for r in fit if r['seed']==seed and r['prefix']==50000 and r['extra_updates']==u) for u in [0,2000]]
        z=next(r for r in zero if r['seed']==seed and r['prefix']==50000)
        branch_table.append(f'| {seed} | {rr[0]["initial_visited"]} | {rr[0]["new_states"]} | {rr[1]["new_states"]} | {z["new_states"]} |')
    warm='; '.join(f'seed {r["seed"]}: first probe ratio <1 at {r["first_probe_noise_below_movement_actions"]:,} actions; {r["early_repeat_actions"]:,} repeat actions before, {r["later_repeat_actions"]:,} after' for r in decomposition)
    report=f'''# Where noisy-frontier exploration spends its actions

Completed November 10, 2026. Saved-run audits plus 18 new diagnostic/online runs; 720,000 additional real actions and 44,672 new gradient updates. Existing runs preserved. No loss changes, merging thresholds, or disagreement policy.

## Findings

**Repeated physical expansions dominate even before completion.** In the original frequent-training seeds, 87.1%, 88.6%, and 91.7% of expansions before full coverage or the 100k limit revisit previously expanded positions. Because every expansion executes all five deterministic actions, those repeats cannot expose new physical states. They can still provide useful noisy observations and gradient-training data: “no new coverage” does not mean no learning value. Accounting excludes gratuitous actions after full coverage, and uses the whole expansion containing first completion.

Representation acquisition explains only part of this. A descriptive, isolated whole-maze probe first reports same-position noise distance smaller than neighboring-position distance around the checkpoints below. Much of the repeated interaction happens **after** that crossing. This is not a control threshold or a causal definition of sufficient training; diagnostics are sampled every 5k actions, and represent post-training weights.

{warm}.

**Seed 2 was not trapped behind an inaccessible frontier.** At 100k actions, 43 states were unobserved, but 34 queued physical positions could immediately discover a new neighbor. The best such entry ranked 7th. Nevertheless, 88 of the top 100 queue entries repeated an expanded position. This is a snapshot ranking, not a promise that the seventh pop must be productive: every pop updates distances and generates more entries.

The same final archive and frozen model finish in **3,846 additional actions** under a fresh paired noise stream. Fitting its existing replay for 2,000 updates first reduces that to **489 actions**. No newly collected data is used for training in either branch. Removing known nuisance dimensions only for encoding finishes in 584 actions; that is a privileged diagnostic, not an admissible learned solution.

**Small average noise sensitivity is insufficient for a maximum-priority queue.** The queue selects extreme scores from tens of thousands of noisy entries. Residual within-state variation can outrank physically different candidates even when the average noise/movement ratio is about 0.1. At 50k, 96%, 92%, and 96% of the top 100 entries repeat previously expanded positions. After fixed-data fitting, these fractions become 0%, 2%, and 93%. Global probe ratios and average prediction MSE do not reliably reflect that improvement: seed 0's normalized replay MSE slightly increases while its queue ranking dramatically improves. Local ranking matters more directly than aggregate fitting metrics.

## Controlled fixed-archive fitting

Take the original frequent-training checkpoints at 50k actions. Reconstruct their exact active frontier and visited archive from raw observation entry identity; validate reconstruction against all final saved archives. Compare unchanged weights versus 2,000 replay updates using **only that existing dataset**, then freeze both while expanding for 10k more actions. The model, queue archive, and successor-noise stream are paired. No encoder is supplied with future data.

| Seed | Initially observed | New states: frozen original | New states: extra fitting | New states: known noise removed |
|---|---:|---:|---:|---:|
{chr(10).join(branch_table)}

Extra fitting helps all three seeds in this matched intervention, but seed 2 still does not finish. Seeds 0/1 finish in 2,958/2,987 additional actions after fitting; neither unchanged model finishes within 10k. Frozen-original branches intentionally stop online learning, so they are not direct continuations of the baseline online algorithm. Additional fitting can change all aspects of geometry; these experiments do not attribute improvement exclusively to noise suppression.

The zero-noise control sets the eight **known nuisance input coordinates** to zero for ranking, leaving observed transitions and simulator physics untouched and models frozen. It supplies exact observation invariance, so repeated physical states encode identically. Its strong results diagnose remaining sensitivity but do not demonstrate that the agent learned equivalent invariance. All three 50k archives finish with this oracle intervention in 3,016 / 2,917 / 5,772 extra actions.

![Fixed-archive and repeat diagnostics](assets/frontier-efficiency-diagnostics.png)

## Online optimization timing

Two adaptive follow-ups were chosen after the repeat/queue audit. Both retain persistent search, frequent interleaved epochs, and unchanged losses. **Early:** add 2,000 replay updates immediately after 5k actions. **Distributed:** add 100 updates after each 5k actions, totaling the same 2,000. All six new runs use three paired seeds, 100k actions, 6,112 updates, and 1,562,000 transition presentations per head. Base epochs still train every accumulated transition exactly once per block; extra updates sample replay with replacement. Shared batches across heads. Every new run's initial weights, first 5k experience, and complete successor-noise stream were checked against the original baseline. Extra optimization does not consume environment observations.

| Schedule | Actions to full coverage, seeds 0 / 1 / 2 | Final observed states, seeds 0 / 1 / 2 |
|---|---|---|
{chr(10).join(table)}

**Early extra training achieves full coverage in all three tested seeds**, but is slightly slower than baseline in seeds 0/1. Its benefit is concentrated in seed 2. **Distributed extra training is highly variable:** seed 0 finishes at 25,438 actions, seed 1 reaches only 1,521 states at the limit, seed 2 finishes at 95,213. More optimization changes representation and therefore subsequent collected data; it is not monotonically helpful. Do not select the 25k seed as evidence of a robust schedule, or average censored completion times as if they were observed. Three seeds and one layout do not establish broad reliability.

![Online comparisons](assets/frontier-efficiency-online.png)

![Exploration animation](assets/frontier-efficiency.gif)

## Implications and remaining questions

The existing joint loss is capable of supporting complete noisy-maze exploration in all three early-extra seeds, without state merging. But the original “solved” claim omitted sample cost and seed variation. The action budget remains far above an exact-state graph explorer's upper bound of 1,784 × 5 = 8,920 actions to expand every state once. That bound supplies identity and restoration; it is a reference, not a fair learned-agent baseline or a lower bound on any exploration task.

Both inadequate optimization on a given archive and residual noise-sensitive priority ordering are real contributors. The fixed-archive intervention establishes that better search can come from more training without new data. The zero-noise control establishes that perfect nuisance invariance removes duplicate novelty in this discrete setting. Neither tells us a universally good training stopping rule. Aggregate MSE and average noise ratio alone look insufficient for that rule. A next training/prioritization change should be evaluated on queue-relevant local/tail behavior as well as coverage, with multiple seeds; no new loss is implemented here.

## Validation, artifacts, reproduction

The six original persistent runs were audited at 5k/25k/50k/100k checkpoints. Repeat discovery was checked directly: no repeated physical expansion produced a new position. Exact frontier reconstruction matches saved final observations and archive ordering. Training-prefix and successor-noise controls passed for every new online run and all paired fixed-archive branches. Reports preserve unsuccessful runs as well as successes. All 64 repository tests pass.

```sh
.venv/bin/python scripts/audit_frontier_efficiency.py --out runs/frontier-efficiency-audit-NEW
.venv/bin/python scripts/run_frontier_optimization.py --out runs/frontier-optimization-NEW
.venv/bin/python scripts/branch_frontier_fitting.py --out runs/frontier-fitting-branches-NEW
.venv/bin/python scripts/branch_frontier_fitting.py --zero-only --out runs/frontier-zero-noise-oracle-NEW
.venv/bin/python scripts/report_frontier_efficiency.py
```

Reporting currently names the recorded 20261110 run roots at the top of the script; change those constants when reporting new roots. Launchers refuse existing output directories. Raw data: `runs/frontier-efficiency-audit-20261110`, `runs/frontier-optimization-20261110`, `runs/frontier-fitting-branches-20261110`, `runs/frontier-zero-noise-oracle-20261110`. Branch scripts preserve actual noisy replay even for privileged encoding. Portable combined [results](assets/frontier-efficiency-results.json); code/source snapshots and checkpoints retained. No background work remains.
'''
    Path('docs/frontier-efficiency.md').write_text(report)
    print('\n'.join(table));print(warm)

if __name__=='__main__':main()
