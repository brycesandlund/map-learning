"""Validate matched initial data and report completed online search branches."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
 root=Path('runs/search-budget-20261110');rows=[];curves={};labels=['24','96','384','random']
 for label in labels:
  for seed in range(3):
   folder=root/f'{label}-seed{seed}';r=json.loads((folder/'result.json').read_text());assert r['additional_steps']==1200
   source=Path(r['source']);old=dict(np.load(source/'replay.npz'));new=dict(np.load(folder/'replay.npz'))
   for key in ['observations','next_observations','actions','episodes','positions','next_positions']:np.testing.assert_array_equal(new[key][:30000],old[key])
   assert len(new['actions'])==31200 and r['trained_through']==31200
   before=json.loads((source/'config.json').read_text());after=json.loads((folder/'config.json').read_text())
   before['collect_steps']=31200;before['reward_snapshot_every']=100;before['planner']['simulations']=24 if label=='random' else int(label)
   if label=='random':before['planner']['behavior']='random'
   assert before==after
   if label!='24':
    baseline=dict(np.load(root/f'24-seed{seed}'/'reward_snapshots.npz'))
    snapshots=dict(np.load(folder/'reward_snapshots.npz'))
    np.testing.assert_array_equal(snapshots['reward'][0],baseline['reward'][0])
    np.testing.assert_array_equal(snapshots['position'][0],baseline['position'][0])
   x=new['positions'][30000:];y=new['next_positions'][30000:];moving=np.any(x!=y,axis=1)
   recent=np.array([np.any(np.all(y[i]==x[max(0,i-15):i+1],axis=1)) for i in range(len(y))])
   r['moving_recent16_revisit_fraction']=float(recent[moving].mean()) if moving.any() else None
   r['distinct_branch_states']=len({tuple(p) for p in np.concatenate([x,y])})
   audit=json.loads((folder/'fitting-audit.json').read_text())
   r['seen_edge_accuracy']=1-audit['edges']['groups']['tried']['physical_mistake_rate_non_novel']
   r['novel_successor_disagreement_auc']=audit['edges']['groups']['visited_source']['new_successor_auc']['disagreement']
   r['label']=label;rows.append(r)
   curves[label,seed]=[json.loads(line) for line in (folder/'exploration.jsonl').read_text().splitlines()]
 Path('docs/assets/search-budget-results.json').write_text(json.dumps(rows,indent=2)+'\n')
 means={label:{key:float(np.mean([r[key] for r in rows if r['label']==label])) for key in ['new_states','mean_cumulative_new_states','stationary_fraction','moving_recent16_revisit_fraction','search_seconds','training_seconds','seen_edge_accuracy','novel_successor_disagreement_auc']} for label in labels}
 fig,axes=plt.subplots(1,3,figsize=(15,4.8));colors=['#4477aa','#ee7733','#228833','#aa3377']
 for label,color in zip(labels,colors):
  a=np.array([[r['new_states'] for r in curves[label,seed]] for seed in range(3)]);steps=[r['step'] for r in curves[label,0]]
  axes[0].plot(steps,a.mean(0),label=label,color=color);axes[0].fill_between(steps,a.min(0),a.max(0),alpha=.1,color=color)
 for seed in range(3):axes[1].plot(range(4),[next(r['new_states'] for r in rows if r['seed']==seed and r['label']==label) for label in labels],'o-',label=f'Seed {seed}')
 for offset,key,desc in [(-.18,'stationary_fraction','Stationary actions'),(.18,'moving_recent16_revisit_fraction','Recent revisit, given movement')]:
  axes[2].bar(np.arange(4)+offset,[means[l][key] for l in labels],width=.35,label=desc)
 axes[0].set(xlabel='Additional environment actions',ylabel='New states beyond original 30k replay',title='Mean discoveries; shading = seed range');axes[0].legend(title='MCTS simulations')
 axes[1].set(xticks=range(4),xticklabels=labels,ylabel='New states after 1,200 actions',title='Paired seeds, identical starting checkpoints');axes[1].legend()
 axes[2].set(xticks=range(4),xticklabels=labels,ylim=(0,1),title='Observed behavior during continuation');axes[2].legend(fontsize=8)
 for ax in axes:ax.grid(axis='y',alpha=.2)
 fig.tight_layout();fig.savefig('docs/assets/search-budget.png',dpi=160);plt.close(fig)
 lines=['# Larger MCTS budgets from mature checkpoints','',
 'Twelve controlled online branches: H16 shared 30k models, three seeds, budgets 24/96/384 plus persistent-random control, each given 1,200 additional real actions. Same horizon 16, discount 0.99, reward, model, optimizer, and full raw-replay epoch every ten actions. No loop penalty or new representation objective.','',
 '**Increasing simulations alone did not improve discovery consistently.** Mean additional states were 25.7 at 24 simulations, 26.0 at 96, 1.7 at 384, and 11.3 under persistent random. The large 384-versus-24 drop is driven by seed 0 (72→0); the other two seeds are unchanged. Treat this as a short, seed-dependent result on one maze, not a universal claim that more search hurts.','',
 'The 384 condition has more recent revisits among moving actions (61% versus 52% at 24), while final seen-transition accuracy remains about 99.3–99.5% for all conditions and disagreement–novelty AUROC remains about 0.80–0.82. Fitting and a globally informative reward coexist with weak exploration. The fixed-model audit below independently shows that larger searches optimize repeated disagreement without monotonically increasing novelty.','',
 'This supports testing a path-local repeat penalty at a moderate search budget. [Latent geometry diagnostics](latent-geometry.md) show that imagined-state distances can identify loops, but local latent scale varies and threshold calibration remains an open design choice. No loop penalty was tested in this study.','',
 'All branches restore model parameters, optimizer state, trainer RNG, and full 30k replay. They start at the fixed center, without warmup. Planner, schedule-shuffle and normalization RNGs restart identically per seed across branches. This is a controlled continuation protocol, not a bit-exact extension of the old online process. Higher budgets consume different planner RNG draws, so epsilon decisions are not common random numbers.','',
 '| Simulations | New states, seeds 0 / 1 / 2 | Mean new states | Mean cumulative discoveries | Stationary | Recent revisit given movement | Search seconds |',
 '|---|---|---:|---:|---:|---:|---:|']
 for label in labels:
  rs=sorted([r for r in rows if r['label']==label],key=lambda r:r['seed']);m=means[label]
  lines.append(f'| {label} | '+ ' / '.join(str(r['new_states']) for r in rs)+f' | {m["new_states"]:.1f} | {m["mean_cumulative_new_states"]:.1f} | {m["stationary_fraction"]:.1%} | {m["moving_recent16_revisit_fraction"]:.1%} | {m["search_seconds"]:.0f} |')
 lines+=['','“New” is relative to each seed’s original 30k visited set. The three checkpoints initially visited 1,429/1,533/1,621 of 1,784 states, leaving different opportunities. Mean cumulative discoveries is the interaction-averaged count of new states, rewarding earlier discovery. Stationary includes explicit stay and wall collisions. Recent revisit means a moving action returns to any source position in the last 16 transitions. Search seconds include action selection only; measurements share the machine with other workers and are not isolated benchmarks.','',
 '![Budget results](assets/search-budget.png)','',
 '## Final fitting','',
 '| Simulations | Seen-edge successor retrieval | Disagreement→new successor AUROC |',
 '|---|---:|---:|',
 *[f'| {label} | {means[label]["seen_edge_accuracy"]:.2%} | {means[label]["novel_successor_disagreement_auc"]:.3f} |' for label in labels],
 '', 'Accuracy uses the ensemble mean and the full maze latent catalogue, restricted to actually tried edges. Novelty AUROC uses actions at visited states; the definition of unvisited differs across branches after exploration. These are final-checkpoint metrics, not proof that fitting stayed equally good at every intermediate update.','',
 '## Trajectories and reward maps','',
 'Seed 0, all three MCTS budgets: [exploration animation](assets/search-budget-paths.gif) covers actions 30,000–31,200. [Reward-map animation](assets/search-budget-rewards.gif) shows pre-action snapshots every 100 actions through 31,100; it does not include the last 100 actions. These are illustrative single-seed views, while the tables aggregate all seeds.','',
 '![Reward maps and recent paths](assets/search-budget-rewards.png)','',
 '## Validation and limits','',
 'Every branch preserves all original replay arrays exactly, appends exactly 1,200 transitions, and trains through transition 31,200. Initial reward-map arrays and center positions are bit-identical across budgets for each seed. Configurations differ only in simulation budget/behavior, total duration, and snapshot frequency. All branches perform the same number of joint updates.','',
 'These are short online branches from three mature models in one deterministic coordinate maze. They do not establish from-scratch performance, performance at 30k extra actions, or behavior under noisy observations. Search budget changes both behavior and subsequent data; those feedback effects are part of the online comparison. Larger search can optimize a repeated-disagreement objective more thoroughly without necessarily improving discovery.','',
 '## Reproduction','',
 '`scripts/run_search_branches.py` launches four isolated subprocesses at a time and refuses existing output directories. `scripts/continue_search_budget.py` is the individual-run entry point. `scripts/report_search_budget.py` validates all twelve completed branches and generates this report and portable JSON. Raw logs, source snapshots, replay, checkpoints and reward maps are in `runs/search-budget-20261110`.','']
 audit=json.loads(Path('docs/assets/mcts-budget-audit.json').read_text())
 section=['## Fixed-model MCTS path audit','',
 'For each original H16/shared 30k checkpoint, 24 uniformly sampled visited starts are shared across budgets. Extract the path that follows maximum visit counts at each node (root tie-breaking follows the actual planner; deeper ties choose the lowest action index). Each is a 16-step open-loop diagnostic, not the closed-loop online trajectory. The inspection wrapper verifies its root action, visit counts and Q estimates against production MCTS before evaluation.','',
 '| Simulations | New states / path | Repeated successor steps / 16 | Unique path states (including start) | Imagined return / 24 baseline | True-state return / 24 baseline |',
 '|---|---:|---:|---:|---:|---:|']
 for budget in [24,96,384]:
  rs=[r for r in audit if r['budget']==budget]
  vals=[np.mean([r[k] for r in rs]) for k in ['new_states','repeated_successors','unique_path_states']]
  gains=[]
  for key in ['imagined_return','real_state_return']:
   gains.append(np.mean([np.mean([r[key] for r in rs if r['seed']==seed])/np.mean([r[key] for r in audit if r['seed']==seed and r['budget']==24]) for seed in range(3)]))
  section.append(f'| {budget} | '+' | '.join(f'{v:.2f}' for v in vals+gains)+' |')
 section+=['','Larger budgets find greater imagined reward and greater disagreement along the actual simulator paths, but also more repetition and fewer distinct path states. New-state yield is not monotonic. True-state reward is still the learned disagreement evaluated at actual states, not an oracle novelty reward. Return ratios normalize within each model before averaging seeds. Counts of repeated successors include stationary moves; raw outputs retain stationary counts separately.','']
 offset=lines.index('## Validation and limits');lines[offset:offset]=section
 Path('docs/search-budget.md').write_text('\n'.join(lines));print(json.dumps(means,indent=2))
if __name__=='__main__':main()
