"""Compare online joint-MSE cadences, fitting quality and exploration."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',nargs='+',required=True,type=Path)
    p.add_argument('--retrieval',required=True,type=Path)
    p.add_argument('--calibration',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    args=p.parse_args()
    rows=[]
    for root in args.runs:
        manifest=json.loads((root/'manifest.json').read_text())
        items=json.loads((root/'summary.json').read_text())
        assert len(items)==len(manifest['seeds'])*len(manifest['variants'])
        rows.extend(items)
    retrieval={r['run']:r for r in json.loads(args.retrieval.read_text())}
    calibration={r['run']:r for r in json.loads(args.calibration.read_text())}
    variants=list(dict.fromkeys(r['variant'] for r in rows))
    assets=args.out.parent/'assets';assets.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    colors=plt.cm.tab10(np.linspace(0,1,len(variants)))
    portable=[]
    for color,name in zip(colors,variants):
        group=[r for r in rows if r['variant']==name]
        for index,row in enumerate(group):
            path=Path(row['run']);curve=[json.loads(s) for s in (path/'exploration.jsonl').read_text().splitlines()]
            cadence=[json.loads(s) for s in (path/'cadence.jsonl').read_text().splitlines()]
            x=[r['environment_step'] for r in curve];coverage=[100*r['visited_positions']/row['metrics']['oracle_nodes'] for r in curve]
            axes[0,0].plot(x,coverage,color=color,alpha=.65,label=name if index==0 else None)
            axes[0,1].plot([r['seconds']/60 for r in curve],coverage,color=color,alpha=.65)
            windows=[cadence[i:i+50] for i in range(0,len(cadence),50)]
            wx=[np.mean([r['environment_step'] for r in w]) for w in windows]
            axes[1,0].plot(wx,[np.mean([r['post']['old']['normalized_mse'] for r in w]) for w in windows],color=color,alpha=.65)
            axes[1,1].plot(wx,[np.mean([r['post']['old']['disagreement'] for r in w]) for w in windows],color=color,alpha=.65)
            last=[r for r in cadence if r['environment_step']>row['metrics']['environment_steps']-1000]
            row={**row,'retrieval':retrieval[row['run']], 'calibration':calibration[row['run']], 'training_seconds':curve[-1]['training_seconds'],
                 'guaranteed_trained_through':curve[-1]['guaranteed_trained_through'],
                 'late':{g:{metric:{phase:float(np.mean([r[phase][g][metric] for r in last])) for phase in ['pre','post']}
                 for metric in ['normalized_mse','disagreement']} for g in ['new','old','probe']}}
            portable.append(row)
    for ax,title,xlabel,ylabel in [(axes[0,0],'Coverage versus interactions','Real actions','Reachable positions visited (%)'),
        (axes[0,1],'Coverage versus elapsed time','Minutes (including diagnostics)','Reachable positions visited (%)'),
        (axes[1,0],'Old-replay prediction error after training','Real actions','Normalized latent MSE'),
        (axes[1,1],'Old-replay disagreement after training','Real actions','Normalized mean variance')]:
        ax.set(title=title,xlabel=xlabel,ylabel=ylabel);ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    axes[1,0].set_yscale('log');axes[1,1].set_yscale('log')
    fig.tight_layout();fig.savefig(assets/'joint-exploration-curves.png',dpi=150);plt.close(fig)
    (assets/'joint-exploration-results.json').write_text(json.dumps(portable,indent=2)+'\n')
    text=['# Online exploration with the joint MSE model — 2026-11-09','',
          'Fresh three-layer width-256 encoder and three dynamics members, eight unbounded latents. Joint mean MSE + 0.01 hinge repulsion. Constant dynamics LR 3e-4 and encoder LR 9e-5. No actor, frozen representation, coordinate target, reconstruction, state codebook, or oracle guidance.','',
          'Three seeds per condition, 10k real actions each, the same 12×12 maze, center resets every 2,000 actions, and 1,000 persistent-random warmup actions. Full epochs replace the original replay updates. New-once explicitly trains all new transitions, then takes five random replay updates. All raw transitions remain in replay; repeated visits retain their weighting. This differs from unique-edge sampling and cosine decay in the fixed-dataset memorization tests.','',
          'MCTS replans after every real action. Reward is ensemble variance of predicted means normalized by replay latent variance. Rollouts follow the deterministic ensemble mean. Horizon 4, 24 simulations, discount 0.9, epsilon 0.1 unless noted in a variant. UCT bonus is scaled to root maximum reward times the discounted horizon sum to remain commensurate with small MSE rewards. Gaussian variance outputs do not enter reward or rollouts.','',
          '**Finding:** replay cadence strongly changes retained prediction accuracy, but improved fitting does not yet beat persistent-random exploration. On identical random-control trajectories, full epochs achieve 90.7% seen-edge accuracy versus 56.4% for new-once + 5. The worse-fitting ensemble has about 4.2× less disagreement, showing that agreement alone is not a reliable fitting/convergence criterion.', '',
          'The four-step MCTS full-epoch runs have local untried-action AUROC 0.46–0.52 at the final checkpoint. This is evidence that the current reward does not reliably distinguish untried actions at visited states; it is not proof that all untried actions should be uncertain, nor a causal explanation of every online action.', '',
          '![Exploration and fitting](assets/joint-exploration-curves.png)','',
          '## Results','',
          'Means across seeds. Coverage is exact reachable-position coverage (1,784 positions); AUC is mean coverage across the action budget. Accuracy is evaluation-only nearest-latent next-position retrieval. Seen accuracy weights observed state-actions uniformly, not by visit count.','',
          '| Condition | Final coverage (range) | Coverage AUC | Updates | Elapsed min | Seen accuracy | Unseen accuracy | Seen disagreement | Unseen disagreement |',
          '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name in variants:
        group=[r for r in portable if r['variant']==name]
        mean=lambda f:float(np.mean([f(r) for r in group]))
        cov=[r['coverage_lattice_stats']['final'] for r in group]
        text.append(f"| {name} | {100*np.mean(cov):.1f}% ({100*min(cov):.1f}–{100*max(cov):.1f}) | {100*mean(lambda r:r['coverage_lattice_stats']['auc']):.1f}% | {mean(lambda r:r['metrics']['updates']):,.0f} | {mean(lambda r:r['collection_seconds'])/60:.2f} | {100*mean(lambda r:r['retrieval']['seen']['exact_next_position_accuracy']):.1f}% | {100*mean(lambda r:r['retrieval']['unseen']['exact_next_position_accuracy']):.1f}% | {mean(lambda r:r['metrics']['oracle_seen_disagreement']):.3g} | {mean(lambda r:r['metrics']['oracle_unseen_disagreement']):.3g} |")
    text+=['','## Immediate effect of training','',
           'Pre → post means over the last 100 training events per run; identical diagnostic samples before and after each event. Disagreement is normalized mean variance, not JS divergence.','',
           '| Condition | New-transition MSE | Old-replay MSE | New-transition disagreement | Old-replay disagreement |',
           '|---|---:|---:|---:|---:|']
    for name in variants:
        group=[r for r in portable if r['variant']==name];cells=[]
        for g,metric in [('new','normalized_mse'),('old','normalized_mse'),('new','disagreement'),('old','disagreement')]:
            a,b=[np.mean([r['late'][g][metric][phase] for r in group]) for phase in ['pre','post']]
            cells.append(f'{a:.3g} → {b:.3g}')
        text.append('| '+name+' | '+' | '.join(cells)+' |')
    text+=['','## Does disagreement rank untried actions highly?','',
           'AUROC ranks untried actions against tried actions; 0.5 is chance, 1 is perfect ranking. The local score restricts both groups to physically visited states, avoiding an easy separation of faraway unseen states. These are final-checkpoint diagnostics, not a causal explanation of every online decision. A well-generalized action need not be uncertain merely because it was untried.', '',
           '| Condition | Unseen-edge AUROC | Local untried-action AUROC | Stationary real steps | Action switches |',
           '|---|---:|---:|---:|---:|']
    for name in variants:
        group=[r['calibration'] for r in portable if r['variant']==name]
        text.append('| '+name+' | '+' | '.join(f'{np.mean([r[k] for r in group]):.3f}' for k in ['unseen_edge_auc','untried_action_at_visited_state_auc','stationary_fraction','action_switch_fraction'])+' |')
    text+=['','## Matched-data control and interpretation','',
        'The paired random-epoch and random-new runs were verified to have identical actions, observations and positions for each seed. With the data stream held fixed, their final replay normalized MSE is 5.87e-5 versus 5.30e-4, while disagreement is 1.35e-5 versus 3.23e-6. Physical seen-edge accuracy is 90.7% versus 56.4%. Restricting the retrieval catalogue to visited states changes these accuracies only to 90.8% and 56.5%, so unvisited catalogue points do not explain the fitting gap.', '',
        'These online fits do not retain the ≥98% memorization level demonstrated on a fixed dataset. Raw replay frequency weighting, continual target drift and constant learning rates differ from that offline setup. We have not isolated which change is responsible. Full epochs improve fitting substantially; they do not establish that fitting is solved.', '',
        '## Exploration replay', '',
        'Seed 0 at matched interaction counts. This single-seed animation illustrates behavior; aggregate conclusions use all three seeds.', '',
        '![Exploration replay](assets/joint-exploration.gif)', '',
        '## Limits / artifacts','',
        'This is one deterministic coordinate-observation maze. Online trajectories differ between policies, so end-of-run fitting is not a fixed-data comparison. Final seen/unseen sets differ by trajectory. Latent MSE changes units with the encoder; physical retrieval is a common readout, not an optimal decoder. Runs use three concurrent workers, so wall times are observed costs rather than isolated throughput benchmarks.','',
        'The horizon-1 and new-once + 20 comparisons were chosen adaptively after inspecting initial results. They share seeds and layout; they are diagnostic follow-ups, not untouched validation. Several concurrent workers share the machine, including overlapping adaptive runs, which further limits throughput comparisons.', '',
        '[Portable numeric results](assets/joint-exploration-results.json) include per-run metrics and per-member accuracies. Raw configurations, source snapshots, weights, replay and diagnostics are under the run directories. The primary manifest is `configs/joint_exploration_cadence.json`; the base model/planner configuration is `configs/joint_exploration_base.json`.','']
    text += ['', '## Reproduction', '', '```bash',
        'MPLCONFIGDIR=/tmp/map-live-mpl .venv/bin/python scripts/run_probabilistic_sweep.py --manifest configs/joint_exploration_cadence.json --out runs/joint-exploration-cadence-20261109 --jobs 3',
        'MPLCONFIGDIR=/tmp/map-live-mpl .venv/bin/python scripts/run_probabilistic_sweep.py --manifest configs/joint_exploration_horizon1.json --out runs/joint-exploration-horizon1-20261109 --jobs 2',
        'MPLCONFIGDIR=/tmp/map-live-mpl .venv/bin/python scripts/run_probabilistic_sweep.py --manifest configs/joint_exploration_new20.json --out runs/joint-exploration-new20-20261109 --jobs 2',
        '.venv/bin/python scripts/evaluate_next_state_retrieval.py --summaries runs/joint-exploration-cadence-20261109/summary.json runs/joint-exploration-horizon1-20261109/summary.json runs/joint-exploration-new20-20261109/summary.json --out runs/joint-exploration-cadence-20261109/retrieval.json',
        '.venv/bin/python scripts/analyze_exploration_uncertainty.py --summaries runs/joint-exploration-cadence-20261109/summary.json runs/joint-exploration-horizon1-20261109/summary.json runs/joint-exploration-new20-20261109/summary.json --out runs/joint-exploration-cadence-20261109/calibration.json',
        'MPLCONFIGDIR=/tmp/map-live-mpl .venv/bin/python scripts/report_joint_exploration.py --runs runs/joint-exploration-cadence-20261109 runs/joint-exploration-horizon1-20261109 runs/joint-exploration-new20-20261109 --retrieval runs/joint-exploration-cadence-20261109/retrieval.json --calibration runs/joint-exploration-cadence-20261109/calibration.json --out docs/joint-exploration.md',
        'MPLCONFIGDIR=/tmp/map-live-mpl .venv/bin/python scripts/animate_exploration.py runs/joint-exploration-cadence-20261109/random_epoch-seed0 runs/joint-exploration-cadence-20261109/mcts_epoch-seed0 runs/joint-exploration-cadence-20261109/mcts_new-seed0 runs/joint-exploration-new20-20261109/mcts_new20-seed0 --labels "Persistent random" "MCTS: full epoch" "MCTS: new + 5" "MCTS: new + 20" --end 10000 --stride 50 --out runs/joint-exploration.gif',
        '```', '']
    args.out.write_text('\n'.join(text));print(args.out)


if __name__=='__main__':main()
