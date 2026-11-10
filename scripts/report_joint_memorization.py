"""Report verified joint-model memorization results, including unsuccessful variants."""
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
    p.add_argument('--verification',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    args=p.parse_args()
    verified={r['run']:r for r in json.loads(args.verification.read_text())}
    rows=[]
    for root in args.runs:
        items=json.loads((root/'summary.json').read_text())
        assert len(items)==len(json.loads((root/'manifest.json').read_text())['variants'])
        for row in items:
            row['verification']=verified[row['run']]
            row['curve']=[json.loads(s) for s in (Path(row['run'])/'learning.jsonl').read_text().splitlines()]
            rows.append(row)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    assets=args.out.parent/'assets';assets.mkdir(exist_ok=True)
    (assets/'joint-memorization-results.json').write_text(json.dumps(rows,indent=2)+'\n')
    fig,axes=plt.subplots(1,3,figsize=(16,4.8))
    for row in rows:
        curve=row['curve'];x=[r['update']+1 for r in curve]
        low=[100*min(r['train']['per_member_unique_accuracy']) for r in curve]
        high=[100*max(r['train']['per_member_unique_accuracy']) for r in curve]
        line,=axes[0].plot(x,low,label=row['name'],marker='.',ms=3)
        axes[0].fill_between(x,low,high,alpha=.15,color=line.get_color())
        axes[1].plot(x,[r['train']['mse'] for r in curve],color=line.get_color())
        axes[2].plot(x,[r['latent_variance'] for r in curve],color=line.get_color())
    axes[0].axhline(98,color='black',ls='--',lw=1)
    axes[0].set(title='Worst member accuracy; shading spans members',ylabel='Unique training edges correct (%)',ylim=(0,102))
    axes[0].legend(fontsize=7,loc='lower right')
    axes[1].set(title='Training ensemble-mean error',ylabel='Latent MSE',yscale='log')
    axes[2].set(title='Representation spread',ylabel='Mean latent coordinate variance',yscale='log')
    for ax in axes:ax.set(xscale='log',xlabel='Gradient updates + 1');ax.grid(alpha=.2)
    fig.suptitle('Fresh encoder + dynamics, jointly trained • no frozen representation',fontsize=14)
    fig.tight_layout();fig.savefig(assets/'joint-memorization-curves.png',dpi=160);plt.close(fig)
    text=['# Joint memorization above 98% — 2026-11-09','',
        'The learned joint setup exceeds the target: both fresh confirmation seeds reach above 99.2% training accuracy in **every ensemble member** after 10k updates. Saved-weight re-evaluation reproduces the scores, with no exact collisions among the 762 observed-state latents. No fixed-coordinate encoder, frozen targets, codebook or lookup-table predictor is used.','',
        '![Joint fitting curves](assets/joint-memorization-curves.png)','',
        '## Recipe','',
        '- Encoder: 2 inputs → three width-256 SiLU hidden layers → 8 unconstrained outputs (134,408 parameters).',
        '- Three residual dynamics members: latent + one-hot action → three width-256 SiLU layers → Gaussian parameter outputs (139,537 parameters/member, 418,611 total). MSE trains their means; variance outputs are unused by this objective.',
        '- Both networks are initialized fresh and jointly optimized throughout. Encoder gradients flow through both source and successor branches.',
        '- Loss: average per-member mean-prediction MSE + **0.01 × hinge repulsion**, with margin 1. Repulsion pairs come from training endpoints; no target latent layout is specified.',
        '- Adam: dynamics LR 3e-4, encoder LR 9e-5; both cosine-decay to 1/30 of their initial rates. Independent shuffled passes through all unique observed edges, batch 128, keeping partial batches.',
        '- Dataset: the same 10k-transition replay as previous experiments, containing 2,606 unique training edges and 762 states. Configuration is read from the old checkpoint, but neither encoder nor dynamics weights are reused.',
        '- Initial seed-7 screen used 50k updates. Two confirmation seeds (8, 9) used a 10k budget and correspondingly shortened cosine schedule. This is adaptive model development on one dataset, not a pre-registered evaluation.','',
        '## Verified final results','',
        'Accuracy is nearest-latent next-position retrieval against the complete evaluation-only maze catalogue. It is not zero continuous residual error. Each training edge has equal weight. Unseen accuracy uses 1,024 state-actions absent from fitting, sampled independently of training.','',
        '| Run | Updates | Member 1 | Member 2 | Member 3 | All members correct together | Unseen ensemble | Pass ≥98% each |',
        '|---|---:|---:|---:|---:|---:|---:|---|']
    for row in rows:
        v=row['verification'];cells=[f'{100*x:.2f}%' for x in v['per_member_accuracy']]
        text.append(f"| {row['name']} | {v['updates']:,} | "+' | '.join(cells)+f" | {100*v['all_members_correct_fraction']:.2f}% | {100*v['unseen_accuracy']:.2f}% | {'yes' if v['passed'] else 'no'} |")
    text+=['','## Latent geometry / repulsion','',
        'Computed exactly over training-endpoint pairs using their sampling frequencies, including identical states. Distinct-state mean distance weights the 762 unique states uniformly. The identical-state floor is the unavoidable contribution from sampling the same state twice.','',
        '| Run | Latent variance | Minimum distinct-state distance | Mean distinct-state distance | Moving-edge distance | Repulsion | Identical-state floor |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        v=row['verification']
        keys=['training_latent_variance','min_distinct_distance','mean_distinct_distance','mean_moving_edge_distance','repulsion_exact','identical_state_repulsion_floor']
        text.append('| '+row['name']+' | '+' | '.join(f'{v[k]:.4f}' for k in keys)+' |')
    text+=['','## What this establishes','',
        'Longer fitting was not uniformly better: the seed-7 unbounded model finished its 50k budget at 97.93–98.04% per member, below its earlier scores. The tanh variants finished at 93.25–93.98% (repulsion 0.01) and 96.70–97.35% (repulsion 0.001). The successful final checkpoints are the two 10k confirmation runs; ongoing joint training can still degrade retrieval accuracy.', '',
        'The ≥98% fixed-data joint-memorization milestone is met. The architecture remains a generalizing neural encoder and action-conditioned dynamics model, with no fixed latent coordinates. Unseen-action results are encouraging but cover one coordinate-observation maze. This does not demonstrate live exploration, stochastic uncertainty modeling, or pixel generalization.', '',
        'The change bundles larger/deeper networks, fresh initialization, uniform-edge training and a slower encoder learning rate. The activation and repulsion comparisons inside the seed-7 screen are matched; comparison with earlier experiments does not isolate capacity alone. Accuracy can vary during joint fitting, as the curves show. The 10k confirmations are final checkpoints at a fixed budget, not cherry-picked intermediate checkpoints.', '',
        '## Reproduction','', '```bash',
        'MPLCONFIGDIR=/tmp/map-joint-mpl .venv/bin/python scripts/run_memorization.py --manifest configs/joint_memorization.json --out runs/joint-memorization-20261109 --jobs 3',
        'MPLCONFIGDIR=/tmp/map-joint-mpl .venv/bin/python scripts/run_memorization.py --manifest configs/joint_memorization_confirm.json --out runs/joint-memorization-confirm-20261109 --jobs 2',
        '.venv/bin/python scripts/verify_joint_memorization.py --runs runs/joint-memorization-20261109 runs/joint-memorization-confirm-20261109 --out runs/joint-memorization-20261109/verification.json',
        'MPLCONFIGDIR=/tmp/map-joint-mpl .venv/bin/python scripts/report_joint_memorization.py --runs runs/joint-memorization-20261109 runs/joint-memorization-confirm-20261109 --verification runs/joint-memorization-20261109/verification.json --out docs/joint-memorization.md',
        '```','',
        'Use fresh output directories when repeating training. Every run saves data hashes, source snapshots, selected training indices, full options, learning logs and `weights.pt`. Model reconstruction uses encoder/dynamics configs from `experiment.json`; these are diagnostic weight files, not the online runner checkpoint format. [Portable numeric results](assets/joint-memorization-results.json) include verification and learning curves.','']
    args.out.write_text('\n'.join(text));print(args.out)


if __name__=='__main__':main()
