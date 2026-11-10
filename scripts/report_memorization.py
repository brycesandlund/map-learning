"""Portable tables and learning curves for fixed-data memorization diagnostics."""
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
    p.add_argument('--out',required=True,type=Path)
    args=p.parse_args()
    rows=[]
    for root in args.runs:
        manifest=json.loads((root/'manifest.json').read_text())
        items=json.loads((root/'summary.json').read_text())
        assert len(items)==len(manifest['variants']),f'Incomplete: {root}'
        rows.extend(items)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    assets=args.out.parent/'assets';assets.mkdir(exist_ok=True)
    (assets/'memorization-results.json').write_text(json.dumps(rows,indent=2)+'\n')
    groups=[('Matched 20k loss / encoder comparison',['frozen_nll','frozen_mse','joint_nll','joint_mse']),
            ('Frozen-MSE controls',['tiny128_frozen_mse','coordinates_mse','wide256_frozen_mse','uniform_edges_frozen_mse']),
            ('Longer fitting / repulsion follow-ups',['frozen64_uniform_100k','frozen256_uniform_100k','coordinates256_uniform_100k','joint_mse_repulsion001'])]
    fig,axes=plt.subplots(2,3,figsize=(17,9))
    for column,(title,names) in enumerate(groups):
        for row in rows:
            if row['name'] not in names:continue
            curve=[json.loads(line) for line in (Path(row['run'])/'learning.jsonl').read_text().splitlines()]
            x=[r['update']+1 for r in curve]
            axes[0,column].plot(x,[100*r['train']['unique_edge_exact_retrieval'] for r in curve],label=row['name'],marker='.',ms=4)
            axes[1,column].plot(x,[r['train']['mse'] for r in curve],marker='.',ms=4)
        axes[0,column].set(title=title,ylabel='Unique training edges correct (%)',ylim=(0,102))
        axes[0,column].legend(fontsize=8,loc='lower right')
        axes[1,column].set(ylabel='Training latent MSE',yscale='log')
        for ax in axes[:,column]:
            ax.set(xscale='log',xlabel='Gradient updates + 1');ax.grid(alpha=.2)
    fig.suptitle('Fixed-dataset fitting • one dataset and initialization seed • no new exploration',fontsize=15)
    fig.tight_layout();fig.savefig(assets/'memorization-curves.png',dpi=150);plt.close(fig)
    text=['# Fixed-data memorization experiments — 2026-11-09','',
          'All experiments fit the same saved seed-0 coordinate-maze replay: 10,000 transitions, 2,606 unique state-actions, 762 observed states. Identical observed state/actions have no conflicting targets. The pretrained encoder has 762 distinct float32 outputs on those states, with minimum separation 0.00957. No new exploration is performed.','',
          'The fixed dataset is much easier to fit than the online results suggested, but we have not achieved exact memorization of the entire set. The 128-edge control reaches 100% retrieval for every member; longer frozen-encoder MSE reaches 98.0–98.3% of all 2,606 distinct edges. A fixed-coordinate control reaches 99.65%. These do not mean zero continuous prediction residual.', '',
          'The cleanest joint-training result is the single-factor repulsion ablation: lowering its weight from 10 to 0.01 raises unique-edge accuracy from 56.75% to 90.94% at the same 20k budget, above frozen-MSE’s 86.19%. Joint training itself is not ruled out; objective balance matters.', '',
          '![Learning curves](assets/memorization-curves.png)','',
          '## Training-set results','',
          'Physical correctness means nearest-latent next-position retrieval against the full maze catalogue, used only by evaluation. It is a discrete correctness criterion, not zero continuous regression error. Uniform-edge runs weight each distinct training edge equally; whole-replay runs retain visitation frequency. Compare the unique-edge column for the common 2,606-edge training set. The tiny control fits only 128 edges.','',
          '| Experiment | Updates | Edges | Weighted accuracy | Unique-edge accuracy | Per-member accuracy¹ | Latent MSE | Unseen accuracy² |',
          '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        m,e=row['metrics'],row['experiment'];t=m['train']
        text.append(f"| {row['name']} | {m['update']:,} | {e['unique_training_edges']:,} | {100*t['exact_retrieval']:.2f}% | {100*t['unique_edge_exact_retrieval']:.2f}% | {100*t['individual_exact_retrieval']:.2f}% | {t['mse']:.3g} | {100*m['unseen']['exact_retrieval']:.2f}% |")
    text+=['','¹ Mean over all three members and training examples, with the same weighting as weighted accuracy. Other accuracy columns use the ensemble mean. ² Disjoint state-actions absent from fitting; an evaluator-only sample of up to 1,024 pairs. The tiny condition has a different unseen set.','',
           '## What is controlled','',
           '- All learned encoders start from the same saved replay-baseline checkpoint. Every dynamics ensemble starts freshly initialized; optimizer state is reset. Seed 7 is shared. The pretrained encoder has already seen this replay, so these are memorization diagnostics, not unsupervised generalization claims.',
           '- Frozen tests precompute inputs and targets once. Joint tests backpropagate through both sides. Coordinate controls repeat the observed normalized x/y to eight dimensions, with no learned encoder or hidden geometric input.',
           '- The four primary conditions use width 64, three single-Gaussian heads, 20k updates, batch size 128, and Adam with cosine learning rate decay from 3e-4 to 1e-5. Training shuffles each member independently and keeps partial batches.',
           '- NLL is negative log likelihood divided by eight latent dimensions. MSE is mean squared error of each member’s predictive mean. For joint runs, add 10 times the original hinge repulsion. This changes the relative loss scale between NLL and MSE; it is intentionally not presented as a scale-matched comparison.',
           '- Frozen/coordinate encoders omit constant repulsion. Gaussian variances are not trained by MSE; MSE runs do not report JS/NLL/variance as meaningful diagnostics.',
           '- Longer runs fit unique edges uniformly for 100k updates at initial learning rate 1e-3, cosine-decayed to 1e-3/30. Width 64 and 256 learned-encoder runs otherwise match. The coordinate control uses width 256.',
           '- The low-repulsion joint follow-up changes only repulsion weight from 10 to 0.01 relative to the 20k joint-MSE condition. It retains the initial 3e-4 learning rate and visitation-weighted replay.',
           '- Wider dynamics contain more parameters, while the frozen learned encoder remains exactly unchanged. The checkpoints, data hashes, fitting indices, source snapshots, and actual options are saved per run.','',
           '## Encoder gradient scale audit','',
           'At the common initial encoder and fresh dynamics, over 32 minibatches, the median ratio of weighted repulsion gradient norm to prediction gradient norm is **188.4 for MSE at weight 10**, **0.188 at weight 0.01**, and **3.77 for NLL at weight 10**. These are encoder gradients with the same minibatch shared across heads, an initialization diagnostic rather than proof of the mechanism throughout training. The 188× figure applies to the MSE experiment, not the original NLL-based online objective.', '',
           'Changing NLL to MSE without adjusting the repulsion coefficient is therefore not a neutral objective comparison. The matched coefficient ablation supports loss balance as a cause of poor joint-MSE fitting. Frozen NLL vs frozen MSE omits repulsion and still favors direct MSE on training retrieval.', '',
           '## Model / representation diagnostics','',
           '| Experiment | Dynamics parameters | Encoder drift MSE | Latent variance | Near-bound fraction |',
           '|---|---:|---:|---:|---:|']
    for row in rows:
        m,e=row['metrics'],row['experiment']
        sat=f"{m['saturation']*100:.1f}%" if m['saturation'] is not None else 'n/a'
        text.append(f"| {row['name']} | {e['dynamics_parameters']:,} | {m['encoder_drift_mse']:.4g} | {m['latent_variance']:.4f} | {sat} |")
    text+=['','## Reproduction','', '```bash',
           'MPLCONFIGDIR=/tmp/map-memorize-mpl .venv/bin/python scripts/run_memorization.py --manifest configs/memorization.json --out runs/memorization-20261109 --jobs 3',
           'MPLCONFIGDIR=/tmp/map-memorize-mpl .venv/bin/python scripts/run_memorization.py --manifest configs/memorization_followup.json --out runs/memorization-followup-20261109 --jobs 3',
           'MPLCONFIGDIR=/tmp/map-memorize-mpl .venv/bin/python scripts/report_memorization.py --runs runs/memorization-20261109 runs/memorization-followup-20261109 --out docs/memorization.md',
           '.venv/bin/python scripts/audit_memorization_gradients.py --checkpoint runs/probabilistic-cadence-20261109/replay-seed0/checkpoint.pt --data runs/probabilistic-cadence-20261109/replay-seed0/replay.npz --out runs/memorization-20261109/gradient-audit.json',
           '```','', 'Use fresh output directories to repeat training; existing experiment directories are never overwritten. Raw runs are ignored by git; [portable numeric results](assets/memorization-results.json) and this report are included in the repo.','',
           'This is one exploratory dataset/initialization, with follow-ups chosen after inspecting the first stage. It does not establish a general ranking of algorithms. Physical retrieval can be sensitive to latent geometry; raw MSE comparisons across changing encoders are not common-unit accuracy comparisons.','']
    args.out.write_text('\n'.join(text));print(args.out)


if __name__=='__main__':main()
