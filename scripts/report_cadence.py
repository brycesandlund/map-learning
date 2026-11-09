"""Summarize completed cadence runs, paired learning diagnostics, and physical readouts."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LABELS = {'replay': 'Replay baseline', 'epoch': 'Full epoch', 'new_then_replay': 'New once + replay'}
COLORS = {'replay': '#777777', 'epoch': '#2276b4', 'new_then_replay': '#e87923'}


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    rows = json.loads((args.runs/'summary.json').read_text())
    manifest = json.loads((args.runs/'manifest.json').read_text())
    assert len(rows) == len(manifest['seeds'])*len(manifest['variants']), 'Wait for all runs'
    retrieval = {(r['variant'], r['seed']): r for r in json.loads((args.runs/'retrieval.json').read_text())}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    assets = args.out.parent/'assets'
    assets.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    detailed = []
    for row in rows:
        key, seed = row['variant'], row['seed']
        path = Path(row['run'])
        curve = read_lines(path/'exploration.jsonl')
        cadence = read_lines(path/'cadence.jsonl')
        color = COLORS[key]
        label = LABELS[key] if seed == manifest['seeds'][0] else None
        x = np.array([r['environment_step'] for r in curve])
        cov = 100*np.array([r['visited_positions']/row['metrics']['oracle_nodes'] for r in curve])
        axes[0, 0].plot(x, cov, color=color, alpha=.65, label=label)
        axes[0, 1].plot([r['seconds']/60 for r in curve], cov, color=color, alpha=.65)
        # Bin event diagnostics into consecutive windows of 50 training events.
        windows = [cadence[i:i+50] for i in range(0, len(cadence), 50)]
        wx = [np.mean([r['environment_step'] for r in w]) for w in windows]
        for ax, group, metric in [(axes[0, 2], 'old', 'disagreement'),
                                   (axes[1, 0], 'new', 'disagreement'),
                                   (axes[1, 1], 'probe', 'normalized_mse')]:
            y = [np.mean([r['post'][group][metric] for r in w]) for w in windows]
            ax.plot(wx, y, color=color, alpha=.65)
        axes[1, 2].plot(wx, [np.mean([r['post']['probe_latent_variance'] for r in w]) for w in windows], color=color, alpha=.65)
        late = [r for r in cadence if r['environment_step'] > 9000]
        diagnostics = {group: {metric: {
            phase: float(np.mean([r[phase][group][metric] for r in late])) for phase in ('pre', 'post')}
            for metric in ('disagreement', 'normalized_mse', 'nll_per_dim', 'predictive_variance')}
            for group in ('new', 'old', 'probe')}
        detail = {'variant': key, 'seed': seed, 'coverage': row['coverage_lattice_stats']['final'],
                  'coverage_auc': row['coverage_lattice_stats']['auc'], 'updates': row['metrics']['updates'],
                  'seconds': row['collection_seconds'], 'training_seconds': curve[-1]['training_seconds'],
                  'guaranteed_trained_through': curve[-1]['guaranteed_trained_through'],
                  'late_diagnostics': diagnostics, 'retrieval': retrieval[key, seed], 'metrics': row['metrics']}
        detailed.append(detail)
    panels = [(axes[0, 0], 'Exploration by action budget', 'Environment actions', 'Reachable positions visited (%)'),
              (axes[0, 1], 'Exploration by elapsed time', 'Minutes (includes diagnostics)', 'Reachable positions visited (%)'),
              (axes[0, 2], 'Old replay: disagreement after fitting', 'Environment actions', 'JS divergence (nats)'),
              (axes[1, 0], 'Latest transitions: disagreement after fitting', 'Environment actions', 'JS divergence (nats)'),
              (axes[1, 1], 'Fixed maze probe: prediction error', 'Environment actions', 'Normalized latent MSE'),
              (axes[1, 2], 'Fixed maze probe: latent spread', 'Environment actions', 'Mean coordinate variance')]
    for ax, title, xlabel, ylabel in panels:
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        ax.grid(alpha=.2)
    axes[0, 0].legend()
    fig.suptitle('Training cadence • three seeds per condition • single-Gaussian MCTS', fontsize=15)
    fig.tight_layout()
    plot = assets/'cadence-comparison.png'
    fig.savefig(plot, dpi=150)
    plt.close(fig)
    (args.runs/'cadence-summary.json').write_text(json.dumps(detailed, indent=2, allow_nan=False)+'\n')
    # Keep compact numeric results in the repo, independently of ignored raw runs.
    (assets/'cadence-results.json').write_text(json.dumps(detailed, indent=2, allow_nan=False)+'\n')
    text = ['# Training cadence experiment — 2026-11-09', '',
        'Nine completed 10,000-action runs: three seeds (0–2) per schedule. All use the same 12×12 coordinate maze, fixed-center resets every 2,000 actions, 1,000 random warmup actions, tanh encoder (8 latents, width 64), three single-Gaussian heads, batch size 128, and MCTS horizon 4 / 24 simulations.', '',
        '- **Replay baseline:** five random replay updates every 10 actions.',
        '- **Full epoch:** every transition once per head, independently shuffled, every 10 actions; replaces random replay updates.',
        '- **New once + replay:** every new transition once per head, then five random replay updates. Includes the full warmup buffer at the first event.', '',
        'Partial batches are retained. Both new schedules reached all 10,000 transitions in every head. Encoder and dynamics train jointly; each update also samples 128 replay-state pairs for repulsion. No new policy is learned.', '',
        'Both added-training schedules reduced final mean replay disagreement, but neither improved mean physical next-position retrieval. Full epochs modestly increased mean coverage with mixed per-seed effects; new-once training left mean final coverage essentially unchanged. This does not establish insufficient fitting as the main exploration bottleneck.', '',
        '![Cadence comparison](assets/cadence-comparison.png)', '',
        '## Final results', '',
        'Means across three seeds; coverage also shows the seed range. Physical readout uses nearest-latent retrieval from a privileged full-maze catalogue, only for evaluation.', '',
        '| Schedule | Final coverage (range) | Coverage AUC | Updates | Elapsed min | Training min | All-edge exact retrieval | Physical error |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for key in LABELS:
        group = [r for r in detailed if r['variant'] == key]
        mean = lambda f: np.mean([f(r) for r in group])
        text.append(f"| {LABELS[key]} | {mean(lambda r: r['coverage'])*100:.1f}% ({min(r['coverage'] for r in group)*100:.1f}–{max(r['coverage'] for r in group)*100:.1f}) | {mean(lambda r: r['coverage_auc'])*100:.1f}% | {group[0]['updates']:,} | {mean(lambda r: r['seconds'])/60:.2f} | {mean(lambda r: r['training_seconds'])/60:.2f} | {mean(lambda r: r['retrieval']['all']['exact_next_position_accuracy'])*100:.1f}% | {mean(lambda r: r['retrieval']['all']['mean_physical_error']):.4f} |")
    text += ['', 'For context, always predicting the current position gives 31.9% exact accuracy and 0.1702 mean physical error on this maze. A move is 0.25 units. All three conditions have larger mean physical error under this readout; global latent spread alone does not establish a useful map.', '', '## Final ensemble agreement', '',
             '| Schedule | Replay JS | Seen-edge JS | Unseen-edge JS | Latent variance |',
             '|---|---:|---:|---:|---:|']
    for key in LABELS:
        group = [r for r in detailed if r['variant'] == key]
        cells = [f"{np.mean([r['metrics'][metric] for r in group]):.4f}" for metric in
                 ('replay_disagreement', 'oracle_seen_disagreement', 'oracle_unseen_disagreement', 'latent_variance')]
        text.append('| '+LABELS[key]+' | '+' | '.join(cells)+' |')
    text += ['', 'Replay JS is visitation-weighted; seen/unseen edge JS weights distinct physical state-actions uniformly. Final estimates use 32 Monte Carlo draws per member.', '', '## What each training block changes', '',
             'Pre → post means over the final 100 training events of each seed (actions 9,010–10,000). Both measurements use identical transitions and common Monte Carlo seeds. JS estimates use 8 draws/member, so small differences are noisy.', '',
             '| Schedule | Latest-10 JS | Older-replay JS | Fixed-probe JS | Fixed-probe normalized MSE |',
             '|---|---:|---:|---:|---:|']
    for key in LABELS:
        group = [r for r in detailed if r['variant'] == key]
        cells = []
        for label, metric in [('new', 'disagreement'), ('old', 'disagreement'), ('probe', 'disagreement'), ('probe', 'normalized_mse')]:
            values = [np.mean([r['late_diagnostics'][label][metric][phase] for r in group]) for phase in ('pre', 'post')]
            cells.append(f'{values[0]:.4f} → {values[1]:.4f}')
        text.append('| '+LABELS[key]+' | '+' | '.join(cells)+' |')
    text += ['', '## Interpretation limits', '',
        'Online trajectories differ between conditions. Maze-wide probes never influence training or planning but may overlap experienced edges; final metrics in the JSON separately report seen/unseen edges. Normalized latent error and NLL still depend on the learned representation. Physical retrieval is a shared readout, not an optimal decoder. Lower disagreement alone does not establish accurate prediction or calibrated uncertainty. Full epochs also increase the number of repulsion updates, not only dynamics fitting. The new-transition batch receives one full optimizer step despite being smaller than a replay batch.', '',
        'Three workers ran concurrently on one machine, so wall times are observed costs, not isolated throughput benchmarks. Diagnostic overhead is included in elapsed time and excluded from training time. These are three seeds on one deterministic coordinate maze.', '',
        '## Exploration replay', '', 'Seed 0, all three schedules at matched environment steps (50 transitions per frame):', '',
        '![Cadence exploration](assets/cadence-exploration.gif)', '',
        '## Reproduction', '', '```bash',
        'MPLCONFIGDIR=/tmp/map-cadence-mpl .venv/bin/python scripts/run_probabilistic_sweep.py --manifest configs/probabilistic_cadence.json --out runs/probabilistic-cadence-20261109 --jobs 3',
        '.venv/bin/python scripts/evaluate_next_state_retrieval.py --summaries runs/probabilistic-cadence-20261109/summary.json --out runs/probabilistic-cadence-20261109/retrieval.json',
        'MPLCONFIGDIR=/tmp/map-cadence-mpl .venv/bin/python scripts/report_cadence.py --runs runs/probabilistic-cadence-20261109 --out docs/training-cadence.md',
        "MPLCONFIGDIR=/tmp/map-cadence-mpl .venv/bin/python scripts/animate_exploration.py runs/probabilistic-cadence-20261109/replay-seed0 runs/probabilistic-cadence-20261109/epoch-seed0 runs/probabilistic-cadence-20261109/new_then_replay-seed0 --labels 'Replay baseline' 'Full epoch' 'New once + replay' --end 10000 --stride 50 --out runs/cadence-exploration.gif",
        '```', '', 'Raw runs (ignored by git): `runs/probabilistic-cadence-20261109/`. Portable numeric results: [cadence-results.json](assets/cadence-results.json). Each run includes resolved config, source snapshot, checkpoint, raw replay, before/after diagnostics, coverage, and final full-maze evaluations.', '']
    args.out.write_text('\n'.join(text))
    print(args.out)


if __name__ == '__main__':
    main()
