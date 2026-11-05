"""Aggregate completed sweep artifacts; keep different interaction budgets separate."""
import argparse
import json
from pathlib import Path
import numpy as np
from map_learning.evaluation import plotting
from run_probabilistic_sweep import coverage_statistics


def behavior_stats(path):
    with np.load(Path(path)/'replay.npz') as replay:
        actions = replay['actions']
        points = np.stack([replay['positions'], replay['next_positions']], axis=1).reshape(-1, 2)
        _, first = np.unique(np.round(points, 5), axis=0, return_index=True)
        first_steps = first//2+1
        last_start = 4*len(actions)//5
        return {'blocked_or_stay_fraction': float(np.mean(np.all(np.isclose(replay['positions'], replay['next_positions']), axis=1))),
                'action_change_fraction': float(np.mean(actions[1:] != actions[:-1])),
                'stay_action_fraction': float(np.mean(actions == 4)),
                'last_discovery_step': int(first_steps.max()),
                'new_positions_final_fifth': int((first_steps > last_start).sum()),
                'final_fifth_steps': len(actions)-last_start,
                'moving_fraction_final_fifth': float(np.mean(np.any(replay['positions'][last_start:] != replay['next_positions'][last_start:], axis=1)))}


def group(rows):
    return {name: [r for r in rows if r['variant'] == name] for name in sorted({r['variant'] for r in rows})}


def comparison_table(groups, prefix=''):
    text = ['| Setting | Seeds | Mean position AUC | Mean final positions | Mean final rooms | Position range |',
            '|---|---:|---:|---:|---:|---:|']
    for name, rows in groups.items():
        a = np.array([r[f'{prefix}coverage_lattice_stats']['auc'] for r in rows])*100
        f = np.array([r[f'{prefix}coverage_lattice_stats']['final'] for r in rows])*100
        rm = np.array([r[f'{prefix}coverage_room_stats']['final'] for r in rows])*100
        text.append(f'| {name} | {len(rows)} | {a.mean():.1f}% | {f.mean():.1f}% | {rm.mean():.1f}% | {f.min():.1f}–{f.max():.1f}% |')
    return text


def coverage_figure(plt, groups, budget, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, rows in groups.items():
        curves = [[v for v in (json.loads(line) for line in (Path(r['run'])/'exploration.jsonl').read_text().splitlines())
                   if v['environment_step'] <= budget] for r in rows]
        x = np.array([v['environment_step'] for v in curves[0]])
        for ax, field, title in zip(axes, ['coverage_lattice', 'coverage_rooms'], ['Reachable positions', 'Rooms']):
            ys = np.array([[v['visited_positions']/r['metrics']['oracle_nodes'] if field == 'coverage_lattice' else v[field]
                            for v in curve] for r, curve in zip(rows, curves)])*100
            display = {'random': 'Persistent random', 'random_iid': 'Uniform random',
                       'single_gaussian': 'Single Gaussian + MCTS', 'wide128': 'Width 128 + MCTS',
                       'horizon8': 'Horizon 8 + MCTS', 'base': 'Default + MCTS',
                       'unbounded': 'Unbounded + MCTS'}.get(name, name)
            line, = ax.plot(x, ys.mean(0), label=f'{display} (n={len(rows)})')
            ax.fill_between(x, ys.min(0), ys.max(0), alpha=.12, color=line.get_color())
            ax.set(xlabel='Real environment steps', ylabel='Coverage (%)', title=title, ylim=(0, 102))
            ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle(f'{budget:,}-step comparison; mean and min–max across fresh seeds')
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--screen', nargs='+', required=True)
    parser.add_argument('--long', nargs='+', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    screen = [r for folder in args.screen for r in json.loads((Path(folder)/'summary.json').read_text())]
    long = [r for folder in args.long for r in json.loads((Path(folder)/'summary.json').read_text())]
    common_budget = min(r['metrics']['environment_steps'] for r in long)
    max_budget = max(r['metrics']['environment_steps'] for r in long)
    for row in screen+long:
        row['behavior'] = behavior_stats(row['run'])
    for row in long:
        with np.load(Path(row['run'])/'replay.npz') as replay:
            p, q = replay['positions'][:common_budget], replay['next_positions'][:common_budget]
            config = json.loads((Path(row['run'])/'config.json').read_text())
            row['common_budget_steps'] = common_budget
            row['common_coverage_lattice_stats'] = coverage_statistics(p, q, row['metrics']['oracle_nodes'])
            row['common_coverage_room_stats'] = coverage_statistics(p, q, config['maze']['size']**2, True)
    screen.sort(key=lambda r: r['coverage_lattice_stats']['auc'], reverse=True)
    groups = group(long)
    full_groups = group([r for r in long if r['metrics']['environment_steps'] == max_budget])
    bundle = {'screen': screen, 'long': long}
    for name, filename in [('activation_fixed_data', 'activation-comparison.json'),
                           ('screen_retrieval', 'screen-retrieval.json')]:
        if (output/filename).exists():
            bundle[name] = json.loads((output/filename).read_text())
    (output/'results.json').write_text(json.dumps(bundle, indent=2, allow_nan=False)+'\n')
    plt = plotting()
    fig, axes = plt.subplots(1, 2, figsize=(12, max(7, .34*len(screen))), sharey=True)
    labels = [r['variant'] for r in screen]
    for ax, metric, title in zip(axes, ['auc', 'final'], ['Average position coverage over 10k steps', 'Position coverage at 10k steps']):
        ax.barh(labels, [100*r['coverage_lattice_stats'][metric] for r in screen],
                color=['#777777' if r['variant'] == 'random' else '#247ba0' for r in screen])
        ax.set(xlabel='Coverage (%)', title=title, xlim=(0, 100))
        ax.grid(axis='x', alpha=.2)
    axes[0].invert_yaxis()
    fig.suptitle('Architecture and hyperparameter screen: 12×12 maze, one seed')
    fig.tight_layout()
    fig.savefig(output/'screen.png', dpi=160)
    plt.close(fig)
    coverage_figure(plt, full_groups, max_budget, output/'long-coverage.png')
    coverage_figure(plt, groups, common_budget, output/'matched-coverage.png')
    text = ['# Probabilistic maze sweep — 2026-11-04', '', '## Protocol', '',
            'All runs use normalized 2D coordinates in the deterministic 12×12 maze (layout seed 7; 1,784 reachable positions), 1,000 warmup actions, and center resets every 2,000 steps. Warmup uses persistent random except in the independent-action random control, which uses uniform actions throughout. No actor, codebook, state merging, reconstruction loss, or oracle guidance enters learning/planning. Models jointly train the encoder and Gaussian-mixture dynamics ensemble with NLL and pairwise repulsion.', '',
            'The initial screen tests 15 learned configurations plus persistent random for 10,000 interactions on seed 0. Except where overridden, the model has 8 latents, two width-64 hidden layers, three ensemble members and three Gaussian components; Adam 3e-4, variance floor 1e-4, repulsion weight 10, and five updates every ten interactions. MCTS uses horizon 4, 24 simulations, discount .9, and epsilon .1. The horizon-8 variant also uses 48 simulations.', '',
            'An adaptive follow-up adds latent dimension 4 and three variants with 64 Monte Carlo samples per member (default, latent-2, and horizon-1), motivated by an independent reward-noise probe. All four use the same seed-0 10k budget. This is a mostly one-factor screen, not an exhaustive search over combinations.', '',
            'Fresh-seed confirmation uses model/behavior seeds 3, 4, and 5. Single-Gaussian and width-128 runs started while screening finished, based on their provisional ranking; their outcomes did not affect screening selection. Both run for 100,000 steps. Horizon-8 later overtook them in the screen. Because it expands about 300 nodes/action versus 60 for the default, its confirmation budget was adjusted to 20,000 steps. This explicitly changes the original plan to give both eventual finalists 100k. Width-128 is retained as an additional architecture control. All policies are compared fairly using their first 20k transitions; 100k results are presented separately. The maze layout is unchanged, so this tests seed robustness, not transfer to unseen mazes.', '',
            'A separate adaptive activation diagnostic fits tanh and identity encoders from scratch for 50,000 updates each on the same stored data and whole-episode split. It is followed by a matched 20k online comparison of the default bounded and unbounded encoders on seeds 3–5. These six online follow-ups are excluded from the original screen ranking. They reuse seeds 3–5 after examining the seed-5 replay, so this activation comparison is exploratory rather than an untouched validation set. They appear in the common-budget table, and the fixed-data findings are reported separately.', '',
            'Coverage AUC is the exact mean fraction of reachable positions visited after every real interaction, calculated from raw replay. Final coverage and time to coverage thresholds are reported separately. Oracle geometry is confined to evaluation and hyperparameter selection. Baselines are matched by interactions and update schedule, not wall-clock compute. Collection times reflect varying concurrency and are not a controlled speed benchmark. No online checkpoint/model-selection criterion uses coverage.', '',
            '## Screening results', '',
            '| Setting | Position AUC | Final positions | Final rooms | Updates | Collection seconds |',
            '|---|---:|---:|---:|---:|---:|']
    for r in screen:
        text.append(f"| {r['variant']} | {100*r['coverage_lattice_stats']['auc']:.1f}% | {100*r['coverage_lattice_stats']['final']:.1f}% | {100*r['coverage_room_stats']['final']:.1f}% | {r['metrics']['updates']:,} | {r['collection_seconds']:.0f} |")
    text += ['', '![Screen](screen.png)', '', f'## Matched {common_budget:,}-step confirmation', '']
    text += comparison_table(groups, 'common_')
    text += ['', '![Matched coverage](matched-coverage.png)', '', f'## {max_budget:,}-step architecture comparison', '']
    text += comparison_table(full_groups)
    text += ['', '![Long coverage](long-coverage.png)', '', '### Individual final evaluations (budgets differ)', '',
             '| Setting | Seed | Steps | Position AUC | Final positions | Steps to 80% positions | Replay JS | Seen / unseen JS | No-motion actions |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in long:
        m = r['metrics']; unseen = m['oracle_unseen_disagreement']
        js = f"{m['oracle_seen_disagreement']:.3f} / {unseen:.3f}" if unseen is not None else f"{m['oracle_seen_disagreement']:.3f} / —"
        text.append(f"| {r['variant']} | {r['seed']} | {m['environment_steps']:,} | {100*r['coverage_lattice_stats']['auc']:.1f}% | {100*r['coverage_lattice_stats']['final']:.1f}% | {r['coverage_lattice_stats']['to_80'] or 'not reached'} | {m['replay_disagreement']:.3f} | {js} | {100*r['behavior']['blocked_or_stay_fraction']:.1f}% |")
    if bundle.get('activation_fixed_data'):
        paired = bundle['activation_fixed_data']
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        labels, keys = ['All pairs', 'Observed pairs', 'Unobserved pairs'], ['all', 'seen', 'unseen']
        x = np.arange(3)
        for j, row in enumerate(paired):
            label = {'tanh': 'Bounded (tanh)', 'identity': 'Unbounded (identity)'}[row['activation']]
            accuracy = [100*row['retrieval'][k]['exact_next_position_accuracy'] for k in keys]
            error = [row['retrieval'][k]['mean_physical_error'] for k in keys]
            for ax, values in zip(axes, [accuracy, error]):
                ax.bar(x+(j-.5)*.34, values, .34, label=label)
        axes[0].set(ylabel='Exact next-position retrieval (%)', ylim=(0, 100), title='Prediction accuracy')
        axes[1].set(ylabel='Mean Euclidean error (maze units)', title='Physical prediction error')
        for ax in axes:
            ax.set_xticks(x, labels); ax.grid(axis='y', alpha=.2); ax.legend()
        fig.suptitle('Fixed-data encoder ablation: identical replay, 50k updates, seed 5')
        fig.tight_layout(); fig.savefig(output/'activation-comparison.png', dpi=160); plt.close(fig)
        text += ['', '## Fixed-data activation diagnostic', '',
                 'Two fresh fits use the same 100k-transition uniform-random seed-5 replay and the same whole-episode 80/20 split: 80k training transitions, 20k held out, 50k optimizer updates each. Only the encoder output activation changes. No new interactions are collected. The evaluator retrieves the nearest encoded maze position to the ensemble-mean prediction; the full catalogue is never available to the agent. This is a specific readout, not an optimal decoder. Observed/unobserved refers to physical state-action pairs in the fitting buffer.', '',
                 '| Activation | All-pair retrieval | Unobserved-pair retrieval | Mean physical error | Horizontal / vertical latent step |',
                 '|---|---:|---:|---:|---:|']
        for row in paired:
            r = row['retrieval']
            text.append(f"| {row['activation']} | {100*r['all']['exact_next_position_accuracy']:.1f}% | {100*r['unseen']['exact_next_position_accuracy']:.1f}% | {r['all']['mean_physical_error']:.4f} | {r['horizontal_vertical_step_ratio']:.2f} |")
        text += ['', '![Activation comparison](activation-comparison.png)', '',
                 'Both offline fits preserve sensitivity to horizontal and vertical movement. The original online uniform-random seed-5 model was nearly insensitive to vertical movement (horizontal/vertical step ratio about 758), so the output bound alone does not explain that earlier failure. Offline fitting sees the full buffer from the beginning; online fitting has a changing data distribution. The online activation comparison tests whether the fixed-data prediction benefit carries over to exploration.', '']
    text += ['', '## Interpretation limits', '',
             'These are coordinate-observation results on a single deterministic maze layout. They do not test pixel representation learning, nuisance invariance, stochastic navigation, or transfer. Ensemble disagreement is a heuristic, not calibrated epistemic uncertainty. All-state latent errors use each model’s learned coordinate system and should not be treated as directly comparable physical errors. Final probes include seen transitions. NLLs across separately learned encoders are not model-selection scores.', '',
             'Each run retains its resolved config, exact source snapshot/hashes, training and exploration logs, replay, checkpoint, and final evaluation. The machine-readable `results.json` includes per-run paths and full diagnostics. The manifests record the grid and seeds; the runner isolates RNGs in worker processes and refuses to overwrite incomplete runs.', '']
    (output/'report.md').write_text('\n'.join(text))


if __name__ == '__main__':
    main()
