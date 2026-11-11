"""Forecast accuracy with scale-normalized errors and paired seed trajectories."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('--results', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
rows = json.loads(a.results.read_text())['runs']
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
summary = []
colors = ['#777777', '#bd8dce', '#7041a1']
for col, obj in enumerate(['dynamics', 'adjacency']):
    groups = []
    for suffix, label, color in zip(
        ['base', 'multistep-control', 'multistep'],
        ['Baseline', 'Added H1', 'Added H32' if obj == 'dynamics' else 'Added H16'],
        colors,
    ):
        group = sorted([r for r in rows if r['name'] == obj + '-' + suffix], key=lambda r: r['seed'])
        groups.append(group)
        horizons = [r['horizon'] for r in group[0]['metrics']]
        values = np.array([[v['relative_to_latent_stay'] for v in r['metrics']] for r in group])
        axes[0, col].plot(horizons, values.mean(0), marker='o', color=color, label=label)
        for i, h in enumerate(horizons):
            metrics = [r['metrics'][i] for r in group]
            summary.append({'method': obj + '-' + suffix, 'horizon': h, 'n': len(group),
                **{k: float(np.mean([r[k] for r in metrics])) for k in metrics[0]},
                'physical_errors_by_seed': [r['predicted_xy_rmse'] for r in metrics]})
    for i, seed in enumerate([r['seed'] for r in groups[0]]):
        axes[1, col].plot(range(3), [g[i]['metrics'][-1]['predicted_xy_rmse'] for g in groups],
                          marker='o', alpha=.65, label=f'Seed {seed}')
    axes[0, col].set(title=obj.capitalize(), xlabel='Forecast horizon', ylabel='Latent MSE / stay MSE', xticks=horizons)
    axes[0, col].legend(fontsize=8)
    axes[1, col].set(xticks=range(3), xticklabels=['Baseline', 'Added H1', 'Added H32' if obj == 'dynamics' else 'Added H16'],
                     ylabel='H32 decoded position RMSE (maze units)')
    for ax in axes[:, col]:
        ax.grid(alpha=.2)
axes[1, 1].legend(fontsize=8)
fig.suptitle('Common recorded ancestry paths • Six fresh seeds\nH32 paths average 29.4 steps after clipping to available ancestry')
fig.tight_layout()
fig.savefig(str(a.out) + '.png', dpi=170)
plt.close(fig)
Path(str(a.out) + '.json').write_text(json.dumps(summary, indent=2) + '\n')
