"""Scientific views of shared-replay latent geometry; all distances audited in 8D."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/map-learning-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np

from map_learning.maze import Maze
from map_learning.config import load_config


KINDS = ['baseline', 'kl', 'l2']
LABEL = {'baseline': 'Baseline', 'kl': 'KL', 'l2': 'L2'}
COLORS = {'baseline': '#777777', 'kl': '#0072b2', 'l2': '#009e73'}


def spatial_colors(points):
    return np.column_stack([.12+.8*points[:, 0]/12, .12+.8*points[:, 1]/12, np.full(len(points), .50)])


def latent_panel(ax, data, row):
    seen = data['counts'] > 0
    colors = spatial_colors(data['points'])
    ax.add_collection3d(Line3DCollection(data['pc'][data['edges']], colors='#777777', alpha=.20, linewidths=.35))
    ax.scatter(*data['cloud_pc'].T, c=colors[data['cloud_ids']], s=.7, alpha=.12, depthshade=False)
    ax.scatter(*data['pc'][seen].T, c=colors[seen], s=5, alpha=.9, depthshade=False)
    for setter in [ax.set_xlim, ax.set_ylim, ax.set_zlim]:
        setter(-2.5, 2.5)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=28, azim=-65)
    ax.set_xticks([-2, 0, 2]); ax.set_yticks([-2, 0, 2]); ax.set_zticks([-2, 0, 2])
    ax.tick_params(labelsize=7, pad=0)
    ax.set_xlabel('PC1', fontsize=8, labelpad=-3)
    ax.set_ylabel('PC2', fontsize=8, labelpad=-3)
    ax.set_zlabel('PC3', fontsize=8, labelpad=-3)
    ax.set_title(f'{LABEL[row["kind"]]} · seed {row["seed"]}\nPC1: {100*row["pca_spectrum"][0]:.2f}% · RMS scale: {row["rms_centroid_scale"]:.3f}', fontsize=10)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    a = p.parse_args()
    rows = json.loads((a.root/'results.json').read_text())
    assert len(rows) == 18
    assets = Path('docs/assets')
    cfg = load_config('configs/noisy_frontier.json')
    lookup = {(r['kind'], r['seed'], r['prefix']): r for r in rows}
    data = {k: dict(np.load(a.root/f'{k[0]}-seed{k[1]}-at{k[2]}.npz')) for k in lookup}
    for prefix in [50000, 75000]:
        fig = plt.figure(figsize=(12, 11), layout='constrained')
        for i, seed in enumerate([9, 10, 11]):
            for j, kind in enumerate(KINDS):
                key = kind, seed, prefix
                latent_panel(fig.add_subplot(3, 3, i*3+j+1, projection='3d'), data[key], lookup[key])
        fig.suptitle(f'Same replay, three encoders · {prefix:,} actions\nPosition colors and true maze edges; each panel centered and divided by one RMS scale, never whitened', fontsize=12)
        fig.savefig(assets/f'l2-geometry-{prefix//1000}k.png', dpi=160)
        plt.close(fig)

    # First model/source pair chosen in advance, not the most dramatic seed.
    prefix, seed = 50000, 9
    fig = plt.figure(figsize=(16, 4.4), layout='constrained')
    ax = fig.add_subplot(1, 4, 1)
    d = data['baseline', seed, prefix]
    Maze(cfg.maze).draw(ax)
    seen = d['counts'] > 0
    ax.scatter(*d['points'][seen].T, c=spatial_colors(d['points'][seen]), s=6)
    ax.set_title(f'Physical positions observed\nSame {seen.sum()} positions in all panels', fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    for j, kind in enumerate(KINDS):
        latent_panel(fig.add_subplot(1, 4, j+2, projection='3d'), data[kind, seed, prefix], lookup[kind, seed, prefix])
    fig.suptitle('Shared noisy replay at 50k · PCA of position centroids; faint dots are actual noisy encodings\nPCA retains >99.9% of centroid spread; equal units on all three axes within each panel', fontsize=12)
    fig.savefig(assets/'l2-geometry-overview.png', dpi=160)
    plt.close(fig)

    fig = plt.figure(figsize=(12, 4.6), layout='constrained')
    axs = []
    for j, kind in enumerate(KINDS):
        ax = fig.add_subplot(1, 3, j+1, projection='3d')
        latent_panel(ax, data[kind, seed, prefix], lookup[kind, seed, prefix]); axs.append(ax)
    fig.suptitle('Same 50k replay · seed 9 · common physical-position colors\nOne scale per panel preserves shape; rotations and PCA axes are not shared coordinates', fontsize=12)
    def rotate(frame):
        for ax in axs: ax.view_init(elev=25, azim=-65+9*frame)
    animation = FuncAnimation(fig, rotate, frames=40, interval=150)
    animation.save(assets/'l2-geometry-rotation.gif', writer=PillowWriter(fps=7), dpi=100)
    plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(13, 4.2), layout='constrained')
    for i, kind in enumerate(KINDS):
        rr = [lookup[kind, s, 50000] for s in [9, 10, 11]]
        groups = [[100*r['pca_spectrum'][1] for r in rr],
                  [r['local_anisotropy'][1] for r in rr],
                  [r['local_weak_direction_noise_ratio'][1] for r in rr]]
        for ax, values in zip(axs, groups):
            ax.scatter(i+np.linspace(-.12, .12, 3), values, color=COLORS[kind], s=45)
            for x, v, s in zip(i+np.linspace(-.12, .12, 3), values, [9, 10, 11]):
                ax.annotate(str(s), (x, v), xytext=(4, 2), textcoords='offset points', fontsize=8)
            ax.set_xticks(range(3), [LABEL[k] for k in KINDS]); ax.set_yscale('log'); ax.grid(axis='y', alpha=.2)
    axs[0].set(title='Second spatial direction', ylabel='PC2 share of centroid variance (%)')
    axs[1].set(title='Local directional distortion', ylabel='Median strong / weak spatial stretch')
    axs[2].set(title='Noise along the weak direction', ylabel='Median projected noise RMS / weak stretch')
    axs[2].axhline(1, color='black', linestyle=':', alpha=.5)
    fig.suptitle('Full 8D geometry at 50k · dots identify the three paired model seeds\nLocal probes use two open orthogonal moves and ≥8 acquired observations at each location', fontsize=12)
    fig.savefig(assets/'l2-geometry-local.png', dpi=160); plt.close(fig)

    fig, axs = plt.subplots(2, 3, figsize=(13, 8), layout='constrained')
    for j, kind in enumerate(KINDS):
        d = data[kind, seed, prefix]
        repeat = d['expanded'][d['queue_ids']]
        productive = d['productive'][d['queue_ids']]
        scale = np.median(d['scores'][repeat])
        for mask, label, color in [(repeat, 'Already expanded', '#d55e00'), (productive, 'Exposes new neighbor', '#009e73')]:
            values = np.sort(d['scores'][mask]/scale)
            axs[0, j].plot(values, np.arange(1, len(values)+1)/len(values), label=label, color=color)
        cut = d['scores'][d['top100'][-1]]/scale
        axs[0, j].axvline(cut, color='#333333', linestyle=':', label='Top-100 cutoff')
        axs[0, j].set(xscale='log', xlim=(.05, 100), ylim=(0, 1), title=LABEL[kind],
                       xlabel='8D priority / median repeated-entry priority', ylabel='Fraction below score')
        axs[0, j].grid(alpha=.2)
        if j == 0: axs[0, j].legend(fontsize=8)
        Maze(cfg.maze).draw(axs[1, j]); pts = d['points']; top = d['top100']; ids = d['queue_ids'][top]
        axs[1, j].scatter(*pts[d['counts'] > 0].T, c='#dddddd', s=5)
        status = np.where(d['productive'][ids], '#009e73', np.where(d['expanded'][ids], '#d55e00', '#e69f00'))
        axs[1, j].scatter(*pts[ids].T, c=status, s=18, edgecolors='black', linewidths=.25)
        q = lookup[kind, seed, prefix]['queue']
        axs[1, j].set_title(f'Top 100: {q["top100_productive_fraction"]:.0%} productive; {q["top100_repeat_fraction"]:.0%} repeats', fontsize=10)
        axs[1, j].set_xticks([]); axs[1, j].set_yticks([])
    fig.suptitle('Same archive and frontier · seed 9 at 50k · scores computed in the original 8D latent space\nGreen: exposes new neighbor. Red: already expanded. Orange: other observed but unexpanded.', fontsize=12)
    fig.savefig(assets/'l2-geometry-queue.png', dpi=160); plt.close(fig)
    (assets/'l2-geometry-results.json').write_text(json.dumps(rows, indent=2)+'\n')


if __name__ == '__main__':
    main()
