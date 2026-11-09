"""Animate saved maze replays, with accumulated coverage and reset-safe trails."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

from map_learning.config import load_config
from map_learning.maze import Maze


def load_run(folder, label):
    cfg = load_config(folder/'config.json')
    with np.load(folder/'replay.npz', allow_pickle=False) as replay:
        data = {key: replay[key].copy() for key in ('positions', 'next_positions', 'episodes')}
    if not len(data['positions']):
        raise ValueError(f'Empty replay: {folder}')
    points = np.stack([data['positions'], data['next_positions']], axis=1).reshape(-1, 2)
    data['unique'], data['first'] = np.unique(points, axis=0, return_index=True)
    starts = np.r_[0, np.flatnonzero(np.diff(data['episodes']))+1]
    data['episode_start'] = np.maximum.accumulate(np.where(np.isin(np.arange(len(data['episodes'])), starts), np.arange(len(data['episodes'])), 0))
    data['reset_positions'] = np.unique(data['positions'][starts], axis=0)
    return label, cfg, data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', type=Path, help='Run folders containing config.json and replay.npz')
    parser.add_argument('--labels', nargs='+', help='One display label per run')
    parser.add_argument('--out', required=True, type=Path, help='Output GIF path')
    parser.add_argument('--start', type=int, default=0, help='First transition index (zero-based)')
    parser.add_argument('--end', type=int, help='Exclusive transition index; default: shortest replay length')
    parser.add_argument('--stride', type=int, default=500, help='Transitions advanced per frame')
    parser.add_argument('--trail', type=int, default=200, help='Maximum recent path length')
    parser.add_argument('--fps', type=int, default=12)
    args = parser.parse_args()
    if args.labels and len(args.labels) != len(args.runs):
        parser.error('--labels must match the number of runs')
    if min(args.stride, args.trail, args.fps) < 1:
        parser.error('stride, trail and fps must be positive')
    if args.out.suffix.lower() != '.gif':
        parser.error('--out must end in .gif')
    if args.out.exists():
        parser.error(f'Refusing to overwrite {args.out}')
    runs = [load_run(p, label) for p, label in zip(args.runs, args.labels or [p.name for p in args.runs])]
    limit = min(len(d['positions']) for _, _, d in runs)
    end = limit if args.end is None else args.end
    if not 0 <= args.start < end <= limit:
        parser.error(f'Require 0 <= start < end <= {limit}')
    frames = list(range(args.start+args.stride, end+1, args.stride))
    if not frames or frames[-1] != end:
        frames.append(end)
    fig, axes = plt.subplots(1, len(runs), figsize=(max(7, 4*len(runs)), 4.4), squeeze=False)
    artists = []
    for ax, (label, cfg, data) in zip(axes[0], runs):
        Maze(cfg.maze).draw(ax)
        ax.set_title(f'{label}\nSeed {cfg.seed}')
        ax.scatter(*data['reset_positions'].T, marker='*', s=90, color='black', zorder=5)
        visited = ax.scatter([], [], s=4, color='#abcbd8', alpha=.65)
        trail, = ax.plot([], [], color='#ee672f', lw=1.2)
        agent = ax.scatter([], [], s=35, color='#c92127', zorder=6)
        artists.append((visited, trail, agent))
    title = fig.suptitle('')
    fig.tight_layout(rect=[0, .13, 1, .88])
    fig.text(.5, .02, 'Blue: accumulated visits • Orange: recent path • Red: agent • Stars: episode starts', ha='center', fontsize=8)

    def update(t):
        for (_, _, data), (visited, trail, agent) in zip(runs, artists):
            visited.set_offsets(data['unique'][data['first'] < 2*t])
            start = max(int(data['episode_start'][t-1]), t-args.trail)
            path = np.concatenate([data['positions'][start:t], data['next_positions'][t-1:t]])
            trail.set_data(path[:, 0], path[:, 1])
            agent.set_offsets(data['next_positions'][t-1:t])
        title.set_text(f'Environment step {t:,} • trails restart at episode boundaries')

    animation = FuncAnimation(fig, update, frames=frames, interval=1000/args.fps)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    animation.save(args.out, writer=PillowWriter(fps=args.fps))
    plt.close(fig)
    print(args.out)


if __name__ == '__main__':
    main()
