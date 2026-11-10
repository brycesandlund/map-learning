from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from map_learning.maze import Maze,reachable_graph
from map_learning.config import load_config
from analyze_reward_stability import rank


def main():
 rows=json.loads(Path('docs/assets/latent-geometry-results.json').read_text());dyn=json.loads(Path('docs/assets/geometry-dynamics-results.json').read_text())
 mean=lambda rs,k:np.mean([r[k] for r in rs])
 points,adj,index=reachable_graph(Maze(load_config('configs/horizon_base.json').maze));spacing_rows=[]
 for file in sorted(Path('runs/latent-geometry-20261110').glob('*step30000.npz')):
  data=dict(np.load(file));z=data['latent'];move=adj!=np.arange(len(points))[:,None]
  d=np.linalg.norm(z[adj]-z[:,None],axis=-1);d[~move]=np.nan;local=np.nanmedian(d,axis=1);q=np.quantile(local,[.05,.5,.95])
  source=Path('runs/horizon-duration-20261110')/file.stem.replace('-step30000','')
  replay=dict(np.load(source/'replay.npz'));counts=np.zeros(len(points))
  for p in np.concatenate([replay['positions'],replay['next_positions']]):counts[index[tuple(p)]]+=1
  spacing_rows.append({'run':file.stem,'local_spacing_q05_q50_q95':q.tolist(),'q95_over_q05':float(q[2]/q[0]),'spacing_visitation_spearman':float(np.corrcoef(rank(local),rank(counts))[0,1])})
 Path('docs/assets/latent-local-spacing.json').write_text(json.dumps(spacing_rows,indent=2)+'\n')
 lines=['# Latent geometry and action structure','',
 'Audited H8 independent and H16 shared, seeds 0–2, at initialization and 6k/18k/30k interactions. Observations are two-dimensional coordinates. All maze geometry, shortest paths, linear probes, and labels are post-hoc evaluation only.','',
 '**The learned latent is a distorted coordinate surface, with small but consequential extra directions.** At 30k, the first two principal components explain 98.0–100.0% of variance across six models. This does not mean two coordinates suffice for the existing dynamics: removing lower-variance directions can sharply reduce successor accuracy. Three components restore nearly all accuracy in five models; the sixth needs four. This is an input ablation without retraining, not a minimal-dimensionality theorem.','',
 '## What the pictures show','',
 '![Geometry](assets/latent-geometry.png)','',
 'Colors encode physical x/y consistently across panels. Gray edges are actual simulator adjacency, overlaid for evaluation; they are not an extracted learned graph. Rings mark states unvisited at that checkpoint. PCA axes are fitted separately at each checkpoint, so rotations/reflections are not evidence of meaningful representation drift.','',
 'A maze-like layout is already visible at random initialization: a smooth MLP applied to coordinates preserves substantial spatial structure, and wall-shaped gaps exist in the evaluated free-space coordinates themselves. The picture alone is not evidence that the encoder learned walls. Initialization uses the original seed and architecture; its diagnostic coordinate probe uses the corresponding final visited set.','',
 '## Euclidean latent distance does not encode maze distance','',
 '| Model (mean of three seeds) | PC1+PC2 variance | Latent–physical distance Spearman | Latent–shortest-path Spearman | Across-wall/open distance ratio |',
 '|---|---:|---:|---:|---:|']
 for group in ['h16_shared_30k','h8_independent_30k']:
  for cutoff in [0,6000,18000,30000]:
   rs=[r for r in rows if r['group']==group and r['cutoff']==cutoff]
   lines.append(f'| {group}, {cutoff:,} | {mean(rs,"pca2_variance"):.2%} | {mean(rs,"latent_physical_spearman"):.3f} | {mean(rs,"latent_geodesic_spearman"):.3f} | {mean(rs,"barrier_vs_open_median_distance_ratio"):.3f} |')
 lines+=['','Pair correlations use all distinct pairs in the 1,784-state maze, with each model seed aggregated equally. Physical Euclidean distance itself has Spearman 0.718 with shortest-path distance. Learned latent distances do not improve that relationship.','',
 'The last column compares median latent distances for pairs physically 0.5 units apart: open pairs require two moves; barrier pairs require at least eight. Ratios near one show no consistent wall-dependent expansion. These pairs are matched on physical separation, not on location or local encoder stretching; this is not proof that every wall is ignored.','',
 'A diagnostic linear coordinate decoder fitted on 80% of visited states recovers coordinates less accurately after learning than at initialization. The embedding becomes nonlinear rather than simply a rotated/scaled copy of x/y. Full probe errors and nearest-neighbor graph distances are retained in JSON. Neither a failed linear decoder nor increased distortion implies lost state information.','',
 'Because the encoder maps two coordinate inputs into eight outputs, its smooth image can be a curved two-dimensional surface spanning three or more principal components. Global PCA dimension is different from intrinsic dimension.','',
 'Local scale is also nonuniform: across the six final models, the 95th-percentile local moving-edge length is 3.5–8.9 times the 5th percentile. A single latent-distance threshold therefore represents different physical tolerances in different regions. Local spacing is positively correlated with visitation in all six models (Spearman 0.31–0.64). Per-model local spacing and its correlation with visitation are retained in `latent-local-spacing.json`; these associations do not isolate the cause of the distortion.','',
 '## Small directions carry predictive information','',
 '| Model / seed | Full 8D input | Keep PC1–2 | Keep PC1–3 | Keep PC1–4 |',
 '|---|---:|---:|---:|---:|']
 for r in dyn:
  vals=[r['pca_input_ablation'][str(k)]['all_accuracy'] for k in [8,2,3,4]]
  lines.append(f'| {r["group"]} / {r["seed"]} | '+' | '.join(f'{v:.1%}' for v in vals)+' |')
 lines+=['','Inputs are projected onto the top components and reconstructed back into eight dimensions before entering the unchanged dynamics. Targets and nearest-state retrieval catalogue retain all eight dimensions. Accuracy is over **all maze states and four directional actions**, including unexplored regions; it is not the earlier seen-transition ~99% metric. Off-manifold inputs and lack of retraining are limitations of this ablation.','',
 '![Three-dimensional view and ablation](assets/latent-pca-ablation.png)','',
 '## Dynamics encode action-dependent obstacles','',
 'For the six final models, a blocked directional action predicts median displacement only 0.020–0.034 times the median actual moving-edge length; an allowed move predicts 0.96–0.99 times that scale. Predicted displacement separates blocked from allowed actions with AUROC 0.92–0.96 across the whole maze. That supports the interpretation that obstacle structure lives substantially in the action-conditioned dynamics, even though Euclidean latent distances do not encode detours.','',
 '![Action vectors](assets/latent-action-vectors.png)','',
 'This figure shows every sixth state for one action (right), projected to PCA2. Red sources are physically blocked, blue sources can move. Green arrows are true latent displacements; orange arrows are ensemble-mean predictions. The important third component is hidden here.','',
 '## Can latent distances identify loops?','',
 'For each frozen final model, 256 persistent-random trajectories of length 32 start at visited states. Recursively predicted states are compared to **earlier imagined states on the same path**. The score is negative minimum latent distance; labels use whether the true simulator path revisits any earlier position. Stationary steps are excluded, so this tests moving returns rather than just stay actions or wall collisions.','',
 '| Model | Revisit AUROC through H16 | Through H32 | H16 recall at 5% false-positive rate |',
 '|---|---:|---:|---:|']
 for group in ['h16_shared_30k','h8_independent_30k']:
  rs=[r for r in dyn if r['group']==group]
  vals=[np.mean([r['loop'][str(h)]['auc'] for r in rs]) for h in [16,32]]
  recall=np.mean([r['loop']['16']['true_positive_at_threshold'] for r in rs])
  lines.append(f'| {group} | {vals[0]:.3f} | {vals[1]:.3f} | {recall:.1%} |')
 lines+=['','This makes a path-local loop penalty plausible. It does not yet validate a threshold or a benefit under MCTS-selected trajectories. The recall threshold was chosen post hoc from the same diagnostic samples, separately per model; its normalization uses an oracle median edge length. No such threshold, oracle graph, or loop penalty was fed to the live experiments. Detection is less reliable at longer horizons.','',
 'Because observations are coordinates, an encoder is already a function of two inputs and exact revisits have identical encodings. These findings do not establish state equivalence under pixel noise, partial observability, or continuous non-repeating observations.','',
 '## Reproduction','',
 'Run `scripts/analyze_latent_geometry.py`, `scripts/audit_geometry_dynamics.py`, then `scripts/report_latent_geometry.py` with the repository virtualenv. The first refuses an existing `runs/latent-geometry-20261110`; preserve or choose a fresh output directory to recompute. Portable metrics: `docs/assets/latent-geometry-results.json` and `docs/assets/geometry-dynamics-results.json`. Per-state latent arrays and PCA projections: `runs/latent-geometry-20261110/*.npz`.','']
 Path('docs/latent-geometry.md').write_text('\n'.join(lines))
 with np.load('runs/latent-geometry-20261110/h16_shared_30k-seed0-step30000.npz') as data:z=data['latent'];points=data['points']
 centered=z-z.mean(0);_,s,v=np.linalg.svd(centered,full_matrices=False);pc=centered@v[:3].T
 color=np.column_stack([points[:,0]/12,points[:,1]/12,np.full(len(points),.55)])
 fig=plt.figure(figsize=(13,5));ax=fig.add_subplot(121,projection='3d');ax.scatter(*pc.T,c=color,s=5);ax.set_box_aspect(np.ptp(pc,axis=0));ax.set(xlabel='PC1',ylabel='PC2',zlabel='PC3',title='H16 shared seed 0: curved surface in 3D')
 ax=fig.add_subplot(122)
 for r in dyn:ax.plot([1,2,3,4,8],[r['pca_input_ablation'][str(k)]['all_accuracy'] for k in [1,2,3,4,8]],'o-',label=f'{"H16 shared" if r["group"].startswith("h16") else "H8 independent"} / seed {r["seed"]}')
 ax.set(xlabel='Number of retained principal components',ylabel='Whole-maze successor retrieval accuracy',ylim=(0,1),xticks=[1,2,3,4,8],title='Frozen dynamics after projecting its latent inputs');ax.legend(fontsize=8);ax.grid(alpha=.2)
 fig.tight_layout();fig.savefig('docs/assets/latent-pca-ablation.png',dpi=160);plt.close(fig)
if __name__=='__main__':main()
