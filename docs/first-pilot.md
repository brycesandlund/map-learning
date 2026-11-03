# First infrastructure pilot

Date: 2026-11-02. One seed; these are untuned diagnostics, not evidence of exploration superiority.

All five models use the same 20,000-step persistent-random dataset in a 12×12 continuous maze, with 16,000 training transitions and 4,000 held-out transitions from disjoint episodes. Each receives 1,000 updates. The coordinate observation baseline uses 32-dimensional latents and five bootstrapped dynamics heads. The exact evaluation lattice has 1,784 reachable positions.

The training trajectories visit 64.6% of rooms and 52.5% of reachable lattice states. This coverage belongs to the fixed data collector, not the learned planners.

| Variant | Local distance MAE (steps) | Mean self-distance | Effective latent rank | Used prototypes |
|---|---:|---:|---:|---:|
| QRL | 6.35 | 0.00 | 2.00 | continuous |
| Time-Myopic, raw elapsed time | 5.72 | 6.69 | 1.86 | continuous |
| Time-Myopic, exact-revisit minimum | 3.63 | 0.00 | 2.02 | continuous |
| QRL, K=128 | 11.25 | 0.00 | 1.62 | 3 |
| Time-Myopic, K=128 | 4.69 | 12.98 | 0.00 | 1 |

Local MAE uses the same oracle pairs with shortest distances ≤20 for all variants. Global MAE is not comparable because Time-Myopic clips its target at 20. The collapsed model can score deceptively well on MAE by predicting an average; inspect rank, self-distance, correlation, and code usage together.

## Observations

* The losses, dataset sharing, checkpoint loading, and exact oracle evaluation run end to end.
* QRL has exactly zero self-distance by architecture; this short run has not learned the full maze topology. See the distance field and pair scatter.
* Plain Time-Myopic elapsed-time labels conflict on revisits. Minimum separation for exact observation pairs changes self-distance from 6.69 to 0.00 steps in this run. This uses observed data only and is not a semantic-equivalence solution.
* Compression overmerged. QRL retained only 3 of 128 codes; Time-Myopic retained 1. Their smaller graphs are not a success. Leave the bottleneck off for initial distance-loss baselines, then sweep codebook size, commitment and entropy weights.
* Online disagreement and hybrid modes, frozen goal rollouts, offline continuation, standalone evaluation, pixel encoding, and nuisance observations received smoke coverage. No learned-planner advantage is established by these checks.
* The automated suite passes 17 tests, including an exact small line learning check, IQE triangle/gradient tests, reset-safe temporal sampling, exact-revisit handling, checkpoint continuation, and exhaustive wall-contact exclusion.

## Reproduce

```bash
UV_CACHE_DIR=.uv-cache uv sync --extra dev
source .venv/bin/activate
map-learning collect --out runs/new-maze-data
map-learning sweep --data runs/new-maze-data/replay.npz --out runs/new-pilot --seeds 0 --codebook-sizes 0 128 --set train.updates=1000 --set train.log_every=250
map-learning train --config configs/time_myopic_min.json --data runs/new-maze-data/replay.npz --out runs/new-myopic-min --set train.updates=1000 --set train.log_every=250
```

The earlier pilot data under `runs/superseded-wall-endpoint-bug/` is invalidated and kept only for debugging. The current results were generated after fixing collinear endpoint collisions and rerunning the complete collection/training pipeline.
