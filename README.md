# Map learning

Learn a navigation representation from observations and actions, then search through
learned latent dynamics. No actor, critic, policy gradient, goal-conditioned policy,
observation reconstruction, or pretrained model is used.

This is a small experimental implementation of **QRL** and **Time-Myopic** distance
objectives, not a reproduction of either paper's complete agent or benchmark results.
Compression is an experimental addition, not a proven state-equivalence algorithm.

## Setup

```bash
cd /Users/brycesandlund/GitHub/map-learning
UV_CACHE_DIR=.uv-cache uv sync --extra dev
source .venv/bin/activate
pytest -q
```

Python 3.11+; CPU is the default and is sufficient for these mazes. PyTorch, NumPy,
and Matplotlib are the only runtime dependencies. `uv.lock` pins the resolved environment.
Alternatively: `python -m pip install -e '.[dev]'` in a virtual environment.

## First experiments

Collect one dataset, then fit both objectives to the **same data and episode split**:

```bash
map-learning collect --out runs/maze-data
map-learning train --config configs/qrl.json \
  --data runs/maze-data/replay.npz --out runs/qrl
map-learning train --config configs/time_myopic.json \
  --data runs/maze-data/replay.npz --out runs/time-myopic
map-learning train --config configs/time_myopic_min.json \
  --data runs/maze-data/replay.npz --out runs/time-myopic-min
```

The default is 20,000 interactions in a seeded 12×12 continuous maze, followed by
3,000 training updates. For a quick end-to-end run, pass `--config configs/smoke.json`
to both collection and training (a 4×4 maze, 1,200 interactions, 100 updates).
Output directories must be new; existing runs are never overwritten.

Try compression, changing the number of available prototypes and the usage penalty:

```bash
map-learning train --config configs/qrl_compressed.json \
  --data runs/maze-data/replay.npz --out runs/qrl-k64 \
  --set model.codebook_size=64 --set train.compression_weight=0.01

map-learning sweep --data runs/maze-data/replay.npz --out runs/comparison \
  --seeds 0 1 2 --codebook-sizes 0 32 128 --set train.updates=3000
```

`sweep` runs both objectives for each seed/codebook size and writes `summary.csv`.
For each seed, the held-out episodes are the same across methods. Set
`train.exact_revisit_min=true` to include the Time-Myopic duplicate correction in a
sweep (QRL ignores it). For an ablation at fixed alphabet size, vary
`train.compression_weight` including zero. Codebook size zero disables the entire
bottleneck and its regularizers.

## Act using the learned map

Frozen-map exploration after offline fitting:

```bash
map-learning rollout --checkpoint runs/qrl/checkpoint.pt \
  --out runs/qrl-distance-rollout --steps 2000 \
  --set planner.behavior=distance
map-learning rollout --checkpoint runs/qrl/checkpoint.pt \
  --out runs/qrl-disagreement-rollout --steps 2000 \
  --set planner.behavior=disagreement
map-learning rollout --checkpoint runs/qrl/checkpoint.pt \
  --out runs/qrl-goal-rollout --steps 400 --goal 2.5 3.5
```

An explicit goal is converted to an observation by the evaluation environment; the
planner receives that observation, not a route or the maze adjacency. A goal should
be a reachable free-space point. Success means coming within half an action step.
`episodes.json` records completed episodes; the final partial episode is not counted.

Online interaction and learning, with short updates interleaved continuously:

```bash
map-learning online --config configs/qrl.json --out runs/online-qrl \
  --set planner.behavior=hybrid --set collect_steps=20000
map-learning online --config configs/time_myopic.json --out runs/online-myopic \
  --set planner.behavior=disagreement --set collect_steps=20000
```

Choices are `random`, `distance`, `disagreement`, and `hybrid`. After random warmup,
beam search expands the five discrete controls through an ensemble of latent
transition models, scores branches, executes the first action, then replans from the
next real observation. Distance is measured from the episode's starting observation;
set `planner.distance_reference=current` for distance from the current observation.
An explicit goal instead scores negative predicted distance to the goal.

Disagreement is discounted variance across independent bootstrapped dynamics heads.
In online runs it is normalized by replay latent standard deviations. Frozen rollouts
use raw variance; weight settings are therefore not directly comparable between the
two modes. Heads share one encoder: this measures dynamics disagreement, not full
representation uncertainty. Deeper model rollouts can exploit model errors; vary
`planner.horizon` and inspect actual coverage. This first implementation uses beam
search, not MCTS or A*.

## Objectives

Let `z = f(o)` and `d(z, z')` be the learned distance.

**QRL:** an independently implemented interval quasimetric embedding (IQE-maxmean)
head enforces nonnegativity, zero self-distance and triangle inequality. It need not
be symmetric. For each latent component it computes the length of the union of
directed intervals `[z_i, z'_i]`, ignoring reversed intervals. Components are combined
by a learned convex mixture of their maximum and mean.

The primal objective is

```
push(d(random_state, random_state))
  + lambda * (mean(relu(d(o_t, o_next) - 1)^2) - epsilon^2)
```

`push(d) = scale * softplus(-d / scale)` supplies a smaller gradient for already-large
distances. Set `train.push_scale=0` for the linear `-mean(d)` objective. The separate
dual update performs projected ascent:

```
lambda <- max(0, lambda + dual_lr * (mean_violation - epsilon^2))
```

This is a stochastic optimization of an **average** constraint. Finite training and
positive epsilon do not guarantee each edge is bounded by 1. Monitor violation and
the multiplier. Collision/no-op transitions still have unit action cost but identical
observations have distance zero. No oracle distances enter training.

**Time-Myopic:** an MLP on `z_next - z` predicts
`psi = 1 - exp(-relu(MLP(z_next - z)))`, regressed with MSE onto
`min(observed_elapsed_steps / horizon, 1)`. The head bias starts positive to avoid a
dead ReLU at initialization. This head does **not** enforce triangle inequality or
zero self-distance. The public distance method multiplies `psi` by the horizon so
its scale is in steps; it saturates for distant states.

Pairs never cross episode resets. Half are sampled within the horizon, half from the
full configurable window; zero-lag and distant examples are included. Plain observed
time supervision is policy-dependent and has contradictory labels on revisits.
`train.exact_revisit_min=true` substitutes the minimum elapsed separation found for
each **exact observation pair**, within `train.max_pair_steps`, across the training
replay. It also sets exact self-pairs to zero. This implements the paper's duplicate
handling idea using raw bytes instead of MD5. It does not identify semantically
equivalent images or use coordinate labels. The index costs O(N × max_pair_steps),
is cached offline and rebuilt when online replay changes; leave it off for large
online experiments until incremental indexing is implemented.

**Dynamics:** each independently initialized head learns `g_i(z_t, a_t) -> z_next`
using latent MSE on its own bootstrap batch. Both encoder input and target are
detached for this loss. Only the distance and optional bottleneck losses train `f`,
so latent dynamics cannot collapse `f` by moving its own prediction targets. Raw
observations are re-encoded after every map update; no stale latent replay is used.

## Compression and graph reuse

An optional learned codebook with K prototypes maps each observation to one of at
most K latent vectors. The forward pass uses a hard nearest-prototype assignment;
encoder gradients use a straight-through estimator. The regularizer is

```
mean(||code - stopgrad(z)||^2)
  + commitment_weight * mean(||z - stopgrad(code)||^2)
  + compression_weight * H(mean_batch soft_assignments)
```

Minimizing marginal assignment entropy encourages reuse of fewer codes. It is an
MDL-inspired **surrogate**, not exact minimization of the number of used states.
Soft entropy can disagree with hard usage; we report hard used-code count and
perplexity as well. Distance preservation competes with compression, and too much
compression can produce a constant map. The codebook is initialized from replay
encoder outputs by farthest-point sampling. In planning, predicted states are
projected back onto prototypes after measuring disagreement.

A stateless encoder already maps repeated *identical observations* to the same
latent, irrespective of trajectory history. The difficult experiment is different
observations of equivalent states. Try changing appearance with
`maze.observation=features` and `maze.nuisance_dims=8`; these append independently
sampled nuisance values. The evaluation checks same-position latent disagreement,
without using position equivalences during training. A compact codebook alone does
not prove correct equivalence or preserve all goals. This is a rate/distortion
tradeoff to measure, not a complete solution to state abstraction.

## Environment and observations

The maze has continuous coordinates, axis-aligned walls, swept-segment collision
tests, and center resets. A seeded spanning tree connects rooms; extra doors create
loops. It is **inspired by** the Cell-Free LGE figure, not the authors' exact layout.
The environment accepts continuous 2D controls, while the initial data collector,
learned dynamics, and beam planner use five controls: right, left, up, down, stay.
This makes exhaustive branching and exact evaluation practical. It is not yet a
continuous-action planner.

Observation options:

* `coordinates`: normalized `(x, y)`, matching the simple starting point in LGE.
* `features`: a fixed random Fourier transform of position, optionally with nuisance
  dimensions, without exposing raw coordinates to the model.
* `pixels`: two 48×48 channels, walls and agent marker; a CNN encoder. No reconstruction
  loss. Low image resolution can alias nearby physical positions.

Example:

```bash
map-learning collect --out runs/features-data \
  --set maze.observation=features --set maze.nuisance_dims=8
map-learning collect --config configs/smoke.json --out runs/pixels-data \
  --set maze.observation=pixels
```

The dataset's environment configuration overrides the environment fields during
offline training so it cannot be accidentally reinterpreted with another geometry.

## Outputs and evaluation

Every training run saves resolved `config.json`, dependency/Git metadata and a source
snapshot, `training.jsonl`, `checkpoint.pt`, `metrics.json`, `diagnostics.png`,
`latent.png`, and `graph.json`. Collection/online runs save the complete `replay.npz`.
Replay is append-only in memory and compressed on disk; no samples are evicted.
Large pixel datasets will eventually require a disk-backed replay implementation.

Offline evaluation reserves 20% of complete episodes. Oracle evaluation separately
enumerates all reachable positions under the discrete action set and uses BFS for
exact shortest action counts, including wall rejection. **This graph is never passed
to training or planning.** Oracle plots include unvisited space; metrics separate
visited endpoint pairs from pairs with an unvisited endpoint. The shortest-path
target is capped for Time-Myopic; its MAE is therefore not directly comparable to
QRL's uncapped MAE. The `oracle_local_pair_mae` uses pairs within the common horizon.

Diagnostics include coverage, distance error/correlation, transition prediction
error, QRL constraint violation, triangle violations, self-distance, effective latent
rank, nuisance invariance, and codebook usage. With compression, `graph.json` contains
learned prototype nodes and counted action-labelled transitions. Without compression
it exports an exact-observation identity graph as a baseline, not an inferred quotient
graph. Multiple targets for a node/action are reported as an ambiguity diagnostic;
in a deterministic maze this can reveal excessive merging.

Resume offline optimization (the update count is **additional** updates):

```bash
map-learning train --config configs/qrl.json --data runs/maze-data/replay.npz \
  --resume runs/qrl/checkpoint.pt --out runs/qrl-more --set train.updates=1000
```

Only update count and log frequency may change when resuming. Online environment and
planner state are not checkpointed for exact continuation yet. For consistent
held-out metrics use the `train` command; standalone `evaluate` explicitly labels
its supplied data metrics as replay diagnostics.

## Sources

* [QRL, Wang et al., ICML 2023](https://arxiv.org/abs/2304.01203), Eq. 6 and 12.
* [IQE, Wang & Isola](https://arxiv.org/abs/2211.15120).
* [Time-Myopic Go-Explore](https://arxiv.org/abs/2301.05635), Eq. 2–4 and §5.2.
* [Cell-Free Latent Go-Explore](https://proceedings.mlr.press/v202/gallouedec23a.html),
  maze inspiration. No upstream implementation code is vendored.
