# Map learning

Learn a navigation representation from observations and actions, then search through
learned latent dynamics. No actor, critic, policy gradient, goal-conditioned policy,
observation reconstruction, or pretrained model is used.

Includes **QRL** and **Time-Myopic** distance objectives, plus a **jointly trained
Gaussian-mixture dynamics ensemble with C-SWM-style repulsion**. These are small
experimental implementations, not reproductions of complete published agents.
The probabilistic mode uses continuous latents without a codebook, compression
penalty, state merging, or distance labels. Older codebook experiments remain available.

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

## Distance-centric noisy-frontier experiments

The persistent frontier can now train its deterministic encoder with adjacent
attraction, QRL-inspired upper-bound expansion, temporal stress, or Student-t
neighbor affinities. These use actual recorded transitions/ancestry and the same
Euclidean nearest-visited queue; auxiliary dynamics are detached from the encoder.
See the [results and animations](docs/distance-study.md) and
[loss definitions and controlled protocol](docs/distance-objectives.md).

```bash
python scripts/persistent_frontier.py --config configs/deterministic_frontier.json \
  --arm interleaved100 --seed 100 --distance-spec configs/distance_adjacency.json \
  --out runs/adjacency-frontier-new
```

This example uses adjacent attraction weight .3, repulsion .03 and latent L2 .03.
Other objective specs are `configs/distance_{affinity,qrl_upper,temporal,temporal_upper}.json`;
affinity and QRL specs require `--repulsion-weight 0` for the documented settings.
The other example specs retain repulsion .03; all these example specs use L2 .03.
Temporal labels follow actual parent links across simulator restores, never queue-pop
order. No physical-state merging, shortest-path labels, or noise resampling enters training.

## Joint probabilistic dynamics and exploration

```bash
map-learning online --config configs/probabilistic_maze.json --out runs/gmm-mcts
map-learning online --config configs/probabilistic_maze.json \
  --set planner.search=shooting --out runs/gmm-shooting
map-learning online --config configs/probabilistic_maze.json \
  --set planner.behavior=random --out runs/gmm-random

python scripts/run_probabilistic_pilot.py --out runs/gmm-comparison --seeds 0 1 2
```

The configuration uses a 6×6 deterministic maze with normalized coordinate inputs,
3,000 interactions, 300 random warmup steps, and two gradient updates every five
interactions starting at step 300 (1,082 updates total). Each method trains on its
own collected data. The same seed has identical warmup transitions across methods.
Random exploration repeats the previous action with probability 0.8; it is a strong
baseline for long corridors. Maze layout seed and model/behavior seed are separate.
Use `--set maze.seed=...` to test another layout. No parameters are pretrained.

The shared encoder produces `z = latent_bound * tanh(encoder(observation))`.
Each ensemble member outputs K mixture logits, K residual means, and K diagonal
variance vectors. Means are `z + delta`; variances are
`variance_floor + softplus(raw_variance)`. The default is M=3, K=3, D=8, hidden=64.
The bounded encoder fixes latent units; it is not a quantizer. Gaussian predictions
remain unbounded: sampled states are **not** clipped or projected back onto the
encoder manifold. This mismatch is explicitly measured in evaluation.

For each head independently resample a replay minibatch, then minimize

```
L = mean_heads[-mean_batch log p_m(f(next_observation) | f(observation), action) / D]
    + repulsion_weight * mean_pairs relu(repulsion_margin**2 - ||f(o_i)-f(o_j)||²)
```

Both source and target encoder branches receive gradients, in the **same backward
pass** as dynamics. The repulsion pairs are independent raw replay observations;
there is no oracle filtering, identity lookup, or shortest-path supervision. Random
pairs can include revisits, and the loss does not guarantee nuisance invariance.
The repulsion weight is 10, Euclidean margin 1, variance floor 1e-4, initial variance
0.01, and Adam learning rate 3e-4. NLL is per dimension and may be negative because
it is a density. Changing encoder geometry changes NLL; do not rank independently
learned representations by NLL alone. This mode has no learned navigation distance.

Exploration reward is the generalized Jensen–Shannon divergence
`U(z,a) = mean_m E_{x~p_m}[log p_m(x|z,a) - log(mean_j p_j(x|z,a))]`.
It compares complete mixture distributions, including weights and variances,
without matching component IDs. Identical broad distributions have zero reward.
The estimator uses eight samples per member, averages **signed** contributions,
then clamps the result to [0, log M]. Monte Carlo noise and finite ensembles mean
this is not calibrated epistemic uncertainty. Shared errors can remain invisible.

Search maximizes expected `sum_h discount**h * U(z_h,a_h)` with discount 0.9,
horizon 4, epsilon-random 0.1. Models stay fixed inside each search; only the first
action is executed, and the tree is rebuilt at the next observation.

- `mcts`: UCT actions, 24 simulations, chance-node progressive widening
  `ceil(coefficient * visits**exponent)`, defaults 1 and 0.5. New outcomes are
  sampled from the uniform ensemble predictive mixture; existing outcomes are
  reused uniformly. Final choice uses root visit counts. There is no transposition
  table, node-merging radius, learned value/policy network, or oracle lookahead.
- `shooting`: 64 random action sequences, four particles per sequence, with
  stochastic next states and expected discounted disagreement. This has a different
  compute budget and does not condition later actions on sampled outcomes.
- `random`: the same model training schedule with persistent-random actions.

Repeated imagined visits receive repeated disagreement reward; models do not learn
inside imagination. Discounting and real-step replanning are the current controls,
not a guarantee against loops. Finite-horizon models can exploit invalid imagined
states. Goal matching and goal-directed control are not yet implemented in this mode.

To vary representation and compute independently, use `model.mixture_components`,
`model.ensemble_size`, `train.repulsion_weight`, `train.repulsion_margin`,
`updates_per_train`, `planner.horizon`, `planner.simulations`,
`planner.disagreement_samples`, `planner.candidates`, and `planner.particles`.
The existing `maze.observation=pixels` or `features` and `maze.nuisance_dims` options
also work. The reported pilot uses coordinates, not visual nuisance variation.

Runs save a checkpoint, append-only raw replay, config, source snapshot, training
and exploration JSONL logs, coverage/latent/uncertainty plots, and `probe.npz`.
Probabilistic evaluation uses an oracle lattice **only for diagnostics**: coverage,
seen/unseen state-action error, a persistence baseline, mean-rollout errors,
disagreement-error correlation, latent variance/effective rank, and sampled states
outside encoder bounds. It does not export an exact-identity graph. Online runs have
replay diagnostics; independent held-out episodes are only reported by offline
`train`. The all-state oracle probe includes both training and unseen transitions.

`scripts/refit_probabilistic.py` continues fitting a fixed replay with an optional
`--freeze-encoder`, recording disagreement every 500 updates. This is an optimization
diagnostic, not an independent exploration trial. See `training_runs.md` and
`docs/probabilistic-pilot.md` for actual results and limitations.

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

### Longer probabilistic sweeps

The manifest runner executes independent online runs in separate processes and
keeps raw logs, replay, checkpoints, source snapshots, and evaluator diagnostics:

```bash
.venv/bin/python scripts/run_probabilistic_sweep.py \
  --manifest configs/probabilistic_sweep.json \
  --out runs/new-probabilistic-screen --jobs 3
```

`configs/probabilistic_sweep.json` screens fifteen model/loss/search configurations
and a persistent-random reference on a 12×12 maze for 10,000 interactions. Its
selection metric is exact position-coverage AUC: average coverage after every
interaction, computed from raw replay. This rewards early discoveries rather
than only final coverage. The random reference trains the same model but chooses
actions independently of it. The scripts never provide evaluator geometry to
the encoder, dynamics, or planner.

A manifest supplies a base config, shared dotted-key overrides, variants, and
seeds. Reusing a completed output checks its resolved config; incomplete runs
are preserved and reported as failures. Start them again in a fresh output
folder after inspection. Each run still uses the source snapshot stored inside
its own directory. `summary.json` is updated as runs finish; `failures.json`
records exceptions. Use a separate manifest and fresh behavior/model seeds to
confirm selected settings at a larger budget.

```bash
.venv/bin/python scripts/report_probabilistic_sweep.py \
  --screen runs/new-probabilistic-screen \
  --long runs/new-long-random runs/new-long-selected \
  --out runs/new-sweep-report
```

The reporting script generates per-run tables, mean/range coverage curves, and a
machine-readable result bundle. It is specific to the documented 12×12,
10k-screen/100k-confirmation protocol; raw metrics remain available for other
protocols. Coverage is matched by environment interactions, not wall-clock
compute. No online training resume is implemented: a final checkpoint can be
used for offline refitting or frozen-policy rollouts, but does not restore an
interrupted online environment/planner state.

For a bounded-versus-unbounded representation ablation, use
`--set model.latent_activation=identity` with the probabilistic model. The default
is `tanh`, including when loading older checkpoints that omit this field. Identity
leaves the encoder output unbounded; likelihood and repulsion are unchanged.
Bound-saturation diagnostics are reported as null in that case. This is an
architecture option, not a new compression loss.

If an online run finished training and saved its checkpoint/replay but final
evaluation failed, recover only that evaluation after fixing the cause:

```bash
.venv/bin/python scripts/recover_evaluation.py \
  --run runs/your-completed-run --reason 'Describe the corrected evaluation issue'
```

The recovery command validates the complete interaction/update budget and matching
config/replay, and records unchanged hashes for the checkpoint, replay, and config.
It refuses incomplete training runs. Then rerun the same sweep manifest/output to
rebuild summaries; completed runs are reused and earlier failure records are
archived as resolved. Older configs are compared after filling default fields.

### Animated exploration replays

See [exploration animations](docs/exploration-animations.md) for the full-run and
late-episode comparisons, plus reproduction commands. The reusable
`scripts/animate_exploration.py` accepts one or more saved runs, labels, a time
window, frame stride, trail length, and playback speed. It shows accumulated
coverage and recent movement while keeping reset jumps out of the path.

### Training cadence experiments

Probabilistic online runs support `--set training_schedule=epoch` (one complete
shuffled replay pass per ensemble member at each training event), or
`--set training_schedule=new_then_replay` (each new transition once, then the
usual replay updates). The default `replay` retains the original sampling.
`train_every` controls the action interval; `updates_per_train` controls random
replay updates and is ignored by `epoch`. Both new schedules include all warmup
transitions and retain partial batches. Enable `cadence_diagnostics=true` to log
before/after prediction and disagreement measurements without affecting the
agent's RNGs or training data.

The matched three-seed comparison is configured in
`configs/probabilistic_cadence.json`; results and reproduction commands are in
[training cadence](docs/training-cadence.md).

### Fixed-data memorization diagnostics

[Memorization experiments](docs/memorization.md) separate fixed-encoder dynamics
fitting from joint representation training, comparing Gaussian NLL and mean MSE.
The experiment scripts retain partial batches, check for conflicting targets,
report training accuracy separately from unseen state-actions, and include
small-subset, fixed-coordinate, model-width, and repulsion-weight controls.
Use `configs/memorization.json` and `configs/memorization_followup.json` with
`scripts/run_memorization.py`; the report contains full reproduction commands.
These diagnostic weights are saved as `weights.pt` with experiment/model metadata,
not as the online runner's resumable `checkpoint.pt` format.

The follow-up [joint memorization experiment](docs/joint-memorization.md) reaches
above 99.2% in every ensemble member on two fresh initialization seeds. Encoder
and dynamics are both learned from scratch. Reproduce using
`configs/joint_memorization_confirm.json`; independently check saved weights with
`scripts/verify_joint_memorization.py`. `model.hidden_layers` controls coordinate
encoder and dynamics depth (default 2 maintains checkpoint compatibility).

### Online joint-MSE exploration

Set `train.prediction_objective=mse`, `train.encoder_lr_multiplier=0.3`, and
`model.mixture_components=1` to train the deterministic mean-prediction objective
within the existing joint model. MSE planning uses normalized mean disagreement,
ensemble-mean successors, and reward-scaled UCT; it never samples unused Gaussian
variance outputs. NLL training and stochastic planning remain the defaults.

The online cadence comparison and adaptive horizon/replay-budget follow-ups are
configured in `configs/joint_exploration_cadence.json`,
`configs/joint_exploration_horizon1.json`, and
`configs/joint_exploration_new20.json`. See the
[online joint exploration report](docs/joint-exploration.md) for results,
calibration diagnostics and the exploration animation.

See [reward-map animations and search diagnostics](docs/reward-visualization.md) for contemporaneous action rewards and visited-state tallies.

The [completed overnight study](docs/overnight-findings.md) covers 55 runs: horizon and ensemble-data sweeps, 30k-interaction follow-ups, matched-path rollout audits, and reward-map animations.

See [novelty diagnostics](docs/novelty-diagnostics.md) for 72 checkpoint audits relating prediction errors and disagreement to new-state discovery, directional choices, and imagined-path selection.

See [latent geometry](docs/latent-geometry.md) for PCA views, distance structure, action vectors, and ablations of low-variance latent directions.

See [larger search budgets](docs/search-budget.md) for matched online continuations at 24/96/384 MCTS simulations, a random control, fixed-model path diagnostics, and trajectory animations.

See [archive and path novelty experiments](docs/novelty-rewards.md) for controlled reward comparisons, real-history loop suppression, imagined-reward errors, and perfect-dynamics frontier-search diagnostics.

See [noisy-observation frontier search](docs/noisy-frontier.md) for the two-arm simulator-restoration experiment: nearest-visited latent distance versus ensemble disagreement, three matched seeds, nuisance-invariance diagnostics, and an exploration animation. Reproduce with `scripts/run_noisy_frontier.py` and `scripts/report_noisy_frontier.py`; configuration: `configs/noisy_frontier.json`.

The [persistent-frontier follow-up](docs/persistent-frontier.md) keeps all visited observations and queued states across training, and compares full-epoch training every 1,000 expansions against a matched-budget epoch interleaved every 100. Includes exact priority-refresh audits and a six-run exploration animation.

See [frontier sample-efficiency diagnostics](docs/frontier-efficiency.md) for repeat-expansion accounting, audits of overlooked frontier entries, fixed-data fitting interventions, privileged noise-removal controls, and paired online experiments on when to spend extra optimization.

See [queue density and top-k expansion](docs/queue-density.md) for rank-band novelty densities, within-batch duplication, and equal-action comparisons of fixed top-k, latent-diverse batches, and sequential queue updates.

See [learned-prior KL compression](docs/kl-bottleneck.md) for the β sweep, Gaussian versus mixture priors, delayed compression controls, nuisance-sensitivity diagnostics, and a three-seed exploration animation. The stochastic β=0 control is retained alongside the original deterministic baseline.

See the [joint KL × repulsion sweep](docs/kl-repulsion.md) for the full grid, intermediate-coefficient follow-up, fresh-seed confirmation, collapse versus nuisance-separation diagnostics, and exploration animation.

See [moment-fitted prior experiments](docs/moment-prior.md) for the analytical replay/batch Gaussian fits, a fresh β × repulsion sweep, delayed-compression tests, new-seed confirmation, and the full 100k-action animation. The fitted prior removed avoidable mismatch but did not improve exploration; the previous learned-prior setting remains preferred.

See [KL mechanism ablations](docs/kl-mechanisms.md) for controlled noise × KL comparisons, shared-data queue tests, faster prior optimization, and fresh-seed confirmation. KL without sampled dynamics inputs retained the exploration gain; fixing prior lag alone did not improve it at the current coefficients. Includes all results and a full 100k-action animation.

See [latent L2 and observation-noise experiments](docs/l2-noise.md) for the coefficient sweep, fresh-seed confirmation, and controlled queue diagnostics. L2 at λ=.03 completed all three fresh seeds faster than mean-input KL, despite a worse average noise/movement ratio. Includes shared-data ranking measurements, noise-redraw and clipping controls, an analytical collapse boundary, and a full 100k-action animation.

See [shared-replay latent geometry](docs/l2-geometry.md) for side-by-side baseline/KL/L2 maps, a rotating view, directional noise measurements, and queue-ranking diagnostics. The baseline becomes nearly one-dimensional; L2 preserves two substantial spatial directions.

The current [deterministic formulation](docs/deterministic-formulation.md) uses `z=f(o)`, residual dynamics MSE, repulsion, and optional latent L2. Use `configs/deterministic_frontier.json` with `scripts/persistent_frontier.py --config`; no posterior, prior, or predictive-variance outputs are built. Historical KL runs remain available separately. The document includes checkpoint conversion and exact-update validation.

### Continuous-coordinate frontier experiments

The [continuous maze study](docs/continuous-study.md) replaces the five lattice-producing actions with real-valued 2D controls, while retaining observation noise and the persistent latent-distance frontier. It compares joint dynamics and adjacency encoders, fresh seeds/layouts, action-proposal and noise controls, and smaller movements. Coverage is reported at several spatial resolutions; evaluation bins never enter the agent.

```
.venv/bin/python scripts/continuous_frontier.py --objective dynamics \
  --seed 100 --out runs/new-continuous-dynamics
.venv/bin/python scripts/continuous_frontier.py --objective adjacency \
  --seed 100 --out runs/new-continuous-adjacency
```

Use `configs/continuous_frontier.json` (the runner's default). Dynamics takes the actual action vector via `model.continuous_action_dim=2`; zero retains legacy discrete actions. The continuous runner supplies these vectors and uses simulator restores, not an imagined planner or actor. `--proposal iid`, `--step-size .125`, and `--nuisance-dims 0` reproduce the principal ablations. See [protocol and manifests](docs/continuous-study-protocol.md) for budgets and all staged decisions.

The [combined dynamics/adjacency and loss-scale study](docs/joint-loss-study.md) adds a jointly differentiated adjacency term and sweeps dynamics, adjacency, repulsion, L2, and hinge margin. `--objective hybrid --dynamics-weight 1 --adjacency-weight .3` activates the combination. `--repulsion-weight .03 --latent-l2 .03 --repulsion-margin .25` sets the regularizers; the hinge is `max(margin² - squared_distance, 0)`. The shared `Trainer` config uses `train.dynamics_weight` and `train.adjacency_weight`; existing defaults remain 1 and 0. The report includes frozen fresh-seed and additional-layout comparisons, failed configurations, and an animation.

The [compression and ancestry-prediction study](docs/compression-study.md) tests L1, bounded fixed-noise training channels, and multi-step prediction on real parent-pointer paths for both base objectives. Use `--experiment-spec <json>` with `scripts/continuous_frontier.py`; example specs and sweep manifests are in `configs/compression-20261111`. Search encodings remain deterministic. The report distinguishes noisy squared adjacency from a learned predictor after a noisy channel, and includes matched noiseless/H1 controls.

The [random-walk and hybrid study](docs/walk-study.md) compares independent and persistent walks, encountered-state resets, and walk prefixes followed by learned frontier search. `scripts/walk_frontier.py` keeps the existing frontier runner unchanged and accepts `--walk-steps`, `--persistence`, `--reset-every`, `--reset-until`, `--stay-probability`, and `--turn-on-block`. Unlearned controls use `--objective walk`. Protocol and immutable sweep manifests are linked from the report.

### Noisy pixel exploration

The completed [pixel study](docs/pixel-study.md) contains 372 runs of image-only representation learning and nonreactive walking controls. Spatial attention learns a noise-robust map and reaches 99.9% mean coverage in winding corridors; persistent walking remains stronger on the original maze and branching dead ends. [All fresh results](docs/assets/pixel-confirmation-details.md), [animation](docs/assets/pixel-snake-seed120.gif), [protocol and accounting history](docs/pixel-study-state.md).

The new `PixelMaze` uses one composited RGB image with noise across the whole frame; it is separate from the legacy two-channel pixel renderer above. The spatial encoder learns its own feature heatmaps, followed by a 2D latent and joint residual-dynamics training. No simulator coordinates or masks enter the learner. The working baseline uses a 5k observation-blind walk prefix, then persistent nearest-latent frontier expansion with exact simulator restores.

Run one 100k-action experiment (MPS on a compatible Mac):

```sh
.venv/bin/python scripts/run_pixel_study.py \
  --arms configs/pixel-study-20261111/working-baseline.json \
  --out runs/pixel-new-experiment --seeds 200 --workers 1
```

Use a new output directory. To use CPU, copy the arm JSON and change `device` to `cpu`; adaptive trajectories need not be bitwise identical across backends. `scripts/pixel_frontier.py --help` exposes maze families, image noise, architecture/loss settings and nonreactive walking controls. `scripts/animate_pixel_study.py` renders saved trajectories alongside the actual noisy input images.
