# Training run log

This file records executed experiments and their outcomes. Append future runs with configuration changes, dataset provenance, commands, metrics, and interpretation. Raw artifacts are under `runs/` (Git-ignored); the summaries here are intended to be versioned.

## 2026-11-02 — Initial maze distance-learning pilot

**Status:** completed. One seed, untuned hyperparameters. This validates the experiment infrastructure; it does not establish an exploration or navigation advantage.

### Question

Can QRL and Time-Myopic distance objectives learn a useful map from the same observed trajectories? Does a finite latent codebook with a usage-entropy penalty encourage reusable states without destroying distance information?

### Shared setup

| Setting | Value |
|---|---|
| Dataset | `runs/maze12-data/replay.npz` |
| Collection | 20,000 transitions; persistent random actions, repeat probability 0.8 |
| Maze | 12×12 rooms; layout seed 7; extra-door probability 0.18 |
| Motion | Continuous position; step size 0.25; right/left/up/down/stay controls |
| Reset | Center start; 400 steps per episode |
| Observations | Normalized coordinates; no nuisance dimensions |
| Experiment seed | 0 for all five variants |
| Split | 16,000 training transitions; 4,000 held-out transitions; complete episodes kept together |
| Model | 32 latent dimensions; hidden width 128; five bootstrapped dynamics heads |
| Training | 1,000 updates; batch size 256; Adam learning rate 0.001 |
| Runtime | CPU, one PyTorch thread; Python 3.12.7, PyTorch 2.15.0, NumPy 2.5.4 |
| Evaluation | Exact BFS distances on 1,784 reachable positions under the discrete controls |

Training data covers **64.6% of rooms** and **52.5% of reachable lattice positions**. This is coverage of the fixed random collector, not of any learned planner. All five variants use the same dataset and episode split. Oracle geometry is used only in evaluation.

### Loss-specific settings

* **QRL:** IQE-maxmean, eight latent coordinates per component; softplus push scale 20; initial Lagrange multiplier 1; projected dual ascent learning rate 0.01; epsilon 0.1 (average squared edge-violation tolerance 0.01).
* **Time-Myopic:** horizon 20; maximum sampled separation 100; MSE on capped elapsed-time targets. The minimum variant replaces exact observation-pair labels with their minimum observed separation within that window, including zero for exact self-pairs.
* **Compression:** optional 128-prototype codebook, soft-assignment temperature 0.1, usage-entropy weight 0.01, commitment weight 0.25. The two compressed variants use the same settings otherwise; compressed Time-Myopic uses raw elapsed-time labels.
* **Dynamics:** latent next-state MSE, with encoder inputs and targets detached, and independent bootstrap batches per ensemble head. No actor or goal-conditioned policy is trained.

### Results

| Variant | Local distance MAE (steps) | Mean self-distance | Effective latent rank | Used codes on oracle states |
|---|---:|---:|---:|---:|
| QRL | 6.351 | 0.000 | 2.003 | continuous |
| Time-Myopic (raw elapsed time) | 5.718 | 6.694 | 1.857 | continuous |
| Time-Myopic (exact-revisit minimum) | 3.634 | 0.000 | 2.022 | continuous |
| QRL + 128-code bottleneck | 11.252 | 0.000 | 1.625 | 3 |
| Time-Myopic + 128-code bottleneck | 4.691 | 12.983 | 0.000 | 1 |

Local MAE uses the same sampled oracle pairs whose shortest distance is at most 20, so this column shares a target across variants. Global MAE is not directly comparable: Time-Myopic clips its target at 20 whereas QRL does not. A collapsed constant predictor can obtain a superficially reasonable MAE; inspect self-distance, latent rank, and code usage alongside it.

| Variant | Held-out dynamics MSE | Held-out edge violation MSE | Held-out temporal MSE |
|---|---:|---:|---:|
| QRL | 0.0591452 | 0.22868 | n/a |
| Time-Myopic (raw elapsed time) | 0.0307415 | 51.11068 | 0.06876 |
| Time-Myopic (exact-revisit minimum) | 0.00744291 | 0.04982 | 0.02515 |
| QRL + 128-code bottleneck | 0.0154575 | 0.09430 | n/a |
| Time-Myopic + 128-code bottleneck | 1.08734e-12 | 143.58653 | 0.12397 |

Dynamics MSE depends on latent scale and can be tiny after collapse; it is not a standalone quality measure. Edge violation is `mean(relu(d(o, next_o) - 1)^2)`; only QRL explicitly constrains this quantity. Temporal MSE for the exact-revisit variant uses corrected labels, so it measures a different supervision target from raw Time-Myopic.

### Interpretation

* **Exact-revisit handling matters:** in this seed it reduces Time-Myopic self-distance from 6.694 to 0.000 and local shortest-distance MAE from 5.718 to 3.634. This supports investigating contradictory trajectory labels; it does not solve semantic state equivalence.
* **QRL is not converged:** uncompressed QRL has final minibatch squared edge violation 0.2483, above its 0.01 tolerance; its multiplier has risen to 4.2563. Its zero self-distance follows from the architecture, but the learned field does not yet reproduce the full maze topology.
* **Compression overmerged:** QRL uses only 3 of 128 prototypes on oracle states; Time-Myopic uses 1 and has effectively zero latent rank. These are failure cases, not successful graph simplification. A constant map also has no conflicting outgoing graph edges, so transition ambiguity alone cannot detect collapse.
* **Keep compression opt-in.** Follow-up experiments should separate the effect of the finite alphabet from the entropy penalty, vary codebook size/weights, and compare several seeds before drawing conclusions.
* **No planner-performance claim:** these five runs are offline representation experiments. Planning received functional smoke coverage separately, listed below.

### Run directories and artifacts

| Variant | Directory |
|---|---|
| QRL | `runs/maze12-pilot/qrl-k0-seed0/` |
| Time-Myopic (raw elapsed time) | `runs/maze12-pilot/time_myopic-k0-seed0/` |
| Time-Myopic (exact-revisit minimum) | `runs/maze12-myopic-min/` |
| QRL + 128-code bottleneck | `runs/maze12-pilot/qrl-k128-seed0/` |
| Time-Myopic + 128-code bottleneck | `runs/maze12-pilot/time_myopic-k128-seed0/` |

Each training directory contains `config.json`, `metadata.json`, `training.jsonl`, `checkpoint.pt`, `metrics.json`, `graph.json`, `diagnostics.png`, `latent.png`, and a `source/map_learning/` snapshot. `metadata.json` records per-source SHA-256 hashes. There was no Git commit at execution time (`git_head: null`); use the snapshots for exact code provenance.

Collection artifacts: `runs/maze12-data/`. Combined results: `runs/pilot-summary.csv`; four-way sweep results: `runs/maze12-pilot/summary.csv`. The narrative first-pilot summary is also in `docs/first-pilot.md`.

### Reproduction commands

Run from the repository root. The output names below match the completed runs; use fresh names when repeating, since the CLI refuses to overwrite existing run directories.

```bash
UV_CACHE_DIR=.uv-cache uv sync --extra dev
source .venv/bin/activate

map-learning collect --out runs/maze12-data

map-learning sweep \
  --data runs/maze12-data/replay.npz \
  --out runs/maze12-pilot \
  --seeds 0 --codebook-sizes 0 128 \
  --set train.updates=1000 --set train.log_every=250

map-learning train --config configs/time_myopic_min.json \
  --data runs/maze12-data/replay.npz \
  --out runs/maze12-myopic-min \
  --set train.updates=1000 --set train.log_every=250
```

### Functional checks executed

These use the 4×4 smoke configuration rather than the 12×12 comparison above. They check execution paths, not exploration superiority.

| Directory/check | What ran | Outcome |
|---|---|---|
| `runs/smoke-data/` | 1,200 random interactions | Replay collected successfully |
| `runs/online-disagreement/` | QRL; disagreement planner; 400 interactions, 200-step warmup | Completed; 18.75% room coverage |
| `runs/online-hybrid-compressed/` | Time-Myopic; 16 codes; hybrid planner; 400 interactions, 200-step warmup | Completed; 18.75% room coverage |
| `runs/resume-base/` → `runs/resume-check/` | 10 offline updates, then 10 additional updates from checkpoint | Checkpoint continuation completed |
| `runs/goal-rollout/` | Frozen resumed QRL model; goal (3.5, 2.5); 100 interaction steps | One completed episode; goal **not reached** |
| `runs/evaluate-check/` | Standalone checkpoint/replay evaluation | Completed |
| `pytest -q` | 17 tests | Passed, including loss behavior, IQE gradients/triangle inequality, reset-safe pairs, exact revisits, pixel/nuisance training, checkpoint continuation, known-dynamics planning, and collision geometry |

An explicit goal overrides the exploration scoring in the goal rollout even though its saved planner behavior is `random`. The online `train.updates` configuration is not the executed update count: online scheduling uses `train_every` and `updates_per_train` instead.

### Superseded runs — do not use for comparisons

`runs/superseded-wall-endpoint-bug/` contains earlier exploratory datasets and runs invalidated by a collision bug: a movement parallel to a wall could enter through its endpoint. The physical transition was fixed, an exhaustive reachable-state wall-contact regression test was added, and the current dataset and all five pilot models were regenerated.

An earlier evaluation attempt also exposed a NaN correlation for constant predictions. The evaluator now reports `null` when correlation is undefined, allowing collapsed models to be recorded rather than aborting the run.

### Suggested next runs (not executed)

1. Repeat the uncompressed baselines across several seeds; increase QRL optimization and inspect constraint convergence.
2. Compare raw versus minimum-corrected Time-Myopic labels on matched trajectories.
3. At fixed codebook sizes, ablate entropy weight down to zero and monitor code usage together with distance distortion and self-distance.
4. Add nuisance observations to test merging across different observations of the same physical position.
5. Compare random, distance, and disagreement planning under matched interaction budgets after validating the maps and dynamics.


## 2026-11-04 — Joint mixture dynamics and disagreement search


Implemented and executed a continuous-latent, jointly trained Gaussian-mixture ensemble with C-SWM-style repulsion. The agent uses distribution disagreement as intrinsic reward, with either stochastic MCTS or particle random shooting. No actor, codebook, explicit state merging, reconstruction, shortest-path labels, or learned distance objective is used.

## Main result

The implementation learns a noncollapsed representation and useful one-step dynamics, but **the first uncertainty-search policies do not beat persistent random exploration**. This is a pilot on one deterministic coordinate-observation maze, not a conclusion about the research idea.

| Online policy | Updates per run | Mean room coverage | Mean reachable-position coverage |
|---|---:|---:|---:|
| Persistent random | 1082 | 88.9% | 74.1% |
| MCTS | 1082 | 65.7% | 51.0% |
| Random shooting | 1082 | 57.4% | 46.3% |
| Persistent random, more fitting | 5410 | 88.9% | 74.1% |
| MCTS, more fitting | 5410 | 71.3% | 54.0% |

![Online coverage comparison](runs/probabilistic-summary/coverage-comparison.png)

## Exact setup

- Environment: existing continuous point maze, 6×6 rooms, layout seed 7, extra-door probability 0.18, step size 0.25, five actions (right/left/up/down/stay), center reset every 500 steps. There are 441 reachable positions under these controls.
- Observation: normalized 2D coordinates. No stochastic transitions, nuisance dimensions, or pixels in the pilot. The unit tests cover probabilistic training on pixels and noisy features, but those are not exploration results.
- Three model/behavior seeds (0, 1, 2), one fixed maze layout. Each run has 3,000 real interactions and 300 persistent-random warmup steps. Matched-seed warmups were checked for exact equality across methods.
- Representation: bounded continuous 8D tanh encoder, two 64-wide hidden layers. Three ensemble members, each with three diagonal Gaussian components. Independent initialization and independent replay resampling for each member; shared encoder.
- Mean = current latent + predicted residual. Variance = 1e-4 + softplus(raw); initialized around 0.01. Gaussian densities use stable log-sum-exp. No output clipping or projection during imagination.
- Training: mean member NLL per latent dimension + 10 × C-SWM hinge repulsion on independently sampled replay pairs, with Euclidean margin 1. Adam 3e-4; both source and target encoder branches receive gradients from likelihood. No detach or target encoder.
- Initial schedule: two updates every five interactions starting at step 300, for 1,082 total updates. Follow-up schedule: ten updates, for 5,410 total. All real observations are retained and re-encoded when sampled.
- Intrinsic reward: Monte Carlo generalized Jensen–Shannon divergence between complete member mixture distributions. Eight draws per member in search; 32 in evaluation. Component identities are irrelevant. Clamp only the final estimate to [0, log(3)] nats.
- Return: sum of per-transition disagreement discounted by 0.9 per step, horizon 4. Execute one real action and replan. Ten percent epsilon-random exploration.
- MCTS: 24 simulations, UCT coefficient 0.3, chance progressive widening ceil(visits^0.5). Chance samples come from the uniform ensemble predictive mixture; existing samples are reused uniformly. Select the root action by visit count. New tree for every real observation.
- Shooting: 64 independent random action sequences, four stochastic particles each, expected discounted score. This is an open-loop sequence search with a different compute budget from MCTS.
- Random baseline: repeat the previous action with probability 0.8, otherwise sample uniformly. It trains the same model on the same update schedule, but never consults it for actions. Its trajectories were verified identical between the two learning budgets.

## Per-seed results

| Updates/5 steps | Policy | Seed | Rooms | Reachable positions | Replay JS (nats) | All-state normalized MSE |
|---:|---|---:|---:|---:|---:|---:|
| 2 | random | 0 | 91.7% | 76.0% | 0.535 | 0.00332 |
| 2 | random | 1 | 75.0% | 66.2% | 0.307 | 0.00590 |
| 2 | random | 2 | 100.0% | 80.3% | 0.503 | 0.00253 |
| 2 | mcts | 0 | 61.1% | 50.8% | 0.589 | 0.00365 |
| 2 | mcts | 1 | 63.9% | 45.8% | 0.456 | 0.00725 |
| 2 | mcts | 2 | 72.2% | 56.5% | 0.610 | 0.00283 |
| 2 | shooting | 0 | 61.1% | 51.7% | 0.643 | 0.00306 |
| 2 | shooting | 1 | 47.2% | 36.3% | 0.546 | 0.01126 |
| 2 | shooting | 2 | 63.9% | 50.8% | 0.586 | 0.00290 |
| 10 | random | 0 | 91.7% | 76.0% | 0.195 | 0.00568 |
| 10 | mcts | 0 | 72.2% | 56.9% | 0.302 | 0.00248 |
| 10 | random | 1 | 75.0% | 66.2% | 0.242 | 0.00545 |
| 10 | mcts | 1 | 66.7% | 48.8% | 0.205 | 0.01801 |
| 10 | random | 2 | 100.0% | 80.3% | 0.212 | 0.00207 |
| 10 | mcts | 2 | 75.0% | 56.5% | 0.375 | 0.00168 |

Room coverage counts room cells visited; position coverage counts reachable physical lattice locations. These oracle labels are used only by the evaluator and never by learning/search. Different policies collect different data, so coverage is now an actual policy outcome rather than the same offline-data statistic repeated for each loss.

The normalized MSE is ensemble-mixture-mean error divided by each model's own average latent coordinate variance. All-state probes include seen and unseen transitions, not an independent test set. Independent encoders can distort geometry differently, so these errors are diagnostics, not a universal model ranking. NLL can be negative and is not directly comparable across learned latent coordinate systems.

## What the diagnostics say

All nine initial runs retained substantial latent variance and effective rank about 1.93–2.48. Their one-step means beat a current-state persistence predictor on the all-state probe. This establishes basic learning, not a faithful topology or reliable long-horizon planner.

In the initial MCTS runs, seen/unseen state-action disagreement was respectively 0.599/0.645, 0.399/0.388, and 0.604/0.645 nats. The reward does not reliably distinguish a familiar transition from an untried one. Across all nine initial runs, Pearson correlation between disagreement and one-step mean error ranged from about -0.328 to 0.209. Mean error is not a complete measure of distribution calibration, but this is weak evidence for using the current score as exploration value.

Gaussian imagination also extends beyond the bounded encoder's range. A single-sample-per-position probe for action 0 (right) found out-of-bounds rates from roughly 1% to 58% across initial runs. This measures a distribution-support mismatch, not the frequency of invalid states actually visited by search. There is no learned manifold constraint, and no clipping was silently added.

## Fixed-replay fitting diagnostic

Both diagnostics start from the initial random-seed-0 checkpoint and its same 3,000 transitions. They perform 2,000 additional updates with no new interaction, evaluating every 500 updates. One continues joint training; the other freezes the encoder. These are optimization probes, not new coverage trials.

| Model at evaluation | Replay JS | Replay normalized MSE |
|---|---:|---:|
| Starting checkpoint | 0.5349 | 0.003075 |
| Joint, +2,000 updates | 0.2370 | 0.004178 |
| Frozen encoder, +2,000 updates | 0.2472 | 0.002641 |

Disagreement falls on fixed experience even when the representation is fixed. Joint training reduces disagreement while mean error worsens in this probe, underscoring that agreement alone is not accuracy. The matched online follow-up with more fitting improves MCTS room coverage to 71.3%, still below random's 88.9%. No broad hyperparameter sweep or claim of superiority was made.

![Learning diagnostics](runs/probabilistic-summary/learning-diagnostics.png)

## Artifacts and reproduction

Run from `/Users/brycesandlund/GitHub/map-learning`; use fresh output names (the CLI refuses to overwrite runs):

```bash
.venv/bin/python scripts/run_probabilistic_pilot.py \
  --out runs/new-probabilistic-pilot --seeds 0 1 2

.venv/bin/python scripts/run_probabilistic_pilot.py \
  --out runs/new-probabilistic-more-learning --seeds 0 1 2 \
  --methods random mcts --set updates_per_train=10

.venv/bin/python scripts/refit_probabilistic.py \
  --checkpoint runs/probabilistic-pilot-seed0/random-seed0/checkpoint.pt \
  --data runs/probabilistic-pilot-seed0/random-seed0/replay.npz \
  --out runs/new-refit --updates 2000
# Add --freeze-encoder for the second refit diagnostic.
```

Actual runs are in `runs/probabilistic-pilot-seed{0,1,2}/`, `runs/probabilistic-more-learning-seed{0,1,2}/`, and `runs/probabilistic-refit-{joint,frozen}/`. Combined machine-readable results are in `runs/probabilistic-summary/all-results.json`. Each online run saves config, source hashes and snapshot, checkpoint, raw replay, training/exploration logs, metrics, coverage/diagnostic plots, and the evaluation-only `probe.npz`.

Validation: **25 tests pass**, including mixture likelihood against PyTorch distributions, bimodal fitting, sampling frequencies, permutation-invariant distribution disagreement, zero disagreement for shared aleatoric uncertainty, gradients through both encoder branches, joint noncollapse/learning, exact checkpoint continuation, MCTS chance averaging, shooting selection, and pixel/noisy-feature training. Existing QRL/Time-Myopic and collision tests also pass. Additional CLI smoke checks covered online MCTS, frozen shooting, offline fitting with held-out episodes, and resuming that fit from 20 to 25 updates. `compileall` and `git diff --check` pass.

## Next discriminating experiments

1. Compare distribution JS with a mean-disagreement reward and a one-step horizon, holding representation/data budgets fixed; this separates reward calibration from imagined-path exploitation.
2. Test longer fitting and confidence calibration on fixed data, with independent evaluation of transition likelihood and error. Independent replay batches do not make the ensemble a Bayesian posterior.
3. Add nuisance observations and branching transition noise after the clean-maze exploration baseline is understood. A bimodal unit test validates the distribution implementation, not robust aleatoric/epistemic separation in an environment.
4. Test multiple maze layouts and larger interaction budgets. The current three seeds vary initialization and behavior, not environment layout.

## 2026-11-04 — Longer probabilistic dynamics sweep (complete)

Completed **41 online runs / 1,580,000 interactions**, plus two 50k-update offline fits. Twelve runs reached 100k steps, nine follow-up runs reached 20k, and twenty screening runs reached 10k. No unresolved runs remain. All runs retain checkpoints, raw replay, resolved configs, source snapshots, logs, and metrics.

Full report and figures: [probabilistic-sweep.md](docs/probabilistic-sweep.md). Machine-readable bundle: `runs/probabilistic-sweep-summary-20261104/results.json`.

### Results at 100k steps

| Setting | Seeds | Mean position AUC | Mean final positions | Mean final rooms | Position range |
|---|---:|---:|---:|---:|---:|
| random | 3 | 83.8% | 99.4% | 100.0% | 99.0–99.8% |
| random_iid | 3 | 61.5% | 82.6% | 84.5% | 79.4–86.9% |
| single_gaussian | 3 | 63.1% | 74.2% | 78.2% | 57.3–90.9% |
| wide128 | 3 | 50.8% | 56.3% | 63.2% | 42.7–64.9% |


### Matched first 20k steps

| Setting | Seeds | Mean position AUC | Mean final positions | Mean final rooms | Position range |
|---|---:|---:|---:|---:|---:|
| base | 3 | 32.3% | 45.7% | 52.3% | 38.5–58.9% |
| horizon8 | 3 | 32.9% | 48.2% | 58.1% | 39.8–55.5% |
| random | 3 | 48.7% | 72.5% | 81.7% | 69.3–75.8% |
| random_iid | 3 | 26.3% | 42.2% | 47.2% | 35.4–53.3% |
| single_gaussian | 3 | 38.8% | 54.7% | 60.2% | 51.0–60.9% |
| unbounded | 3 | 31.8% | 45.9% | 52.8% | 32.3–60.1% |
| wide128 | 3 | 35.2% | 49.1% | 56.2% | 37.4–57.2% |


Removing the tanh encoder bound improved fixed-data next-position retrieval from **74.3% to 89.3%**. In the online activation follow-up, mean retrieval improved from **52.6% to 64.5%**, but coverage remained **45.7% versus 45.9%**. The paired offline fits used identical data, split, initialization seed, and sampling sequence. The online activation follow-up is exploratory and reuses confirmation seeds.

No learned configuration beat persistent random in the mean matched-budget results. Single-Gaussian MCTS explored faster than iid random early (100k AUC 63.1% versus 61.5%), but stalled and ended with less coverage. Two single-Gaussian seeds found no new positions in the last 20k steps despite moving on 85–87% of those steps. Global latent variance alone also missed a severe loss of vertical sensitivity in one iid-control encoder.

Validation: **30 tests passed**; all 41 interaction/update budgets, replay lengths, checkpoint existence, coverage consistency, and matched warmups were audited. One post-training evaluation error was recovered with checkpoint/replay/config hashes unchanged; the failure archive records its resolution. No training was discarded. `compileall` and `git diff --check` pass.

### Protocol and decisions recorded during the run


Requested a hyperparameter/architecture sweep and substantially longer training. Screening manifest: `configs/probabilistic_sweep.json`; runner: `scripts/run_probabilistic_sweep.py`; artifacts: `runs/probabilistic-sweep-20261104/`.

Predeclared setup: 12×12 maze, layout seed 7, coordinate observations, reset every 2,000 steps, 1,000 random warmup steps, 10,000 total interactions per screening run. Base training has five updates every ten steps (4,505 total). Fifteen learned variants plus a persistent-random baseline, screening seed 0. Variants change latent dimension (2/8/16), hidden width (64/128), ensemble members (3/5), Gaussian components (1/3), repulsion weight (1/10/30), variance floor (1e-4/1e-3), learning rate (3e-4/1e-3), updates per training event (5/20), search horizon (1/4/8), random-action probability (.1/.3), and MCTS versus shooting. This is a one-factor screen, not a full factorial search; horizon 8 also doubles simulations to 48.

Selection: rank learned configurations by exact reachable-position coverage AUC, defined as mean coverage after every real interaction. Confirm the top two for 100,000 steps on fresh model/behavior seeds alongside a matched random baseline. AUC rewards early discovery; final coverage and steps to coverage thresholds are also recorded. No oracle state labels enter the learned model or planner. Screening selection uses evaluator labels only, as ordinary hyperparameter tuning.

Infrastructure: isolated worker processes, bounded concurrency, full per-run configs/source snapshots/checkpoints/replays/logs, exact coverage AUC and threshold computation, explicit failure logs. Replay online sampling now gathers only sampled rows instead of rematerializing the entire buffer after every append. Cached/offline and uncached/online samples match exactly under the same RNG. Removed unused latent-scale computation from probabilistic training. Validation before launching: 27 tests passed.

Additional controls during the sweep: a 100k independent-uniform-action random baseline on seeds 3–5 (`configs/probabilistic_long_iid.json`) measures the advantage from action persistence. An evaluation-only MC noise probe (`scripts/probe_disagreement_noise.py`, `runs/probabilistic-disagreement-noise-20261104.json`) compares eight-sample immediate-disagreement action rankings with a 256-sample reference on 64 states, 20 repetitions. Agreement was 62.4% for the 10k base checkpoint, 28.0% for latent-2, and 60.8% for the 100k random-seed-3 checkpoint. The 256-sample reference is still an estimate; this is not a control policy or an oracle-correct-action test.

Based on that diagnostic, added an adaptive four-run screen (`configs/probabilistic_sweep_followup.json`): latent dimension 4, 64 MC samples/member, latent-2 with 64 samples, and horizon-1 with 64 samples. Same 10k seed-0 budget. Selection for fresh-seed confirmation uses AUC across both screening batches. This extension is explicitly adaptive, not part of the original 16-run grid.

While the final two screening runs completed, launched 100k single-Gaussian and width-128 runs on fresh seeds 3–5, the leading completed settings at that point. Their outcomes are excluded from hyperparameter selection; if the completed grid ranks a different setting above them, these remain additional architecture controls. Manifests: `configs/probabilistic_long_gaussian.json` and `configs/probabilistic_long_wide.json`.

Added post-hoc evaluation-only physical next-state retrieval (`scripts/evaluate_next_state_retrieval.py`): predict the ensemble-mixture mean, match its latent to the closest encoded position in the full maze catalogue, and compare with the true next position. Direct-distance computation avoids squared-norm cancellation near saturated coordinates. This is a particular readout, not an optimal decoder or a test of the agent's planning loop. Also measure horizontal/vertical latent displacement on valid physical movements. A uniform-random seed-5 model at 100k has mean horizontal/vertical displacement ratio about 758 despite substantial latent variance and effective rank ~2.04; its nearest-latent next-position accuracy is only 12.8%. This demonstrates that global spread and low latent MSE alone do not establish preservation of navigational information.

Compute-driven confirmation change: horizon-8/48-simulation MCTS overtook the other learned screening settings (its monotone-coverage AUC lower bound at ~7.3k already exceeded single-Gaussian). It expands about 300 search nodes/action rather than 60 and takes much longer per interaction. Confirm horizon-8 on fresh seeds 3–5 for **20,000 steps** using `configs/probabilistic_long_deep.json`, comparing it with the exact first-20k prefixes of all other policies. The single-Gaussian and width-128 architecture tests still run for **100,000 steps**. This explicitly replaces the initial plan to give both eventual screen finalists 100k; do not compare the 20k final coverage directly against 100k finals or label horizon-8 a 100k result. Width-128 remains the additional architecture control if displaced from the top two by horizon-8.

Added an opt-in probabilistic encoder setting, `model.latent_activation=identity`, to remove the tanh output bound while keeping the default unchanged. Old checkpoints/configs default to tanh. Saturation and out-of-encoder-bounds metrics are null for an unbounded encoder rather than falsely reporting a bound violation. All 29 tests pass, including a new quantitative JS comparison with independent density quadrature and default/checkpoint compatibility for the activation setting.

Controlled activation diagnostic: two fresh offline fits on exactly the same `runs/probabilistic-long-iid-20261104/random_iid-seed5/replay.npz`, with seed 5, 50,000 updates, and the same whole-episode 80/20 split. Only tanh versus identity changes. Outputs: `runs/probabilistic-activation-{tanh,identity}-20261104/`. These fits use stored experience and add no interactions. They are excluded from exploration ranking. Unlike the online run, they see the full training buffer from the outset, so only the paired offline comparison isolates activation.

The paired 50k-update offline fits completed. Tanh: 74.3% all-state next-position retrieval, mean physical error 0.1711, horizontal/vertical latent-step ratio 0.98. Identity: 89.3%, error 0.03394, ratio 2.90. Only 0.011% of identity predictions have more than one room of Euclidean error, versus 5.17% for tanh. Both were trained on the same 80k transitions and evaluated on the same 20k held-out transitions plus the full evaluation-only maze catalogue. The fresh offline tanh fit preserves both directions, unlike the original online iid seed-5 run, so the bound alone does not explain that failure; online curriculum/optimization also differs. Effective covariance rank is lower for identity (1.38 versus 2.85), demonstrating why this statistic is not itself a representation-quality score. Full metrics: `runs/probabilistic-sweep-summary-20261104/activation-comparison.json`.

Based on this paired diagnostic, added a matched **20k online** activation comparison on seeds 3–5: default `base` versus `unbounded`, otherwise identical GMM ensemble and horizon-4 MCTS (`configs/probabilistic_online_activation.json`). This is an adaptive follow-up, excluded from the original 20-run screening ranking. Compare these six runs with other policies at the common 20k horizon. No further configurations are planned in this batch. Offline coverage-image captions were clarified as replay coverage; model weights and training metrics were unchanged.

Recovery note: the single-Gaussian seed-5 run completed all 100k steps and saved its checkpoint/replay, then final evaluation failed because the process predated the new `latent_activation` config field and lazily imported the updated evaluator. Added a legacy-field default in evaluation and re-evaluated the saved run with `scripts/recover_evaluation.py`. It validates completed step/update budgets and config/replay consistency, and verifies checkpoint/replay/config SHA-256 hashes are unchanged. `evaluation_recovery.json` records the repair. Rebuilding the sweep summary validates all three completed runs and archives the earlier failure as resolved. No training was discarded or repeated. New completed-run config validation fills newly introduced default fields for older configs.

## 2026-11-09 — Training cadence experiment (completed)

Question: does promptly fitting every transition, or repeatedly fitting all experience, reduce dynamics-ensemble disagreement and improve useful prediction/exploration?

Manifest: `configs/probabilistic_cadence.json`. Three matched seeds (0, 1, 2), 10,000 actions each, 12×12 maze seed 7, center reset every 2,000 actions, 1,000 persistent-random warmup actions. All conditions use the same tanh encoder (8 latents, width 64), three single-Gaussian dynamics heads, batch size 128, Adam 3e-4, repulsion weight 10, and MCTS horizon 4 / 24 simulations. No policy network is trained.

- `replay`: existing five independently sampled replay minibatches per head every 10 actions.
- `epoch`: one full shuffled pass over all collected transitions per head every 10 actions. Independent permutation per member; final partial batch retained. Replaces the five random replay updates.
- `new_then_replay`: one shuffled pass over transitions collected since the last training event, followed by the original five replay updates. First event includes all 1,000 warmup transitions; subsequent events include 10 new transitions. The new batch has its own optimizer update, not dilution inside a batch of old transitions.

Encoder and dynamics are jointly trained in every update. Repulsion still uses 128 random replay-state pairs per update. Both new schedules flush final partial collection blocks and guarantee every collected transition has reached every head by the end of the run. Uniform random replay baseline has no such explicit guarantee.

`cadence.jsonl` logs before/after diagnostics at each training event: latest 10 transitions, 128 older replay samples, and a fixed 128-transition maze-wide evaluation probe. These track JS disagreement (8 MC draws/member), NLL, latent MSE normalized by probe latent variance, persistence baseline error, predictive variance, and latent variance. Before/after samples are identical and MC seeds shared. Probe transitions never enter replay or optimization; they can overlap experienced state-actions, so this is not a disjoint held-out set. Final evaluation separately reports seen/unseen state-actions over the full maze. Diagnostic RNGs are isolated from learning and planning.

Coverage is measured against both environment actions and wall time; training time is logged separately. These online conditions collect different trajectories, so final prediction errors are not a fixed-data comparison. Three seeds are an exploratory comparison, not a conclusive ranking.

Run command:

```bash
MPLCONFIGDIR=/tmp/map-cadence-mpl .venv/bin/python scripts/run_probabilistic_sweep.py \
  --manifest configs/probabilistic_cadence.json \
  --out runs/probabilistic-cadence-20261109 --jobs 3
```

All **nine runs completed** (90,000 real actions; 147,228 joint updates total), with all requested transition-coverage guarantees verified. Tests: **36 passed**. Seed-0 replay-baseline actions and positions exactly match the previous single-Gaussian seed-0 screen, validating that diagnostics did not change the trajectory.

| Schedule | Updates/run | Final position coverage, mean (range) | Coverage AUC | Final replay JS | Physical next-position exact retrieval |
|---|---:|---:|---:|---:|---:|
| Replay baseline | 4,505 | 37.4% (30.8–42.7%) | 25.3% | 0.1449 | 41.2% |
| Full epoch | 39,158 | 41.2% (32.1–47.9%) | 26.9% | 0.1220 | 30.1% |
| New once + replay | 5,413 | 37.1% (29.0–42.7%) | 22.9% | 0.1026 | 34.1% |

Extra fitting reduced final average replay disagreement but did not improve the common physical readout. Full epochs increased mean final coverage by 3.8 percentage points (two seeds better, one worse), using 8.7× the updates and about 1.6× elapsed collection time. New-once training reduced disagreement the most, with essentially unchanged mean final coverage and lower coverage AUC. These results do not establish insufficient replay fitting as the main bottleneck. Every added optimizer update also trains repulsion, so this is not an isolated dynamics-only optimization test.

Mean elapsed minutes: replay 1.61, epoch 2.65, new-once 1.65; corresponding training-only minutes 0.13, 1.13, 0.15. Concurrent workers and diagnostics limit wall-time comparisons. Physical prediction uses nearest-latent retrieval over all reachable state-actions, not an optimal decoder. Persistence achieves 31.9% exact accuracy and 0.1702 mean physical error, compared with model errors 0.5008 / 0.6401 / 0.6978, respectively.

Full results, before/after diagnostics, plots, seed-0 exploration GIF, and reproduction commands: [training cadence report](docs/training-cadence.md). Portable numeric results: `docs/assets/cadence-results.json`. Raw runs: `runs/probabilistic-cadence-20261109/`. Source snapshots match across all nine runs.

## 2026-11-09 — Fixed-data memorization diagnostics (completed)

All conditions use `runs/probabilistic-cadence-20261109/replay-seed0/replay.npz` (10,000 transitions) and its encoder checkpoint. No exploration or new training observations. Dataset audit: 2,606 unique state-action inputs, 762 distinct observed states, no contradictory next-state targets. All 762 states have distinct float32 encoder outputs; minimum latent separation 0.00957. Stationary transitions are 42.18% of replay; repeated edges occur up to 53 times.

Manifest: `configs/memorization.json`; runner: `scripts/run_memorization.py`; experiment: `scripts/memorize_dynamics.py`. Eight initial conditions, each 20,000 updates: frozen versus jointly trained encoder × NLL versus mean MSE, plus frozen-MSE controls on 128 unique transitions, fixed normalized coordinates (repeated to eight dimensions), dynamics width 256, and uniform unique-edge sampling. All dynamics are freshly initialized with seed 7, all optimizers reset. The pretrained encoder is identical initially in the learned-encoder conditions. Full replay training uses independent shuffled passes per head, including final partial batches. Adam starts at 3e-4 and cosine-decays to 1e-5. The joint conditions retain repulsion weight 10; changing NLL to raw MSE changes its relative strength, so this is not a scale-matched loss comparison.

Frozen encoded inputs and targets are precomputed once. Frozen/coordinate conditions omit repulsion because it cannot affect their parameters. MSE trains Gaussian means only; variance and likelihood diagnostics are not interpreted or reported for MSE runs. Each experiment saves arguments, source/data hashes, source snapshot, selected transition indices, weights and a learning curve.

Training metrics evaluate every fitted transition (both visitation-weighted and unique-edge accuracy). Generalization is measured separately on a fixed sample of up to 1,024 state-actions absent from that condition's training set, never used for gradients. Physical accuracy is nearest-latent retrieval against the full-maze catalogue, only available to evaluation; it does not imply every continuous latent residual is zero. No exact-state graph merging or codebook enters learning. A tiny 128-edge subset is chosen once with a fixed seed shared by any matching controls.

```bash
MPLCONFIGDIR=/tmp/map-memorize-mpl .venv/bin/python scripts/run_memorization.py \
  --manifest configs/memorization.json --out runs/memorization-20261109 --jobs 3
```

Validation: **38 tests pass**, plus an end-to-end 100-update frozen-encoder smoke run with exactly zero encoder drift. All 12 substantive experiments completed (480,000 updates total); no new environment interactions.

**Matched 20k-update results, accuracy across distinct training state-actions:** frozen NLL 72.10%, frozen MSE 86.19%, joint NLL + repulsion 10: 57.21%, joint MSE + repulsion 10: 56.75%. The 128-edge frozen-MSE control reaches 100% for each ensemble member (latent MSE 1.75e-6).

**Adaptive follow-ups:** at 100k updates, uniform-edge frozen MSE achieves 98.00% with width 64 and 98.31% with width 256 (52 and 44 remaining incorrect edges, respectively). Width-256 fixed-coordinate dynamics achieves 99.65% (9 incorrect edges). Initial LR is 1e-3 with cosine decay, versus 3e-4 in the original 20k runs, so the longer-vs-shorter difference includes sampling and optimization changes. Full exact memorization has not been achieved; 100% refers only to the 128-edge subset and discrete next-position retrieval, not zero continuous regression residual.

**Loss-balance ablation:** changing only joint-MSE repulsion weight from 10 to 0.01 at the original 20k budget raises unique-edge accuracy from 56.75% to 90.94% (visitation-weighted: 69.08% to 94.44%), surpassing frozen MSE at that budget. At common initialization over 32 minibatches, median encoder repulsion/prediction gradient norm ratios are 188.4 for MSE at weight 10, 0.188 for MSE at weight 0.01, and 3.77 for NLL at weight 10. Thus the extreme 188× imbalance is specific to the MSE comparison, not the original NLL online objective.

This supports fixed-target fitting as feasible and loss weighting as a major issue in the joint-MSE test; it does not establish that joint representation learning is inherently inferior. Dataset determinism and exact encoder collisions are ruled out for the sampled training data. The last 1–2% of errors in frozen learned latents are still unresolved, and this is one dataset/initialization. No sweeping generalization claims.

Full report, learning curves, unseen-edge scores, parameter counts and reproduction commands: [memorization experiments](docs/memorization.md). Raw runs: `runs/memorization-20261109/` and `runs/memorization-followup-20261109/`. Portable results/audits: `docs/assets/memorization-*.json`. Follow-up manifest: `configs/memorization_followup.json`. Gradient audit: `scripts/audit_memorization_gradients.py`.
