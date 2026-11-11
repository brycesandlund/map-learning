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

## 2026-11-09 — Joint memorization ≥98% target (completed)

Target: at least 98% next-position retrieval for **each** ensemble member over all 2,606 unique training state-actions, with a learned, noncollapsed encoder and no frozen targets. Same seed-0 10k-transition dataset as the preceding diagnostics; no new exploration.

The new models have three SiLU hidden layers of width 256 in both encoder and each dynamics head, eight latent dimensions, and three independent dynamics heads. Both encoder and dynamics are freshly initialized and jointly optimized from the first update. MSE of each head's mean plus hinge repulsion; no reconstruction, codebook, coordinate targets, oracle training signals or actor. The source checkpoint supplies configuration only for fresh encoders. Encoder LR = 0.3× dynamics LR; dynamics LR starts at 3e-4, both cosine-decay to 1/30 of their initial value. Independent shuffled passes over unique observed edges retain partial batches. Repulsion uses independently sampled training endpoints.

Initial seed-7 screen: 50k updates per condition, tanh/repulsion 0.01, tanh/repulsion 0.001, and identity-output/repulsion 0.01. Manifest `configs/joint_memorization.json`. The identity-output model exceeded 98% for each head at the 1k checkpoint; two new initialization seeds (8/9) then test that recipe for 10k updates each, with cosine decay shortened to that budget. Manifest `configs/joint_memorization_confirm.json`. This is adaptive exploration, not a pre-registered comparison.

Accuracy uses nearest-latent retrieval over the full evaluation-only maze catalogue. Every head is reported separately, including the fraction of edges correct for all heads simultaneously. MSE and latent variance/rank/drift are retained; unseen state-actions are evaluated separately. Production MLP depth is now configurable through `model.hidden_layers` (default 2 preserves old checkpoints). Experiment-only knobs choose encoder/dynamics sizes independently, fresh initialization, latent activation and encoder learning rate.

```bash
MPLCONFIGDIR=/tmp/map-joint-mpl .venv/bin/python scripts/run_memorization.py --manifest configs/joint_memorization.json --out runs/joint-memorization-20261109 --jobs 3
MPLCONFIGDIR=/tmp/map-joint-mpl .venv/bin/python scripts/run_memorization.py --manifest configs/joint_memorization_confirm.json --out runs/joint-memorization-confirm-20261109 --jobs 2
```

**Target achieved in both 10k-update confirmation runs**, independently verified by reloading weights. All five runs finished (170,000 updates total); **39 tests pass**.

| Run | Updates | Head 1 | Head 2 | Head 3 | Correct for all heads | Unseen ensemble accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Identity, seed 8 | 10,000 | 99.27% | 99.23% | 99.23% | 99.23% | 95.41% |
| Identity, seed 9 | 10,000 | 99.31% | 99.27% | 99.27% | 99.27% | 96.29% |
| Identity, seed 7 | 50,000 | 97.93% | 98.04% | 97.93% | 97.39% | See full report |
| Tanh, repulsion 0.001, seed 7 | 50,000 | 96.70% | 97.35% | 97.20% | 96.28% | See full report |
| Tanh, repulsion 0.01, seed 7 | 50,000 | 93.98% | 93.71% | 93.25% | 91.63% | See full report |

Both confirmations preserve 762 distinct float32 training-state latents. Their training-state latent variance is about 1.318, minimum distinct-state distances 0.0429/0.0480, mean distances about 4.11, and moving-transition distance about 0.172. Exact expected repulsion is 0.03966/0.03964, of which 0.001572 is the unavoidable identical-state floor. Encoder drift from initialization is nonzero (~1.26 MSE), confirming a learned rather than frozen representation.

Winning configuration: 134,408 encoder parameters and 418,611 dynamics parameters (139,537/member), totaling 553,019. Unbounded learned outputs, MSE + 0.01 hinge repulsion, fresh joint initialization, uniform observed-edge training, encoder LR multiplier 0.3. Gaussian variance outputs are unused in this MSE diagnostic. These results establish the requested fixed-data memorization level, not calibrated stochastic uncertainty or live exploration performance.

Longer training did not guarantee retention: the seed-7 identity model exceeded 99% early but declined to ~98% at 50k. This remains a limitation for ongoing learning. No claim that capacity alone caused the improvement: architecture, initialization, sampling and optimizer schedule differ from the earlier small-model runs. The bounded/unbounded comparisons within the initial screen are matched; confirmation budgets are shorter and chosen adaptively.

Full verified results, learning curves, latent geometry and reproduction commands: [joint memorization](docs/joint-memorization.md). Portable artifact: `docs/assets/joint-memorization-results.json`. Independent verifier: `scripts/verify_joint_memorization.py`. Raw checkpoints: `runs/joint-memorization-confirm-20261109/identity_seed{8,9}/weights.pt`. All experiments use the same dataset hash; no new interactions or privileged transitions enter training.

## 2026-11-09 — Joint MSE online exploration (completed)

Authorized experiment: compare a full raw-replay epoch every 10 real actions against every new transition once followed by five random replay updates, with matched exploration policy/model. Three seeds (0–2), 10k interactions each, fixed-center reset every 2,000 actions, 1,000 persistent-random warmup actions. Both schedules explicitly fit all warmup transitions and retain final partial batches. Twelve initial runs include a persistent-random behavior control for each cadence and seed; paired random controls should collect identical trajectories.

Model: fresh encoder and three fresh dynamics heads, each three width-256 SiLU layers; eight unbounded latents. Joint per-head mean MSE + 0.01 hinge repulsion. Constant dynamics LR 3e-4, encoder LR 9e-5. No pretrained encoder/weights, reconstruction, actor, codebook or oracle-guided training. Raw replay retains visitation frequencies: this differs from uniform unique-edge fitting in the memorization diagnostic and is deliberately held fixed across both cadences. Constant LR replaces finite-budget cosine decay to keep adapting to new data.

Planner: MCTS horizon 4, 24 simulations, discount 0.9, 10% random-action exploration. Imagined successors are deterministic ensemble means, never untrained Gaussian draws. Intrinsic reward is ensemble variance of predicted means averaged over latent dimensions, divided by a replay-state latent-variance scalar fixed within each search. For this MSE mode only, UCT's exploration coefficient is multiplied by the root maximum reward times the discounted horizon sum; otherwise the old absolute 0.3 bonus would overwhelm tiny MSE-scale rewards. No chance-outcome widening in deterministic mode. Legacy NLL/distributional search is unchanged.

Training and final diagnostics label disagreement as normalized mean variance, not JS nats; MSE likelihood/variance fields are null. Before/after cadence probes use isolated RNGs and never feed training/planning. Evaluation on the full maze is privileged, post hoc only. The planner's variance normalization uses replay states, not the evaluation probe.

Validation: **41 tests pass**, including tiny-reward action selection, invariance to unused predictive variances, joint encoder updates, optimizer LR ratio and exact MSE checkpoint continuation. A 103-step end-to-end smoke test verified the final partial training block and MSE-aware evaluation.

```bash
MPLCONFIGDIR=/tmp/map-live-mpl .venv/bin/python scripts/run_probabilistic_sweep.py \
  --manifest configs/joint_exploration_cadence.json --out runs/joint-exploration-cadence-20261109 --jobs 3
```

**All 21 runs completed**, including both adaptive follow-ups: 210,000 real interactions and 457,923 joint updates. Every raw transition was explicitly trained in every head by the end of each run. Source snapshots match across all runs. **42 tests pass**.

| Policy / cadence | Final coverage | Coverage AUC | Seen-edge retrieval | Updates/run |
|---|---:|---:|---:|---:|
| MCTS H4, full epoch | 53.4% | 29.9% | 92.8% | 39,158 |
| MCTS H4, new + 5 | 43.5% | 28.9% | 58.1% | 5,413 |
| MCTS H4, new + 20 | 49.3% | 30.8% | 80.8% | 18,928 |
| Persistent random, full epoch | 59.8% | 33.4% | 90.7% | 39,158 |
| Persistent random, new + 5 | 59.8% | 33.4% | 56.4% | 5,413 |
| MCTS H1, full epoch | 37.4% | 22.0% | 94.5% | 39,158 |
| MCTS H1, new + 5 | 49.8% | 30.8% | 55.1% | 5,413 |

Means over three seeds; all runs use 10k actions. More replay strongly improves retained fitting and the H4 exploration condition, but no learned condition beats persistent random on mean coverage. One-step search hurts the full-epoch condition while helping new+5, so reducing imagined-rollout depth is not a consistent fix. This is an exploratory one-layout comparison, not a statistical generalization claim.

**Matched-data finding:** random-epoch and random-new actions/observations/positions were verified identical for all three seeds. Full epochs achieve 90.7% seen-edge retrieval vs 56.4% for new+5. Yet the worse-fitting new+5 model has ~4.2× **less** replay disagreement (3.23e-6 vs 1.35e-5), with ~9× larger normalized replay MSE (5.30e-4 vs 5.87e-5). Agreement is not a reliable convergence/fitting criterion. Restricting nearest-latent retrieval to visited-state catalogues barely changes the accuracy gap (90.8% vs 56.5%).

**Exploration signal finding:** final H4/full-epoch untried-action AUROC within visited states is 0.461/0.523/0.473, roughly chance; whole-maze unseen-edge AUROC is 0.611 on average. Being untried need not imply uncertainty when the model generalizes correctly, and these final diagnostics are not a causal explanation of every online action. They nevertheless show the reward is not a dependable local visited/unvisited discriminator in these runs.

The ≥98% fixed-dataset fitting level is not maintained online. Raw visit-frequency weighting, continually moving targets and constant learning rates differ from the successful offline fit; this experiment does not isolate those factors. The next bottlenecks are stable online fitting and a reliable uncertainty reward, not merely extra MCTS search.

Full report, before/after diagnostics, per-member retrieval, calibration tables and seed-0 animation: [joint online exploration](docs/joint-exploration.md). Portable results: `docs/assets/joint-exploration-results.json`. Raw data: `runs/joint-exploration-{cadence,horizon1,new20}-20261109/`.

Adaptive follow-up: `configs/joint_exploration_horizon1.json` compares horizon 1 with both cadences, seeds 0–2 (six more 10k-action runs), after the horizon-4 full-epoch curves showed slow exploration around one-third coverage midway through collection. This removes multi-step imagined-state reward while retaining the same model, optimizer, replay schedule and root disagreement objective. Runs are under `runs/joint-exploration-horizon1-20261109/`, two concurrent workers alongside the primary comparison. It tests a planning hypothesis rather than changing training capacity.

Second adaptive follow-up: `configs/joint_exploration_new20.json` increases new-once's replay updates from 5 to 20, retaining horizon 4 and all other settings (three seeds). This was selected after the first three new-once runs achieved only ~58% seen-edge retrieval versus ~93% under full epochs. It tests an intermediate compute budget (18,928 updates/run versus 5,413 and 39,158). It is an adaptive comparison, not a new untouched validation set. The first full-epoch checkpoints also showed local untried-action AUROC near chance (0.46–0.52); calibration is measured only after fitting and never fed to training or planning.

## 2026-11-09 — Visualization-only matched reruns

Added visited-state tallies and contemporaneous reward snapshots every 200 actions. Reran the three seed-0 MCTS cadences under `configs/reward_visualization.json`, saved to `runs/joint-reward-visualization-20261109`. All replay arrays and model/dynamics weights exactly match the original runs. No policy or training change. Added exhaustive horizon-4 search diagnostics; sampled MCTS choices are within 1% of optimum in 71.4%, 90.0%, and 84.4% of decisions for full epoch, new+5, and new+20 respectively. See [visualization and limitations](docs/reward-visualization.md). Full test suite: 44 passed, including recorder noninterference and exhaustive-return checks.

## 2026-11-09–10 — Overnight horizon / ensemble-data study (completed)

**All 55 runs completed and audited: 730,000 interactions, 4,854,377 joint gradient updates.** Same 12×12 maze and coordinate observations; three seeds except the H64 extension (two). Full epochs every ten actions, joint MSE + 0.01 repulsion, three dynamics heads and encoder each width 256 / three layers, eight unbounded latents. No architecture change, graph merging, reconstruction, actor or oracle-guided learning.

| Stage | Runs | Interactions/run | Manifest |
|---|---:|---:|---|
| H4/8/16/32 × shared/independent | 24 | 10k | `horizon_pilot.json`, `horizon_confirm.json` |
| Matched random controls: epoch/new+20 × shared/independent | 12 | 10k | `ensemble_matched.json` |
| H64 × shared/independent | 4 | 10k | `horizon_long.json` |
| H16 MCTS96 and H32 shooting256 | 6 | 10k | `horizon_search_followup.json` |
| H8 independent / H16 shared / random duration follow-up | 9 | 30k | `horizon_duration.json` |

Manifests are under `configs/`, using `horizon_base.json`. The main horizon sweep uses discount 0.99 and 24 MCTS simulations. Prior H4 experiments used discount 0.9, so pure-horizon comparisons use this sweep. Duration conditions were selected from the 10k sweep; this adaptive selection is disclosed.

**Horizon result:** mean 10k coverage independent/shared: H4 53.1/52.0%, H8 57.0/53.9%, H16 45.8/58.2%, H32 48.6/51.8%, H64 47.7/54.5%; random 59.8%. H16 MCTS96 yields 46.6% versus 45.8% at24 simulations (one seed improves, two decline). H32 shooting256 yields 42.8%. Longer or more expensive search is not an automatic improvement.

**Duration result:** at30k, H8 independent / H16 shared / random achieve 85.9/85.6/81.4% final coverage and 98.9/99.6/99.3% seen-edge retrieval accuracy. Learned agents win final coverage in two of three seeds each; coverage AUC is 59.7/60.6/60.5%, so the late gain is not a consistent overall exploration advantage. Every 30k run reproduces its corresponding 10k replay prefix exactly. Full-epoch training grows from39,158 to352,721 updates per run; increased data and increased optimization are coupled.

**Same-path rollout result:** using identical first10k replay sequences, H8 exact32-step retrieval improves44.1→89.9% and H16shared improves41.0→93.4%. Random control also improves37.2→92.0%. These are known paths, not proof of reliable predictions beyond the frontier. Audits retain normalized MSE, teacher-forced error, physical retrieval/error, manifold distance, known-edge prefix fractions, candidate reward ranking, and robust optimism/radius metrics through128 steps. A shooting checkpoint has an actual latent runaway at128; online sweeps therefore stopped at64.

**Shared-batch result:** all12 matched controls have identical replay arrays within each seed. Shared batches reduce mean-normalized reward-map shape changes in all six paired comparisons:16% for full epochs,21% for new+20. Accuracy is about one percentage point lower. Shared encoder gradient diversity changes too; this is not a head-only optimization-noise ablation.

Artifacts: [readable findings](docs/overnight-findings.md), [all results](docs/horizon-experiments.md), [batch comparison](docs/ensemble-batch-findings.md), [rollout methodology/failure case](docs/latent-rollout-audits.md). Repo assets include `horizon-rewards.gif`, `ensemble-batches.gif`, `duration-rewards.gif`, coverage/rollout plots, and portable `horizon-results.json`. Raw runs under `runs/horizon-{pilot,confirm,long}-20261109/`, `runs/ensemble-matched-20261109/`, `runs/horizon-{search,duration}-20261110/` retain configs, source snapshots/hashes, replay, intermediate checkpoints and per-run audits.

Validation: 48 tests passed before the sweeps and again at final verification, including shared-batch sampling/RNG alignment, recording noninterference, exhaustive-return checks and perfect-dynamics rollout checks. Legacy random-epoch replay AND final model weights reproduced exactly for all three seeds. Final report generation asserts all matched-control streams and duration prefixes. No partial runs or failures remain; changes are uncommitted.

## 2026-11-10 — Isolate replay sharing within new-once + 20

Three fresh `shared_replay` runs in `runs/replay-only-sharing-20261110` compared with the existing three `new_then_replay_independent` matched random controls. The explicit new-transition pass (including initial warmup batches) keeps independent per-head orders in both conditions; only the 20 replay updates broadcast the first head's sampled indices in the treatment. Sampling consumes equal RNG draws, preserving the repulsion sampling stream. Exact equality of all replay arrays verified for every seed. Config: `configs/replay_only_sharing.json`.

Mean normalized spatial L1 change:0.5834 independent vs0.4775 shared-replay-only (18.1% reduction). Adjacent-map rank correlation:0.5816→0.7354. Top-decile overlap:0.4538→0.5708. These are temporal reward stability metrics over snapshots at5k–9.8k actions, not prediction accuracy or active-exploration gains. The shared encoder's gradient diversity also changes when replay indices are shared.

Examples generated by `scripts/compare_replay_sharing.py`: `docs/assets/replay-sharing.gif` and `.png`, seed0. Top row uses a common fixed numerical reward scale (pooled99th percentile after2k actions); bottom shows per-frame percentiles of max-action reward. Thus magnitude and spatial ordering are separated. Numeric rewards remain dependent on each learned representation. Full48-test suite passed before the added targeted test; the four ensemble-batch tests then passed, including explicit-new-pass preservation in shared-replay-only mode.


## 2026-11-10 — Prediction error, disagreement, and actual novelty

Completed **72 saved-checkpoint audits across 24 existing runs**, three seeds per condition. No retraining, added agent interactions, or changes to prior run artifacts. Five matched-random conditions (epoch independent/shared; new+20 independent/shared-all/shared-replay-only) at 2k/6k/10k; H8 independent, H16 shared, and random full-epoch duration runs at 6k/18k/30k. Novelty is defined from each checkpoint's replay prefix, including both transition endpoints. Oracle maze adjacency and the full latent catalogue are evaluation-only.

The main distinction is **new successor state versus untried action**. Among actions at visited states, final H8/H16 disagreement identifies a new successor with AUROC 0.820/0.838, despite untried-action AUROC only 0.469/0.557. Latent-error novelty AUROC is 0.803/0.830. Restricting to untried actions retains disagreement novelty AUROC 0.792/0.754. Physical successor mistakes occur on 11.9% of novel transitions in both conditions, versus 2.2%/1.1% for transitions to familiar states. Thus most novel transitions are still predicted correctly. Max-disagreement action selection at a frontier enters a new state in 40.7%/43.9% of cases, versus random 23.7%/23.0%.

Matched-data comparison: independent full epochs give disagreement novelty AUROC 0.673 versus 0.562 for new+20. Sharing only replay batches gives 0.582; among untried actions, independent/shared-replay are essentially equal at 0.606/0.607. Shared batches' previously measured reward stability does not establish better novelty discrimination; path-selection comparisons are mixed.

Final checkpoints also tested **64 visited starts × 128 candidate paths × two action samplers**, with checkpoints at path lengths 4/8/16/32/64. Reward scores use discount 0.99. Outcomes count distinct previously unvisited physical states, never repeated visits. Identical candidate banks compare real-state disagreement, real-state latent error, and recursive imagined disagreement. Real-state scoring is privileged diagnostic information. On independent epoch 10k controls, horizon-32 persistent paths yield 1.85 new states under random selection, 4.95 under real-state disagreement, and 2.88 under imagined disagreement. Within-start score–novelty Spearman is 0.320/0.197 for real/imagined disagreement; imaginary reward selects zero novelty despite a novel candidate in 46.3% of eligible starts (real-state reward: 26.2%). The rollout gap varies by model and is negligible for H8 30k mean selected yield.

Candidate generation also misses reachable novelty. For H8/H16 30k, 184/190 of 192 sampled starts have a novel state reachable within 32 actions. IID candidate banks find one from only 103/99 starts; persistent banks find one from 127/136. These are frozen-model candidate-bank diagnostics, not the actual MCTS tree or a causal account of online behavior. One maze with deterministic coordinate observations and exact discrete novelty; no continuous-state equivalence claim.

The initial 24-start/64-candidate pilot was expanded to improve path-sampling coverage. Reported results use only `runs/novelty-audit-v2-20261110`; the pilot remains separately preserved. All matched conditions' candidate novelty outcomes were verified identical per seed. Scores use a full-catalogue variance scalar instead of replay variance, which leaves within-checkpoint ranks and maximizing choices unchanged.

```bash
.venv/bin/python scripts/analyze_novelty.py --manifest configs/novelty_matched.json --out runs/novelty-audit-v2-20261110
.venv/bin/python scripts/analyze_novelty.py --manifest configs/novelty_duration.json --out runs/novelty-audit-v2-20261110
MPLCONFIGDIR=/tmp/map-novelty-mpl .venv/bin/python scripts/report_novelty.py
.venv/bin/python -m pytest -q
```

**52 tests pass**, including directed frontier distances, absence of frontiers, uniform tie handling, and unique-state counting with exact dynamics. [Full findings and methods](docs/novelty-diagnostics.md), [figure](docs/assets/novelty-diagnostics.png), and portable per-checkpoint `docs/assets/novelty-results.json`. Raw outputs retain per-seed metrics, intermediate checkpoints, distance bands, path correlations, opportunity counts, and horizon-dependent selection outcomes.


## 2026-11-10 — Search budget and latent geometry (completed)

Authorized follow-up: increase MCTS simulations while preserving training and inspect latent geometry. Twelve online branches start from the H16/shared 30k checkpoints (seeds 0–2): simulations 24/96/384 and persistent random, each 1,200 additional interactions, full replay epoch every ten actions. Model/optimizer/trainer RNG and full replay restored; planner/shuffle/scale RNGs restart identically within each seed, with fixed-center reset and no warmup. This is a controlled branch, not an exact old-process continuation. Root: `runs/search-budget-20261110`; launcher: `scripts/run_search_branches.py`. No loop penalty or learning-objective change.

Geometry audit covers H8 independent and H16 shared, three seeds, initialization plus 6k/18k/30k checkpoints. All graph distances and visual adjacency overlays are evaluation only. Learned PCA2 explains 98–100% of variance at 30k, but input ablations show crucial low-variance information: in one H16 model, successor retrieval falls 92.0%→18.3% after projecting to two components, recovering to 92.1% with three. Three components recover nearly all accuracy for five of six models; the sixth needs four. This does not establish a sufficient encoder architecture without retraining.

The encoder resembles a nonlinear coordinate surface and does not consistently expand pairs separated by walls. The dynamics nevertheless predict near-zero movement for blocked actions. Random-initialization visual controls show spatial structure before learning, since inputs are already coordinates. Additional loop-detectability probes distinguish moving returns from fresh positions through recursive latent predictions; no diagnostic threshold enters the agent. Details and figures: [latent geometry](docs/latent-geometry.md). A separate fixed-model MCTS audit inspects visit-selected 16-step paths at 24/96/384 simulations.

Validation: 52 tests pass; a ten-action continuation smoke test preserves every original replay array exactly and explicitly trains all appended transitions. Existing completed runs and uncommitted work preserved.

All twelve branches and fitting audits completed: **14,400 additional interactions and 345,012 joint updates** (28,751 per branch). Every original replay array is unchanged; all branches explicitly train through transition 31,200. Initial reward-map arrays and positions match exactly across conditions per seed. Worker hash and protocol retained in `runs/search-budget-20261110/protocol.json`.

| Behavior | New states by seed 0 / 1 / 2 | Mean new states | Seen-edge accuracy | Moving actions revisiting last 16 states |
|---|---|---:|---:|---:|
| MCTS 24 | 72 / 3 / 2 | 25.7 | 99.42% | 52.3% |
| MCTS 96 | 71 / 3 / 4 | 26.0 | 99.54% | 53.9% |
| MCTS 384 | 0 / 3 / 2 | 1.7 | 99.34% | 61.0% |
| Persistent random | 24 / 0 / 10 | 11.3 | 99.51% | 24.3% |

At 96, mean cumulative discoveries are slightly lower than at 24 despite similar final counts. The 384 decline is entirely driven by seed 0; the other two seeds are unchanged. All conditions retain novel-successor disagreement AUROC around 0.80–0.82. No universal search-budget conclusion from three short branches on one maze.

Fixed-model MCTS audit: same 24 visited starts per original checkpoint, three seeds, H16 and budgets 24/96/384. Preferred visit-selected paths repeat 8.40/9.25/9.57 successors out of 16, with 8.60/7.75/7.43 distinct path states including the start. Both imagined and real-state disagreement returns rise, while new-state counts are non-monotonic (0.35/0.65/0.57). The diagnostic wrapper verifies root choices, counts and Q values against production MCTS. These open-loop preferred paths are distinct from online executed trajectories.

[Search report](docs/search-budget.md), [geometry report](docs/latent-geometry.md), [trajectory animation](docs/assets/search-budget-paths.gif), [reward animation](docs/assets/search-budget-rewards.gif). Geometry includes per-state latent arrays, random-initialization controls, shortest-path comparisons, action-vector views, PCA-input ablations, and moving-loop detection. At H16, loop AUROC is 0.987 for H16/shared models and 0.973 for H8/independent models; thresholds were selected diagnostically and never fed to the agent. Local latent step spacing varies by 3.5–8.9× (95th/5th percentile), with positive visitation correlation 0.31–0.64.

Reproduction: `scripts/run_search_branches.py`, then `scripts/audit_search_branches.py` and `scripts/report_search_budget.py`; geometry: `scripts/analyze_latent_geometry.py`, `scripts/audit_geometry_dynamics.py`, `scripts/report_latent_geometry.py`; fixed-model search: `scripts/audit_mcts_budget.py`. Experiment launchers refuse existing output directories. All jobs completed; no automation or background experiment remains active. Figures and animations inspected; git diff --check clean. No production algorithm change or loop penalty added.


## 2026-11-10 — Archive and path novelty rewards (completed)

Four experimental MCTS rewards from the same H16/shared 30k checkpoints, seeds 0–2, 1,200 additional actions, 24 simulations, unchanged full-replay epoch every ten actions: root distance, minimum distance to imagined history, minimum distance to archive plus imagined history, and disagreement multiplied by d/(d+observed-step-scale). Raw observations are retained; exact duplicate observation bytes are removed only from the nearest-neighbor cache, which does not change a minimum. The entire archive is re-encoded after each training block; new observations enter it immediately. No representation loss or state-merging threshold added.

Distance rewards normalize by the same global latent-variance scalar; UCT uses a floor based on median observed moving-transition length because archive reward at a known root can vanish. Disagreement gating uses that length as a smooth scale, not a binary loop threshold. This scaling choice is part of the objective comparison, and a global scale still has the previously measured spatial nonuniformity limitation. Reward modes live in metadata/result and the experiment runner, not the legacy planner config factory.

Primary manifest: `configs/novelty_rewards.json`, root `runs/novelty-rewards-20261110`; launcher `scripts/run_novelty_rewards.py`. A seed-0 disagreement control reproduced all previous replay arrays and final model/dynamics weights bit-for-bit. All 55 tests pass, including new archive re-encoding and repeat-reward tests. Experiment-only planner: `src/map_learning/novelty_planning.py`; runner snapshots are copied into each run.

Adaptive decision: oracle shortest distance from the fixed reset to unvisited territory is 20/24/23 actions for seeds 0/1/2. Perfect H16 archive reward is therefore initially zero for every plan. Added H32 archive/path reward at the same 24 simulations, three seeds, no other change (`configs/novelty_horizon.json`, `runs/novelty-horizon-20261110`). This was chosen after diagnostic evidence, not prespecified.

Fixed-path diagnostics score identical persistent-random candidate banks (32 starts × 64 paths, horizons 8/16/32/64) with imagined and true-state rewards. The raw outputs are under `runs/novelty-candidate-audit-v2-20261110` with portable `docs/assets/novelty-candidate-results.json`; the first audit is retained separately. The expanded audit measures imagined archive reward assigned to actually non-new steps. A separate perfect-dynamics UCT diagnostic (`scripts/audit_oracle_novelty_search.py`) uses privileged geometry only for evaluation, never for online action selection or training.


Adaptive follow-ups within the novelty study: perfect-dynamics UCT at H16/H32/H64 and budgets 24/96/384 never reached a novel state from reset in 15 searches per setting. Persistent action candidate banks reached novelty in 5/15 H64 trials versus 0/15 IID banks (128 candidates), motivating `configs/novelty_persistent.json`: archive/path H64, 128 persistent shooting candidates, three seeds, 1,200 actions. This changes search, horizon and candidate budget together and is explicitly exploratory.

Path-only planning produced zero new states in all three seeds while real short revisits remained frequent, motivating `configs/novelty_recent.json`: archive/path novelty plus equal-weight novelty relative to the last 16 real observations and the imagined path, H16/24 simulations. This real history survives replanning, is re-encoded after training, and clears on reset. No spatial cutoff or learned state merging; the temporal window is a new hyperparameter. All source/runner snapshots preserved per run. New shooting and recent-memory tests cover deterministic progress, re-encoding and reset behavior.


Completed all **22 new runs**, including one exact reproduction control: **26,400 added interactions and 632,522 joint updates**. Every condition preserves the 30k replay prefix and performs 28,751 updates. Disagreement and persistent-random comparison rows reuse the previous matched branch study. No jobs or monitors remain running.

| Reward / search | New states, seeds 0 / 1 / 2 | Mean | Moving recent-revisit rate |
|---|---|---:|---:|
| Disagreement H16 (prior control) | 72 / 3 / 2 | 25.7 | 52.3% |
| Root distance H16 | 0 / 0 / 0 | 0.0 | 53.5% |
| Path distance H16 | 0 / 0 / 0 | 0.0 | 55.2% |
| Archive + path H16 | 65 / 7 / 0 | 24.0 | 52.1% |
| Disagreement × path gate H16 | 6 / 9 / 2 | 5.7 | 50.9% |
| Archive + path H32 | 6 / 22 / 18 | 15.3 | 50.1% |
| Archive, persistent shooting H64 | 16 / 0 / 13 | 9.7 | 47.6% |
| Archive + recent real history + path H16 | 1 / 7 / 5 | 4.3 | 10.4% |
| Persistent random (prior control) | 24 / 0 / 10 | 11.3 | 24.3% |

No objective consistently improves discovery. The real-history reward materially reduces short loops in all three seeds, but does not improve global exploration. Final seen-edge retrieval remains about 99.4–99.6% in every condition. Root/path agents move more often but remain in familiar territory. Equal weighting of archive and recent novelty is only one tested mixture, not an optimization over shaping coefficients.

Fixed-candidate evidence: at H16, random selection discovers 0.232 new states/path, actual-trajectory archive scoring 1.458, imagined archive scoring 0.906, and the maximum available novelty is 1.562. At H32 the true/imagined archive values are 3.031/1.292; at H64, 5.750/1.646. Selected imagined archive reward assigns 86% of its mass at H16 (95% at H64) to steps that physically enter archived or within-path repeated states. This is erroneous state-novelty reward, not proof that the prediction is outside the continuous encoder manifold.

Perfect-dynamics UCT cannot reach novelty from reset in any of 15 trials per H16/H32/H64 × 24/96/384 setting; nearest frontiers are 20/24/23 steps away. This isolates the difficulty of generating directed routes through familiar regions from model prediction error. Persistent candidate sampling helps reachability slightly but is not sufficient in the online follow-up. Privileged diagnostics never enter training or online decisions.

Artifacts: [full report](docs/novelty-rewards.md), `docs/assets/novelty-reward-results.png`, `archive-reward-diagnostic.png`, `novelty-rewards-paths.gif`, `novelty-followups-paths.gif`, and portable per-seed JSONs. Run manifests and reproduction commands are in the report. Snapshot files from the experiment runner contain disagreement probes; they are not mislabeled as new-objective heatmaps. The archive heatmap is explicitly a final-checkpoint diagnostic. All plots and animation endpoint frames inspected.

**57 tests pass**, including archive/recent re-encoding, lossless duplicate caching, history reset, exact reward examples and persistent shooting on exact dynamics. Original source/replay artifacts preserved; git diff --check clean. All changes remain uncommitted.

## 2026-11-10 — Noisy-observation priority queues (completed)

User-authorized two-arm study: nearest-visited latent distance versus incoming-edge ensemble disagreement, with simulator restoration. Reused `MazeConfig.nuisance_dims=8`: normalized coordinates plus eight fresh independent N(0,1) nuisance values. Deterministic maze transitions unchanged. Added an experiment-only vector-backed maximum-priority queue (`src/map_learning/frontier.py`) and runner (`scripts/noisy_frontier.py`). No physical-ID, observation-byte, or latent deduplication; no threshold. Distance keys decrease eagerly when expanded latents enter the phase-local visited set. Disagreement keys are the incoming parent/action's mean-prediction variance, frozen within a phase. Simulator positions are opaque restoration payloads and evaluation labels; never ranking inputs.

Protocol: three matched seeds, fresh initial weights, 20 phases × 1,000 expansions × five executed actions = 100k action executions per run. No warmup. Center restart and empty queue/visited each phase; retain all raw replay and trained parameters. One shared full-replay epoch after each phase, batch 256, yielding 4,112 updates per run. Width 256 × three hidden layers, latent eight, three heads, existing joint mean-MSE + 0.01 hinge repulsion; both encoder endpoints receive gradients. Encoder/dynamics LR 9e-5/3e-4. Cached source observation retained with each snapshot; successor noise is fresh and its RNG is not rewound by restoration. All five actions including stay are executed. Each branching transition gets a separate replay episode ID. No imagined dynamics rollouts.

Config `configs/noisy_frontier.json`; launch `scripts/run_noisy_frontier.py --out runs/noisy-frontier-NEW --jobs 3`. Completed root `runs/noisy-frontier-20261110`; six runs, 600,000 actions, 120,000 expansions, 24,672 gradient updates. Separate two-phase/50-expansion smoke test excluded. All six exit successfully. Matched parameter hashes, budgets, and successor noise arrays verified identical between arms within seed. Full replay and initial/intermediate/final checkpoints retained; source snapshots and runner copies included. No preexisting run overwritten.

| Priority | Observed states, seeds 0 / 1 / 2 (of 1,784) | Mean coverage | Final-phase unique expanded positions (of 1,000 pops) |
|---|---|---|---|
| Nearest-visited latent distance | 1,730 / 1,081 / 1,139 | 73.8% | 1,000 / 367 / 393 |
| Incoming-edge ensemble disagreement | 118 / 127 / 103 | 6.5% | 4 / 12 / 5 |

Positive finding: existing joint training suppresses nuisance variation without adding a loss. Initial same-position independent-noise distance / adjacent-position centroid distance is 92–142; final ratio is 0.13–0.14 (distance) and 0.10–0.15 (disagreement). Common noise draws across neighboring evaluation positions prevent centroid sampling variance from masquerading as spatial structure. Evaluation probes use isolated RNGs and never enter training. Per-phase diagnostics are after training; that phase's coverage was collected before training.

Disagreement repeatedly selects familiar positions while its scores are frozen: final-phase repeated-expansion fractions 99.6/98.8/99.5%; one seed expands one position 966 times. Unlike distance, it has no within-phase score reduction as experience accumulates. This identifies a weakness of the tested incoming-disagreement-only queue/cadence, not uncertainty-based exploration in general. Both arms learn nuisance invariance and low normalized replay MSE. Distance remains seed-sensitive and repeats substantially in two seeds. Clean coordinate features remain present; no pixels, continuous-position change, or stochastic physical dynamics tested. Only two arms were run; no claim that learning causally beats a frozen random encoder based on coverage alone.

Artifacts: [report](docs/noisy-frontier.md), `docs/assets/noisy-frontier-results.png`, `noisy-frontier.gif`, `noisy-frontier-final.png`, portable per-seed `noisy-frontier-results.json` and duplicate/noise-stream audit `noisy-frontier-audit.json`. Reproduce reporting with `scripts/report_noisy_frontier.py --root runs/noisy-frontier-20261110`. Animation shows simulator restoration jumps and cumulative observed states, not an executable path.

Validation: **61 tests pass**, including eager key updates, exact brute-force distance priority agreement, retained duplicate entries, empty frontier handling, and observation-noise/physical-transition isolation. Figures and animation inspected. Changes remain uncommitted; all prior work preserved.

## 2026-11-10 — Persistent noisy-maze distance frontier (completed)

User-authorized comparison of **two distance-only arms**: retain the full visited observations/frontier at the existing 1,000-expansion training cadence, versus retain them and interleave training every 100 expansions. No new disagreement runs and no loss change. Same eight Gaussian nuisance dimensions, clean normalized coordinate features, maze seed 7, width-256/three-layer models, latent eight, three jointly trained dynamics heads, mean-MSE + 0.01 hinge repulsion, and prior learning rates. No duplicate removal or merging threshold; physical positions only restore/evaluate. All five real actions executed per expansion, including stay; 100,000 actions/20,000 expansions per run.

Persistent queue stores raw observations and opaque restoration payloads. After every training event, re-encode all visited observations and all active frontier observations and recompute all minimum latent distances. Added `src/map_learning/persistent_frontier.py`; exact SciPy cKDTree indexing (eps=0) accelerates nearest-distance lookup without changing the objective. Newly visited points immediately reduce active priorities. All 20,000 visited entries and 80,001 unexpanded frontier entries survive through the end. Fresh observation noise on successor generation; cached source observation on restoration; no search/origin restart after initialization.

Budget matching: every 1,000-expansion block, both arms train on every accumulated transition **exactly once per head** and take ceil(5,000 × block_number / 256) updates. Frequent arm causally partitions the old replay into ten disjoint shuffled slices and adds each interval's 500 new transitions; each new transition is trained at the next update. It uses minibatches of 250–300 to match both sample count and gradient count, whereas epoch arm uses 256 and tails (minimum eight). Temporal ordering and minibatch sizes differ; not a pure timing-only comparison. Repulsion sampling and shared-ensemble batching unchanged. Each run: 4,112 updates, 1,050,000 transition presentations per head. All six runs total 600,000 new real actions and 24,672 joint updates.

| Condition | Observed states, seeds 0 / 1 / 2 (of 1,784) | Mean coverage | Actions to first complete coverage |
|---|---|---|---|
| Earlier restarted-search distance (reference only) | 1,730 / 1,081 / 1,139 | 73.8% | Not reached in these runs |
| Persistent, epoch every 1,000 expansions | 1,782 / 1,784 / 1,632 | 97.1% | Not reached / 82,733 / not reached |
| Persistent, interleaved every 100 expansions | 1,784 / 1,784 / 1,741 | 99.2% | 68,628 / 77,266 / not reached |

Frequent training reaches 90% coverage in 61,778 / 64,002 / 94,263 actions, versus 88,868 / 76,549 / 96,814 at the old cadence. Mean 73,348 versus 87,410: 16.1% fewer actions, direction consistent across three seeds. It does not dominate every earlier milestone (seed 1 reaches 50% later). Persistent old-cadence search improves final coverage in every seed over the prior restarted comparison, though seed 0's intermediate coverage is slower. These are three seeds on one maze, not a general guarantee or significance claim. One frequent-training seed remains 43 states short; no claim of solving every seed within budget.

Mean elapsed time: old cadence ~60.0 seconds/run; frequent ~75.6 seconds/run, with three concurrent workers. Exact re-encoding/priority-refresh costs ~1.4 vs 13.5 seconds; training ~34 seconds in both. Wall times are hardware/concurrency specific. Training after the final expansion contributes to diagnostics and cost but cannot affect coverage.

Root: `runs/persistent-frontier-20261110`, all six subprocesses exited successfully. Runners: `scripts/persistent_frontier.py`, `scripts/run_persistent_frontier.py`; reporting/audit: `scripts/report_persistent_frontier.py`. Manifest, source snapshots, raw replay, initial/intermediate/final checkpoints, final frontier archive, and frame data preserved. Existing runs not overwritten. SciPy added to project dependencies, version 1.18.1 installed in the virtual environment. A too-small 100-expansion smoke configuration failed because it could not allocate one gradient step to each interval; CLI validation was tightened to >=500 and a separate valid smoke run passed. Both smoke paths retained/excluded from results.

Audit: matched initial parameter hashes, every successor-noise draw, sample and update counts at all block boundaries. Independently re-encoded all saved active frontier/visited observations, verified every final priority, and compared 32 queries per run against brute-force nearest distances. **64 tests pass**, including persistence across encoder changes, incremental updates after index refresh, and causal exact-once epoch exposure across all 20 blocks. No oracle labels influence ranking or learning.

Artifacts: [report](docs/persistent-frontier.md), `docs/assets/persistent-frontier-results.png`, `persistent-frontier.gif`, `persistent-frontier-final.png`, `persistent-frontier-results.json`, and `persistent-frontier-audit.json`. Animation shows restoration jumps; no executable navigation path is implied. Reproduce with `scripts/run_persistent_frontier.py --out runs/persistent-frontier-NEW --jobs 3`, then `scripts/report_persistent_frontier.py --root runs/persistent-frontier-NEW`. All jobs finished; prior work preserved; changes uncommitted.

## 2026-11-10 — Noisy-frontier sample-efficiency investigation (completed)

Authorized thorough investigation with adaptive follow-ups. Audited six original persistent runs at 5k/25k/50k/100k checkpoints; ran **18 new experiments**: six online extra-optimization runs, eight fixed-archive fitting/frozen-exploration branches, and four privileged nuisance-removal controls. Total new interaction 720,000 real actions; 44,672 new joint updates. No representation loss or state-merging changes. Every previous run preserved.

Repeat accounting excludes actions after completion (except the containing five-action expansion). Original frequent-training seeds repeat previously expanded physical positions in **87.1 / 88.6 / 91.7%** of expansions before completion or budget. Direct replay checks confirm repeated expansions never reveal new physical positions: all five deterministic actions were already executed. Repeated noisy observations can still aid learning; these counts measure zero physical-coverage yield, not zero information value.

A descriptive isolated probe first has same-position noise distance below adjacent-position distance at 30k / 30k / 25k actions. Repeat actions before that checkpoint: 27,170 / 27,455 / 23,135; after it, before completion/budget: 32,575 / 41,010 / 68,570. Probe crossing is sampled every 5k and is not a causal sufficient-training criterion or a policy threshold. Both early representation acquisition and later priority inefficiency contribute.

Seed 2 final archive has 43 unobserved states, 34 queued physical positions that can immediately reveal an unobserved neighbor, and best productive entry at rank 7. But 88 of its top 100 priorities repeat expanded positions. Ranking is a static post-training snapshot; entry ordering changes after each expansion. Exact reconstruction from raw observation entry identity matches all final saved archives. No oracle identity enters actual search.

### Fixed archive fitting interventions

Starting from identical original frequent-training 50k checkpoints/frontiers, compare no extra fitting against 2,000 updates on existing replay; then freeze both models for 10k additional real actions. Paired successor-noise draws; no new data used for training during these branches. New states found:

| Seed | Initially observed | Frozen original | Extra fitting then frozen | Known nuisance removed (privileged control) |
|---|---:|---:|---:|---:|
| 0 | 1,340 | 231 | 444 | 444 |
| 1 | 1,393 | 264 | 391 | 391 |
| 2 | 718 | 147 | 374 | 1,066 |

After extra fitting, seeds 0/1 finish in 2,958/2,987 additional actions; originals do not finish in 10k. Seed 2 improves but remains incomplete. Original top-100 repeat fractions 96/92/96% become 0/2/93%. Aggregate prediction MSE and whole-maze average noise ratio do not consistently improve alongside ranking: seed 0's normalized replay MSE slightly increases despite strong search improvement. Extra fitting changes geometry, not just noise sensitivity.

At the final 100k seed-2 archive, frozen original completes in 3,846 additional actions; fitting the same replay first completes in **489**. A known-noise-removal control completes in 584. Thus the missing states were not irretrievable. Zero-noise branches are explicitly privileged: set eight known nuisance input dimensions to zero for encoder evaluation only; preserve actual noisy replay/physics and freeze weights. They supply exact nuisance invariance and are not learned-agent improvements. At the 50k archives, those controls finish in 3,016 / 2,917 / 5,772 additional actions.

### Online extra-compute timing

Adaptive tests selected after initial repeat/frontier audit. Both add exactly 2,000 shared replay updates to the existing frequent-training persistent algorithm, with unchanged loss. Early arm spends all extra updates after 5k actions. Distributed arm adds 100 after each 5k actions. At 100k each has 6,112 updates and 1,562,000 transition presentations per head. Initial parameters, first 5k experience, and all successor-noise draws match the original baseline. Extra replay sampling uses replacement; original causal full epochs remain unchanged.

| Schedule | Actions to complete, seeds 0 / 1 / 2 | Final observed states |
|---|---|---|
| Original frequent baseline (reused) | 68,628 / 77,266 / >100k | 1,784 / 1,784 / 1,741 |
| +2,000 updates at 5k actions | 75,613 / 77,931 / 51,844 | 1,784 / 1,784 / 1,784 |
| +100 updates each 5k actions | 25,438 / >100k / 95,213 | 1,784 / 1,521 / 1,784 |

Early extra training completes all three seeds, but is slower than baseline in seeds 0/1; benefit concentrated in seed 2. Distributed extra training has a dramatic seed-0 gain and a seed-1 regression. Do not infer a universal schedule from the best seed or treat censored 100k times as observed completion. More optimization changes representation and future data collection; benefits are not monotonic. No claim of statistical significance or broad reliability from one maze/three seeds.

Artifacts: [full report](docs/frontier-efficiency.md), `docs/assets/frontier-efficiency-online.png`, `frontier-efficiency-diagnostics.png`, `frontier-efficiency.gif`, `frontier-efficiency-final.png`, and combined `frontier-efficiency-results.json`. Reports include all failures as well as gains. Roots: `runs/frontier-efficiency-audit-20261110`, `frontier-optimization-20261110`, `frontier-fitting-branches-20261110`, `frontier-zero-noise-oracle-20261110`. Scripts: `audit_frontier_efficiency.py`, `run_frontier_optimization.py`, `branch_frontier_fitting.py`, `report_frontier_efficiency.py`. Persistent runner gained optional extra-update placement arguments with zero default; original behavior unchanged by default. Audit JSON's original `wasted_actions` field denotes repeat actions without physical discovery, not absence of learning; updated code uses `repeat_actions` wording.

Validation: **64 tests pass**. Paired input/initialization/noise/budget checks passed; fixed-archive membership reconstructed and checked against final saved archives. All 18 scientific runs completed without failures. A plotting font-size API error was fixed and rendering rerun; it did not affect experiment data. Figures and animation inspected, git diff --check clean. No background runs remain; prior uncommitted work preserved.

## 2026-11-10 — Queue density and top-k batching diagnostics (completed)

User asked whether densities of useful versus repeated/noisy states down the queue suggest batching top-k expansion. Audited 28 model/archive snapshots: original 1,000/100-expansion cadences × three seeds × 5k/25k/50k/100k checkpoints, plus four fixed-archive +2,000-update models. Distinguish previously expanded positions, not-yet-expanded positions with no immediate new neighbors, and productive queued positions that would reveal an unobserved neighbor. All queued entries are already observed; truly unobserved physical positions are not directly in the queue. Oracle physical labels/adjacency used only for evaluation.

Original frequent-training 50k queues: productive fractions across whole queue 0.66/0.64/0.42%, versus top-100 3/6/1%; repeat fractions at head 96/92/96%. Thus head is enriched ~4.5×/9.3×/2.4× despite noise/repeat dominance. Useful entries do not form a universally clean dense band just below the top: ranks 101–1,000 have productive densities 2.44/4.11/1.33%, ranks 1,001–10,000 2.43/1.73/0.74%, ranks 10,001+ 0.07/0.19/0.29%. At seed 2's 100k checkpoint, top100 productive density10% versus0.109% overall; those10 productive entries cover only4 distinct productive positions and yield5 distinct new neighbors. Entry counts overstate unique information.

Counterfactuals on eight original/fitted snapshots (three seeds at50k, seed2 at100k): fixed top-k list versus greedy latent-diverse batch (update keys using selected latents, no children until batch ends) versus existing sequential expansion (immediate key updates and child insertion). Fixed and diverse batch yields computed exactly from deterministic adjacency, counting successor overlap once. Sequential runs use actual noisy observations, frozen models, three noise RNG replicates per snapshot,128 expansions each. Total24 short diagnostic rollouts/15,360 action executions; no training. RNG replicates are not additional trained seeds.

At128 expansions/640 actions from original50k snapshots, new physical states:

| Seed | Fixed top128 | Diverse batch128 | Sequential128, noise-replicate range |
|---|---:|---:|---:|
| 0 | 5 | 2 | 7–13 |
| 1 | 6 | 9 | 8–10 |
| 2 | 3 | 4 | 43–48 |

Sequential child insertion lets search follow newly exposed branches immediately; preselected batches cannot. In fitted50k models, fixed top128 yields51/66/12, diverse batches70/79/11, sequential152–164/175–184/68–115. No evidence here for raw top-k batching improving interaction efficiency. Parallel simulator throughput was not measured; diversity-aware batching remains a possible wall-clock compromise. Existing frequent training already does100 sequential expansions between neural updates; preselecting the batch is the distinct intervention. Not a full online batched-policy study.

Root `runs/queue-density-20261110`; scripts `audit_queue_density.py`, `report_queue_density.py`; [report](docs/queue-density.md), `docs/assets/queue-density-bands.png`, `queue-batching.png`, and `queue-density-results.json`. Audits and figures completed; images inspected; no changes to training or deployed planner. Previous runs preserved, no background work remains.


## 2026-11-10 — Learned-prior KL compression sweep (completed)

Authorized exploration of β and justified prior/training variants. **40 new full runs** plus two reused deterministic reference seeds: 14 conditions × three seeds. Every new run completed 100,000 real actions, 20,000 expansions, 4,112 joint gradient updates, and 1,050,000 transition presentations per dynamics head. Total new scientific budget: **4,000,000 actions and 164,480 updates**. A separate 5k-action/30-update smoke test is excluded. Same maze seed 7, 1,784 reachable positions, clean normalized coordinates plus eight fresh Gaussian nuisance inputs, persistent distance frontier, exact simulator restoration, and no physical-state deduplication. No preexisting run overwritten.

Added a stochastic encoder head q(z|o)=Normal(μ(o),diag σ²(o)), initialized σ=0.01. Joint dynamics training receives reparameterized sampled inputs, with the same input sample/batch across three heads; prediction targets and repulsion use deterministic means. Loss is coordinate-mean prediction MSE + 0.01 hinge repulsion + β E KL(q||p). KL sums over eight latent dimensions and averages observations; its scale is nats/observation. Both prediction endpoints and both repulsion endpoints retain encoder gradients. All queue priorities and agent inference still use μ(o), without sampling. Width/depth, encoder/dynamics learning rates, and replay exposure are unchanged. Training every 100 expansions means every **500 actions**; each event trains a portion of the accumulated replay epoch, completing one pass per 1,000 expansions/5,000 actions. No extra replay updates in this study.

Single diagonal Gaussian prior uses analytic KL. Four-component learned Gaussian mixture uses exact posterior entropy and four samples for expected log prior; diagnostics use 32 samples. Means, scales, and mixture logits are learned. β=0 leaves the prior inactive; its logged KL to the initial reference is not a fitted information estimate. Separate sampling generators preserve mean/dynamics initialization and environment-noise controls. Checkpoints save variance/prior parameters, optimizers, and sampling RNG states.

| Condition | Full coverage, seeds 0 / 1 / 2 (actions) | Mean final coverage | Mean coverage over 100k actions |
|---|---|---:|---:|
| Original deterministic | 68,628 / 77,266 / >100k | 99.2% | 55.8% |
| Gaussian β=0 | 42,498 / 86,192 / 91,846 | 100% | 54.6% |
| Gaussian β=1e-6 | 52,472 / 69,202 / 74,798 | 100% | 60.3% |
| Gaussian β=1e-5 | >100k / 55,286 / 56,314 | 91.6% | 57.7% |
| Gaussian β=1e-4 / 1e-3 / 1e-2 | None in all nine runs | 24.6% / 26.0% / 22.9% | 16.8% / 17.5% / 15.0% |
| Mixture-4 β=1e-6 | 85,623 / 72,702 / 93,147 | 100% | 52.8% |
| Mixture-4 β=1e-5 | 54,427 / >100k / >100k | 68.4% | 44.8% |
| Mixture-4 β=1e-4 | None | 25.3% | 18.6% |
| Gaussian, 10× prior LR, β=1e-5 / 1e-4 | None in all six runs | 34.2% / 23.9% | 25.3% / 17.0% |
| Gaussian β=1e-4, delayed to 25k | None | 27.2% | 20.4% |
| Gaussian β=1e-4, delayed to 35k | 42,504 / >100k / >100k | 69.4% | 46.8% |

Best positive setting by post-sweep mean coverage AUC: Gaussian β=1e-6. Mean completion cost 65,491 actions versus stochastic β=0's 73,512: **10.9% fewer**, improving seeds 1/2 and slowing seed 0. Three seeds on one maze, selected after the sweep: exploratory, not held-out validation or optimal-β evidence. β=0 is the matched stochastic baseline; its mean AUC is slightly below the original deterministic agent despite completing all three seeds. AUC integrates the common 250-action logging grid, without assigning completion times to censored runs.

Mechanism: β=0 posterior std shrinks from 0.01 to 0.000629–0.000703, supporting suppression of sampling noise but not equivalence to deterministic training. β=1e-6 does **not** consistently reduce observation-noise sensitivity of means: final same-position-noise / adjacent-position distance ratios 0.111/0.145/0.142 versus β=0's 0.115/0.138/0.143. Thus exploration improves on average without evidence that nuisance information was wiped from mean latents. Strong from-scratch β≥1e-4 gives mean variance around 1e-7, near-maximal repulsion, near-zero KL, and poor coverage: a collapse failure. Mean-only queue inference can exploit distinctions inaccessible to sampled-input dynamics; an information penalty on sampled z need not erase them from μ.

Adaptive decisions preserved: faster prior LR tested a lagging-prior hypothesis and failed; 25k delayed KL was too early for strong invariance. At 35k all β=0 control probes showed noise/movement ratios 0.20–0.30, so delayed KL was repeated there. Both delayed arms initialize the Gaussian prior to current replay posterior moments at activation (a combined delay/initialization intervention). No extra updates; exact β=0 data prefix through the switch, at 296/651 updates, with KL first affecting the next event at 25.5k/35.5k actions. Later activation preserves variation but two seeds still stall. Mixture flexibility, faster fitting, and delayed compression do not beat the small Gaussian setting in this sweep.

Run roots: `runs/kl-gaussian-20261110` (19), `kl-mixture-20261110` (9), `kl-fast-prior-20261110` (6), `kl-warm-start-20261110` (3), `kl-late-start-20261110` (3). Reused original deterministic seeds 1/2; fresh deterministic seed 0 exactly reproduces its previous replay. Runner `scripts/run_kl_bottleneck.py`; implementation `src/map_learning/information_bottleneck.py`; optional persistent-runner flags preserve deterministic defaults. Audit/report: `scripts/audit_kl_bottleneck.py`, `scripts/report_kl_bottleneck.py`. Full specifications, source snapshots, replay, initial/intermediate/final checkpoints, frontiers, and animation frames retained per run.

Validation: **70 tests pass**. Independent audits passed all 40 runs: initialization, pre-training prefixes, complete matched successor-noise streams, update/action/exposure budgets, delayed-control prefixes, deterministic inference RNG, and every final frontier priority recomputed from raw observations. Checkpoints at 25k/50k/100k include queue-rank diagnostics; these compare different online archives and are not identical-data causal comparisons. Checkpoint loading reproduces the next stochastic update exactly and remains compatible with old deterministic models.

Artifacts: [complete report and reproduction commands](docs/kl-bottleneck.md), `docs/assets/kl-bottleneck-sweep.png`, `kl-bottleneck-comparison.png`, `kl-bottleneck.gif`, `kl-bottleneck-final.png`, `kl-bottleneck-audit.json`, `kl-bottleneck-summary.json`. Animation compares deterministic / stochastic β=0 / Gaussian β=1e-6 across all three seeds, through the full 100k actions; red markers show restoration expansions, not a continuous navigation trajectory. Plots and animation inspected. All experiments and audits finished; no background jobs remain. Prior uncommitted work preserved.


## 2026-11-10 — Joint KL × repulsion sweep and fresh-seed confirmation (completed)

User requested joint tuning after identifying fixed repulsion as a limitation of the preceding KL study. **54 new runs + 15 reused controls = 69 comparison rows**. New scientific budget: 5,400,000 real actions and 222,048 joint updates. All runs received 100k actions, 20k expansions, 4,112 updates, and 1,050,000 transition presentations/head. All new jobs exited successfully; prior outputs were preserved.

Primary grid: repulsion α=0.01/0.1/1 × β=0/1e-6/1e-5/1e-4/1e-3, seeds 0/1/2. Reused the 15 exactly matching original Gaussian α=0.01 runs and ran 30 new ones. Added an adaptive intermediate α=0.03 screen with β=0/1e-5/3e-5/1e-4 (12 runs), prompted by poor high-repulsion β=0 performance and a partial rescue at α=0.1, β=1e-4. Selected α=0.03, β=1e-5 after it completed all three screening seeds faster than the previous best. Fixed four arms before launching fresh seeds 3/4/5: (α,β)=(0.01,0), (0.01,1e-6), (0.03,0), (0.03,1e-5), adding 12 confirmation runs. Selection occurred while the remaining primary grid was running; no changes were made based on confirmation results.

| Setting | Full coverage actions, screening seeds 0 / 1 / 2 | Full coverage actions, confirmation seeds 3 / 4 / 5 |
|---|---|---|
| α=0.01, β=0 | 42,498 / 86,192 / 91,846 | >100k / >100k / 44,572 |
| Previous best α=0.01, β=1e-6 | 52,472 / 69,202 / 74,798 | 88,667 / 90,208 / 83,031 |
| Higher repulsion only α=0.03, β=0 | >100k / 94,554 / >100k | >100k / 58,813 / >100k |
| **Joint α=0.03, β=1e-5** | **39,513 / 46,292 / 69,533** | **69,768 / 67,992 / 51,028** |

Confirmation mean completion cost improves from 87,302 to 62,929 actions (27.9% fewer) versus the previous best; mean coverage over the 100k run rises from 47.2% to 56.7%. Improvement holds in each of the three fresh seeds and each screening seed. Six-seed means: 76,396 versus 57,354 actions (24.9% fewer), but the fresh-seed comparison is the cleaner evidence. Joint setting completes 6/6, original stochastic β=0 4/6, increased-repulsion-only 2/6. Individual β=0 runs can still be faster; no uniform dominance or universal optimality claim. Censored runs are not treated as observed 100k completion times.

Two distinct failure modes: α=0.01, β=1e-4 collapses all three screening representations (24.6% mean coverage); α=0.1 at that β rescues two seeds (62,409/60,661 actions) but the third stalls at 629/1,784. At α=1, every tested β gives poor coverage (~5–6%), with substantial latent spread and low repulsion. The β=0 α=1 mean nuisance/movement ratio is ~444 and normalized MSE ~0.853: nuisance separation dominates useful dynamics rather than constant collapse. Full failed-arm results retained.

The successful joint setting does not show improved mean-latent nuisance invariance: confirmation noise/movement ratio 0.160 versus previous-best 0.137; normalized mean-input MSE 0.000415 versus 0.000336, sampled-input 0.000425 versus 0.000352. Both fit their own data well by these probes, but these different online replay distributions are not identical-data causal comparisons. Better exploration does not establish that compression removed more nuisance noise. Mean-only queue inference can still use distinctions that sampled-input dynamics cannot reliably access.

Loss remains prediction coordinate-mean MSE + α hinge repulsion + β KL, with KL summed over eight dimensions and averaged observations. Same learned diagonal Gaussian prior, σ initialization 0.01, architecture, optimizers, seeds/noise controls, and persistent nearest-visited-mean simulator-restoration search. No additional optimization, physical-ID merging, or planning changes. Training every 100 expansions/**500 actions** apportions one full accumulated-replay epoch across every 1,000 expansions/5,000 actions. All runs continue to 100k even after reaching coverage.

Implementation: optional validated `--repulsion-weight` in `scripts/persistent_frontier.py`, with unchanged defaults; new `scripts/run_kl_repulsion.py` supports grids and explicit α:β pairs, records reused/new outputs in manifests, refuses overwrites. Roots: `runs/kl-repulsion-20261110` (30 new), `kl-repulsion-intermediate-20261110` (12), `kl-repulsion-confirm-20261110` (12). Scripts `audit_kl_repulsion.py`, `report_kl_repulsion.py`; original mean/dynamics behavior unchanged.

Validation: **70 tests pass**. All 69 independent audits passed: coefficient/config checks, action/update/sample budgets, matched initial mean/variance/dynamics parameters, 500-action pre-training prefixes, full successor-noise streams, deterministic mean inference RNG, and exact final frontier priorities reconstructed from raw observations. 50k/100k rank diagnostics retained. Audit resume only reuses verified immutable completed runs. All runs/artifacts remain on disk; no background experiments remain.

Artifacts: [report and reproduction commands](docs/kl-repulsion.md), [all condition tables](docs/kl-repulsion-tables.md), `docs/assets/kl-repulsion-grid.png`, `kl-repulsion-screen.png`, `kl-repulsion-confirm.png`, `kl-repulsion.gif`, `kl-repulsion-final.png`, `kl-repulsion-summary.json`, `kl-repulsion-audit.json`. Animation compares all four confirmation arms on seeds 3/4/5 through 100k actions; simulator restoration jumps are not continuous trajectories. One maze layout; continuous positions and other layouts are not tested. Candidate for subsequent experiments: α=0.03, β=1e-5; repository defaults unchanged.

The primary grid, confirmation curves, and animation frames were visually inspected; 100-frame animation verified. Final git diff --check is clean.


## 2026-11-10 — KL and variance checkpoint diagnostics (no new training)

Read all 69 joint-sweep final checkpoints and their existing 5k-action block logs. Compared training-time posterior noise, deterministic mean spread, observation nuisance sensitivity, and raw/weighted KL. On confirmation seeds, previous-best versus joint-winner mean posterior σ is 0.00218 versus 0.0385, Eσ² 4.90e-6 versus 0.00232, and mean spread 0.696 versus 0.686. Winner has 5/2/6 coordinates with Var μ below Eσ², versus none for the previous best. This indicates a stronger stochastic bottleneck, not proof of nuisance removal or an exact informative dimension count.

Additional read-only moment fit of a diagonal Gaussian to each empirical aggregate posterior (2,048 identical isolated replay probes per checkpoint) separates learned-prior KL into the best diagonal-Gaussian bound plus KL from that moment fit to the learned prior. Float64 identity checks passed all 69 cases; reconstructed raw KL matches logged float32 values. Confirmation previous best: 186.7 = 42.9 + 143.8 nats; joint winner: 100.3 = 15.8 + 84.5. Most raw KL reflects avoidable prior mismatch on this empirical probe; the remaining bound is still not mutual information. No prior/encoder weights were changed and no new exploration or optimization was run.

Script `scripts/analyze_kl_variance.py`; [report](docs/kl-variance-diagnostics.md); `docs/assets/kl-variance-diagnostics.json`, `kl-variance-trajectories.png`, `kl-coordinate-variance.png`. All original experiments preserved. Figures inspected; numeric identity and logged-value checks passed.


## 2026-11-10 — Moment-fitted Gaussian prior sweep and follow-ups (completed)

User authorized analytically setting prior moments and retuning β and repulsion. **82 new 100k-action runs**, including one β=0 equivalence reproduction: **8,200,000 actions and 337,184 updates**. Nine matching β=0 screening controls plus three previously audited learned-prior reference runs reused; separate 5k-action/30-update smoke run excluded. Same persistent noisy-maze frontier, exact simulator restoration, 1,784 positions, eight Gaussian nuisance inputs, deterministic mean-latent priorities. Every full run used 4,112 updates and 1,050,000 transition presentations per head. Training every 100 expansions = 500 actions, apportioned accumulated epoch completed each 5k actions; no extra gradient budget.

Added `prior_mode=replay_moments`: for each coordinate fit mean E μ and variance Var μ + E σ² to all current replay source/successor observations, retaining multiplicity. Float64 central moments, detached prior, no artificial prior scale ceiling. Fit before each training event, hold during that event, and refit after each block for checkpoints. Added `batch_moments`: fit the same moments on the 512 current KL/repulsion samples before every update. Default `learned` behavior and earlier checkpoints preserved. Same joint sampled-input MSE, mean targets, mean hinge repulsion, and deterministic agent inference. Each replay-fit run performs 220 fits / 22.2m additional endpoint encodings; extra forward cost is logged, not extra learning updates.

Primary α={.01,.03,.1} × β={0,1e-6,1e-5,1e-4,1e-3,1e-2}, seeds 0/1/2: 45 positive-β runs, nine reused zeros, one full zero reproduction. Adaptive follow-ups: batch fitting at (.01,1e-6)/(.03,1e-5), six runs; lower/intermediate β={1e-7,3e-6}, α={.01,.03}, twelve runs; α=.03 with β={1e-5,1e-4,1e-3} delayed until after 651 updates/35k actions, nine runs. Selection by maximum three-seed coverage AUC among positive-β fitted variants, recorded before candidate confirmation. Selected replay fit α=.01, β=1e-7: screening full coverage 70,188 / 65,137 / 97,758 actions, mean coverage 53.1%, already below β=0's 54.6% and learned winner's 62.0%.

Nine new confirmation runs used fixed candidate, learned winner, and stochastic β=0 on new seeds 6/7/8:

| Condition | Full coverage actions, seeds 6 / 7 / 8 | Final coverage | Mean coverage over 100k |
|---|---|---|---:|
| Learned α=.03, β=1e-5 | 50,307 / 46,906 / 50,292 | 100 / 100 / 100% | 66.5% |
| β=0, α=.01 | 82,943 / >100k / 88,477 | 100 / 91.5 / 100% | 50.2% |
| Replay fit α=.01, β=1e-7 | >100k / >100k / >100k | 64.3 / 93.1 / 87.6% | 37.0% |

No improvement; retain the previous learned-prior winner as preferred experimental configuration. This is a three-seed confirmation on one maze, not proof against all fitted priors. Candidate beats β=0 on seed 7 but loses on seeds 6/8. Noise/movement ratio is lower (.0835 versus .1275 β=0 and .1444 learned), yet queue exploration is worse; average nuisance suppression alone is insufficient. Normalized dynamics MSE is on each run's own replay and representation, not physical prediction accuracy.

At the old winning α=.03, β=1e-5, fitting from the start collapsed (mean variance 2.4e-8, 22.3% final coverage). Delaying KL to 35k preserves spread and reaches 85.5% final coverage, one full completion. Stronger delayed β=1e-4/1e-3 lowers fitted KL to 7.82/4.37 nats and increases posterior variance to .720/1.080 while mean variance remains 4.00/3.17; final coverage falls to 54.3/45.7%. No completion in those six runs. Increasing repulsion to .1 shifts the collapse boundary but produces no primary-grid completions.

Fitting cadence follow-up: replay-event α=.01, β=1e-6 last-batch prior gap typically ~.01 nat, occasionally 12–25; this mixes update lag and sampling mismatch. Per-update batch fit raises mean coverage 44.8→51.9% and completion count 2→3, still below β=0; it also collapses at α=.03, β=1e-5. It changes cadence and estimator together. Initial fitted prior standard deviations .0133–.0251 versus .05 learned imply 4–14× inverse-variance coefficients in the KL mean gradient; an optimization clue, not a causal isolation. Sampled-latent information bounds do not directly constrain the deterministic queue means.

Validation: **75 tests pass**. All **91 manifest entries** passed independent audits: initialization, initial data/noise streams, budgets, deterministic mean inference, checkpoint frontier reconstruction/final priority checks, and full-replay moments. Largest absolute fitted-prior mismatch at 50k/100k refitted checkpoints 5.3e-12 nats. Full β=0 reproduction exactly matches original replay and final encoder/dynamics weights; delayed runs exactly match their 35k β=0 prefixes. Independent moment estimates use a separate sum/second-moment method. Full-replay fitted KL still includes non-Gaussian aggregate mismatch and is not true mutual information.

Roots: `runs/moment-prior-20261110`, `moment-prior-batch-20261110`, `moment-prior-fine-20261110`, `moment-prior-delayed-20261110`, `moment-prior-delayed-strong-20261110`, `moment-prior-confirm-controls-20261110`, `moment-prior-confirm-candidate-20261110`; combined confirmation manifest `runs/moment-prior-confirmation-20261110`. Scripts: `run_moment_prior.py`, `audit_moment_prior.py`, `report_moment_prior.py`, `report_moment_dynamics.py`; one-off completed orchestration `finish_moment_prior.py`.

[Report and reproduction commands](docs/moment-prior.md), [complete tables](docs/moment-prior-tables.md), and `docs/assets/moment-prior-{grid,screen,confirm,dynamics,final}.png`, `moment-prior.gif`, `moment-prior-{summary,audit,selection,initial-scale}.json`. Animation has 100 frames through the full 100k budget with all three confirmation arms/seeds; maps and curves use consistent condition colors. Figures/frames inspected; all runs finished; preexisting uncommitted work preserved.


## 2026-11-10 — KL mechanism ablations and causal controls (completed)

Authorized to distinguish sampling noise, KL's representation effects, prior lag, and exploration/data feedback. **54 shared-data fits +27 predeclared online runs +9 fresh confirmation runs =90 full fits**, each4,112 joint updates and1,050,000 transition presentations/head:370,080 joint updates total. Online training collected3,600,000 actions. Shared-data training reuses100k recorded transitions per fit, collecting no training actions.324 separate frozen continuations ×640 actions =207,360 evaluation actions. A5k replay-only exact-default reproduction is excluded. No existing run overwritten.

Fixed α=.03 and β=0 or1e-5. Nine arms: deterministic means/no KL, prescribed noise/no KL, deterministic means/KL, prescribed noise/KL, adaptive posterior/no KL, original adaptive posterior/KL winner, and winner with prior LR×3, LR×10, or five prior steps. Prescribed isotropic RMS noise schedule fixed from the previous winner's seeds0/1/2; no new-run outcomes used. Shared data uses previous winner and β=0 replay families, source seeds6/7/8, paired fresh model seeds9/10/11. All arms within a source/seed receive identical observations, transition batches, repulsion draws, initial networks and update budgets. Extra prior steps fit the same detached batch without extra encoder/dynamics updates or new data. Queue inference always uses deterministic means.

Main finding: **sampling during dynamics training is unnecessary for the KL exploration gain in this maze**. On initial online seeds9/10/11, no-sampling KL completes56,036 /53,514 /46,698 actions, versus deterministic no-KL >100k /79,067 />100k. Original winner completes55,671 /53,631 /54,328. Prescribed noise alone completes only seed11 (76,516); prescribed noise+KL completes all (60,313 /49,573 /42,403). Mean coverage over100k:44.5% no-sampling/no-KL,36.7% fixed-noise/no-KL,64.6% no-sampling/KL,63.7% fixed-noise/KL,39.7% adaptive-noise/no-KL,60.7% original winner.

Adaptive fresh confirmation was fixed after the initial no-sampling result, with unchanged coefficients:

| Arm | Completion actions, seeds12 /13 /14 | Mean coverage over100k |
|---|---|---:|
| Deterministic means, no KL |98,542 />100k />100k |36.3% |
| Deterministic means, KL |52,379 /54,613 /51,002 |63.1% |
| Original sampled-input winner |56,139 /50,378 /61,592 |60.5% |

No-sampling KL completes6/6 seeds versus deterministic no-KL2/6, faster in every matched seed. It is comparable to the original winner (mean completion52,374 versus55,290;4/6 faster), not universally better. In this ablation posterior variance/prior still train through KL but dynamics consumes no samples; therefore the sampled-information-bottleneck explanation is not necessary. KL's direct mean penalty and variance-branch gradients remain possible mechanisms.

Shared-data/queue controls rule out explaining the full gain solely by better collected experience. At50k on common β=0 archives, frozen640-action continuations discover28.8 new positions for mean/no-KL,27.1 fixed-noise/no-KL,136.1 mean/KL,137.4 fixed-noise/KL,131.3 original winner (averages across three model/source pairs, each three noise replicates). Top100 productive-entry fractions7.0%,12.0%,58.3%,56.3%,53.7%. Every mean-KL paired seed beats mean/no-KL here. At35k on winner archives effects are small/mixed; on β=0 archives mean-KL98.6 versus57.1; at75k mean-KL139.7 versus66.2 with one paired reversal. Productive means an observed queued position whose expansion exposes an unobserved neighbor. No privileged information enters learning or queue keys. Replicates/checkpoints/families are not counted as independent training seeds.

Prior adaptation: on common winner data the final full-replay mismatch falls81.95→12.55→1.39→0.08 nats (original, LR×3, LR×10, five steps). Online gaps82.01/14.37/.95/.16. Online mean coverage60.7/60.7/42.1/49.1%. LR×3 finishes86,037 /41,257 /54,018: mixed paired effects. LR×10 only one completion; five steps two. Better prior fitting does not consistently improve exploration at fixed β; this does not rule out retuning β after changing adaptation speed.

Noise/geometry audits: final common β=0-data models have median noise/local-movement ratio.104 without KL versus.124 for mean/KL,99th percentile.408 versus.459. Mean-KL's stronger queue ranking is not explained by a smaller average nuisance metric. Centroid covariance participation ratios1.53 versus1.94 are descriptive; specific useful geometry is unresolved. A uniform latent rescaling alone cannot change distance-priority order.

Implemented backward-compatible `dynamics_input={posterior,mean,scheduled}`, serialized noise schedules, prior-only step counts, and accurate prior-step diagnostics including old-checkpoint defaults. Added tests for fixed noise independence from variance head, mean-input independence from posterior scale, isolated prior updates, and exact stochastic continuation. **79 tests pass**. Full default5k encoder/dynamics/prior and RNG equivalence with old winner passed. Independent54-fit audit verifies data/batch/model hashes and RNG matching, budgets, common archives and reconstructed rankings. Independent36-run online audit verifies initialization, initial data prefix, full successor-noise streams, RNG states, budgets and exact final frontier priorities. Later continuation checks verify membership, budgets and discovery bounds. Full prior moments recomputed independently for all final online checkpoints. All plots and100-frame animation inspected; git diff --check clean.

[Report](docs/kl-mechanisms.md), [protocol/adaptive log](docs/kl-mechanisms-protocol.md), [all tables](docs/kl-mechanisms-tables.md), [later shared-archive tables](docs/kl-mechanisms-late-tables.md). Assets `docs/assets/kl-mechanisms-{fixed,late,online,confirm,trajectories,final}.png`, `kl-mechanisms.gif`, `kl-mechanisms-summary.json`, late summary and four audit JSONs. Roots `runs/kl-mechanisms-{fixed,online,confirm,late,control}-20261110`. Scripts `kl_mechanisms.py`, `run_kl_mechanisms.py`, `train_kl_mechanism_fixed.py`, `audit_kl_mechanisms.py`, `late_kl_mechanisms.py`, `report_kl_mechanisms.py`, `report_late_kl_mechanisms.py`. All runs finished; no background study work remains. Defaults and preexisting uncommitted work preserved.


## 2026-11-10 — Latent L2 sweep and observation-noise/search diagnostics (completed)

User authorized latent L2 experiments and investigation of noise effects, emphasizing that resampling nuisance is unavailable in general. All training uses ordinary noisy replay; nuisance redraws, known-noise removal, and physical identity labels are evaluation-only. **36 scientific online runs + one full zero-control reproduction + three shared-data fits = 40 new full-budget fits**, each 4,112 joint updates and 1,050,000 transition presentations/head: 164,480 updates total. Six old online controls reused. Online training/reproduction collects 3,700,000 actions; frozen evaluations collect 178,560, for 3,878,560 new actions total. Shared-data fits collect no training experience.

Loss: coordinate-mean joint dynamics MSE + α=.03 C-SWM hinge repulsion + λ E||μ(o)||². L2 sums eight coordinates and averages the same 512 replay states used by repulsion, without another data draw. Mean inputs, β=0 in L2 arms; comparator is learned-Gaussian-prior KL β=1e-5 with mean inputs. Width 256, three hidden layers, three dynamics heads, shared minibatches. Persistent raw-observation frontier/visited set and exact mean-distance priorities. Training every 100 expansions / 500 actions, apportioning one full replay epoch over each 5,000-action block; 100k actions per run. No new full epoch every 500 actions, no exact-state deduplication, no nuisance supervision.

Screen λ={0, 1e-5, 3e-5, 1e-4, 3e-4, .001, .003, .01, .03, .1}, seeds 9/10/11, with matched KL controls. Selected λ=.03 by mean coverage AUC before fresh seeds 15/16/17. Screen AUC 69.6% versus 64.6% KL and 44.5% zero; selected completions 43,247 / 46,002 / 43,569. λ=.1 collapses, final mean coverage 24.3%; very weak λ=1e-5 does not reliably help. All failed arms retained.

| Arm | Completion actions, seeds 15 / 16 / 17 | Mean coverage over 100k | Final mean coverage |
|---|---|---:|---:|
| No regularizer | >100k / >100k / >100k | 37.8% | 85.6% |
| Mean-input KL | 53,929 / 52,243 / 55,568 | 63.3% | 100% |
| **L2 λ=.03** | **50,856 / 36,289 / 51,397** | **68.1%** | **100%** |

Fresh mean completion cost 46,181 versus 53,913 for KL, **14.3% fewer actions**; every paired seed improves, with most of the average advantage coming from seed 16. Selected L2 beats KL on all six screen/confirmation pairs, but only the last three are independent of selection. One maze layout, one nuisance distribution; no continuous-position or reset-free claim.

Shared-data follow-up fits selected L2 on the old β=0 source replay seeds 6/7/8, model seeds 9/10/11, with exactly matching initial networks and transition batches. On the same 50k archives, top-100 productive fractions are **7.0% / 58.3% / 73.7%** for zero/KL/L2; already-expanded fractions 80.7% / 18.7% / 4.0%. Frozen 640-action continuation yields 25.8 / 140.2 / 154.9 new physical positions. At 75k yields are 61.2 / 146.0 / 127.2: L2 does not universally dominate KL at fixed checkpoints. Three noise replicates per model are averaged, not counted as trained seeds. Productive means expansion exposes an unobserved neighbor.

Noise redraw diagnostics hold physical archive membership fixed. After independently redrawing queue and anchor nuisance at 50k, top-100 entry retention is 11.1% / 43.2% / 42.1%, physical-position Jaccard .154 / .385 / .442. This is physical selection churn, not merely duplicate entry exchange. Redrawing both does not rescue baseline continuation yield (25.8→14.7), while L2 is stable (154.9→156.4). Redraws also remove original selection bias; they are not a deployable method or a pure variance-only intervention.

Privileged zeroing of nuisance in all stored/future observations raises baseline discovery 25.8→142.0 at 50k, KL 140.2→186.3, but changes L2 154.9→143.7; at 75k all three improve. It restores exact same-position encoding by construction and is not a guaranteed upper bound. Clipping latent radii at the visited 95th percentile barely changes baseline (25.8→27.2; 61.2→59.7 at 75k). Baseline top-100 norms average the 51st percentile, centered radii the 43rd, whereas nuisance residuals average the 68th percentile. Evidence favors a local distance/ranking issue over a simple large-global-norm explanation.

L2 improves queue ranking despite **worse** mean noise/movement ratio: common-data 50k .137 / .260 / .400 for zero/KL/L2. Absolute nuisance displacement falls .0325→.0126, but moving-neighbor distance contracts .2382→.0314. Final fresh-online ratio .114 / .161 / .378. Conditional nearest-memory physical identity at 50k is 67.8% / 90.8% / 90.7%. Relative geometry matters; average ratio, latent normalized MSE, and global shrinkage are insufficient explanations. Uniform rescaling cannot change queue ordering. Specific causal geometry remains unresolved.

Analytical check: for independent Z,Z′, E[1−||Z−Z′||²]+ ≥1−2 tr Cov(Z). Thus loss ≥α+(λ−2α) tr Cov(Z)+λ||EZ||², ignoring nonnegative dynamics MSE. Constant zero encoder/dynamics attains α. At λ≥2α=.06, collapse is a global optimum of this expected objective; λ=.1 failure is therefore expected. The statement uses the exact sum-coordinate L2 convention, not a universal regularization threshold.

Validation: **80 tests pass**. Independent audit passes all 43 online manifest entries, 18 noise cases, and three shared fits: initialization, initial data prefix, paired complete successor-noise streams, budgets, final frontier reconstruction/priorities, shared-data/batch hashes. Full zero reproduction exactly matches original replay and final encoder/dynamics. CLI rejects silently ignored L2 settings; no experimental behavior changed. Figures and 100-frame animation inspected through the full 100k budget. Runs and audits completed; original defaults and prior work preserved.

[Report and reproduction commands](docs/l2-noise.md), [protocol](docs/l2-noise-protocol.md), [all tables](docs/l2-noise-tables.md). Assets `docs/assets/l2-noise-{sweep,confirm,interventions,stability,final}.png`, `l2-noise.gif`, `l2-noise-{summary,selection,audit,neighbors}.json`. Roots `runs/l2-noise-{screen,confirm,fixed,controls,selected}-20261110`. Scripts `run_l2_sweep.py`, `finish_l2_noise.py`, `noise_search_diagnostics.py`, `audit_l2_noise.py`, `audit_l2_neighbors.py`, `report_l2_noise.py`; reusable L2 support in bottleneck trainer, persistent runner, and fixed-replay runner.


## 2026-11-10 — Shared-replay geometry and deterministic formulation cleanup (completed)

User requested latent visualizations on identical replay and removal of unnecessary distribution terminology/structure from the current baseline/L2 formulation. Geometry analysis uses **18 existing checkpoints**: baseline/KL/L2 × model seeds 9/10/11 × 50k/75k, paired source seeds 6/7/8 from the old β=0 replay family. No new training or environment interactions for these analyses. Source streams, training minibatches, and initial networks were already matched in the preceding study.

Plot centroids of actual acquired encodings per observed physical position, counting each acquired observation once; no noise redraw or zeroing. Same colors denote same physical positions. Gray lines are oracle adjacency for display, not learned edges. Separate PCA per encoder, equally weighted positions, one uniform RMS normalization per panel, equal axis units, no whitening. Distances and directional tests use full 8D latents. First seed shown in the overview; all seeds and both checkpoints retained.

At 50k, baseline PC1 carries about 99.99% of spatial spread in every seed; median local strong/weak stretch ratios **155.9 / 358.8 / 198.6**. KL ratios **1.55 / 40.8 / 1.30**; L2 **1.36 / 1.63 / 1.54**. KL seed 10 opens into two substantial directions by 75k, while baseline remains narrow (ratios 21.6 / 308.9 / 85.3). Finite differences use open +x/+y transitions and ≥8 acquired observations at source/neighbors, with common qualifying positions. Projected within-position noise RMS divided by weak stretch has median **.916 / 1.480 / .849** for baseline, **.152 / .620 / .176** for KL, **.282 / .209 / .233** for L2 at 50k. This directional signal/noise measure exposes distinctions that the average step-length denominator hides. It is not numerically the same statistic as the old pairwise noise/movement ratio.

Near-one-dimensional is not constant collapse: weak distinctions remain. The measured geometry supports a mechanism in which regularization prevents useful spatial variation from being squeezed into a noise-sensitive direction. Repulsion/L2 encouraging more efficient spatial arrangement is an interpretation, not a proof of the entire online gain. Shared replay isolates representation effects; it does not locate the change on different online trajectories.

Static geometry intervention on all 18 queues: fit PCs only to visited latent observations and compare raw, uniform ×2, top two PCs, top two equalized, and all-eight whitening (fixed 1e-12 numerical floor). At 50k, mean top-100 productive fractions for baseline/KL/L2: raw **7.0/58.3/73.7%**; top two **3.3/59.0/73.3%**; equalized top two **18.3/66.0/73.3%**; all-eight whitening **.3/1.0/1.0%**. Baseline equalization is mixed per seed (14/2/39% versus 6/7/8% raw). Uniform scaling exactly preserves rankings. At 75k equalized means 33.7/66.3/69.0% versus raw 24/64/68%. Amplifying existing weak/noisy axes is not a general repair. No new online whitening policy was run or installed.

Implemented clean `train.loss=dynamics` with ordinary encoder, residual point-prediction heads, joint MSE + repulsion + optional `train.latent_l2`. Both encoder endpoints retain gradients. No encoder scale branch, prior, mixture logits, or predictive scales are built. New config `configs/deterministic_frontier.json`; persistent runner accepts `--config` and direct `--latent-l2` without KL flags. Replay schedules, checkpoint loading, MCTS/shooting disagreement, reward probes, and deterministic evaluation support the clean path. Historical KL/probabilistic paths remain reproducible; KL cannot be discarded equivalently merely because its dynamics inputs were deterministic.

Migration utility strips inactive outputs from legacy single-component deterministic MSE checkpoints, preserving weights, Adam state, update count, and replay RNG, and rejecting active KL/sampled inputs. Converted trained seed-15 L2 checkpoint: **exact encoder/prediction match on 2,048 observations and exact next encoder update/loss** in CPU audit. Native model counts: encoder 136,456; three dynamics heads together 411,672. Fresh smaller heads change initialization RNG consumption, so seeded online runs are not claimed to reproduce historical trajectories. Prior results remain tied to their recorded implementation.

New native-config integration run `runs/deterministic-cleanup-smoke-20261110`: **5,000 actions, 20 updates, 5,000 presentations/head**, not a full exploration benchmark. Audit applies one additional replay-only update to each of legacy/converted models in memory, without saving those updates over source checkpoints. No new sweep. **84 tests pass**, with final focused migration/bottleneck checks also passing after a validation guard. Tests cover Adam continuity, round-trip, rejected incompatible conversion, schedules, point-model search/diagnostics/evaluation. Geometry audit independently re-encodes priorities; all figures and sampled rotation frames inspected. Existing artifacts preserved.

[Geometry report](docs/l2-geometry.md), [deterministic formulation and commands](docs/deterministic-formulation.md). Assets `docs/assets/l2-geometry-{overview,50k,75k,local,queue}.png`, `l2-geometry-rotation.gif`, `l2-geometry-{results,transforms}.json`, `deterministic-conversion-audit.json`. Analysis arrays `runs/l2-geometry-20261110`; migrated checkpoint `runs/deterministic-conversion-20261110/l2-seed15.pt`. Scripts `analyze_l2_geometry.py`, `report_l2_geometry.py`, `audit_l2_geometry_transforms.py`, `convert_deterministic_checkpoint.py`.
