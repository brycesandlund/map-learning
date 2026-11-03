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
