# Probabilistic dynamics exploration pilot — 2026-11-04

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

![Online coverage comparison](../runs/probabilistic-summary/coverage-comparison.png)

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

![Learning diagnostics](../runs/probabilistic-summary/learning-diagnostics.png)

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
