# Probabilistic maze sweep — 2026-11-04

Completed **41 online runs (1.58 million real interactions)** and **two 50,000-update offline fits**. Twelve online runs reached **100,000 steps**, 33 times the original pilot budget; nine confirmation/follow-up runs used 20,000 steps, and twenty screening runs used 10,000.

**No learned configuration beat persistent random exploration in the mean matched-budget comparisons.** At 100k, mean position coverage was 99.4% for persistent random, 82.6% for uniform random, 74.2% for single-Gaussian MCTS, and 56.3% for width-128 MCTS. Single-Gaussian MCTS explored faster than uniform random early, but later stalled: its average coverage over the whole run was 63.1% versus uniform random's 61.5%, despite lower final coverage.

**Representation improvements did not automatically improve exploration.** Removing the encoder's tanh bound improved physical next-state prediction in the paired offline experiment and across the online follow-up seeds. Online 20k coverage nevertheless remained 45.7% versus 45.9%. This makes the disagreement reward and action-search strategy separate priorities for investigation.

## Protocol

All runs use normalized 2D coordinates in the deterministic 12×12 maze (layout seed 7; 1,784 reachable positions), 1,000 warmup actions, and center resets every 2,000 steps. Warmup uses persistent random except in the independent-action random control, which uses uniform actions throughout. No actor, codebook, state merging, reconstruction loss, or oracle guidance enters learning/planning. Models jointly train the encoder and Gaussian-mixture dynamics ensemble with NLL and pairwise repulsion.

The initial screen tests 15 learned configurations plus persistent random for 10,000 interactions on seed 0. Except where overridden, the model has 8 latents, two width-64 hidden layers, three ensemble members and three Gaussian components; Adam 3e-4, variance floor 1e-4, repulsion weight 10, and five updates every ten interactions. MCTS uses horizon 4, 24 simulations, discount .9, and epsilon .1. The horizon-8 variant also uses 48 simulations.

An adaptive follow-up adds latent dimension 4 and three variants with 64 Monte Carlo samples per member (default, latent-2, and horizon-1), motivated by an independent reward-noise probe. All four use the same seed-0 10k budget. This is a mostly one-factor screen, not an exhaustive search over combinations.

Fresh-seed confirmation uses model/behavior seeds 3, 4, and 5. Single-Gaussian and width-128 runs started while screening finished, based on their provisional ranking; their outcomes did not affect screening selection. Both run for 100,000 steps. Horizon-8 later overtook them in the screen. Because it expands about 300 nodes/action versus 60 for the default, its confirmation budget was adjusted to 20,000 steps. This explicitly changes the original plan to give both eventual finalists 100k. Width-128 is retained as an additional architecture control. All policies are compared fairly using their first 20k transitions; 100k results are presented separately. The maze layout is unchanged, so this tests seed robustness, not transfer to unseen mazes.

A separate adaptive activation diagnostic fits tanh and identity encoders from scratch for 50,000 updates each on the same stored data and whole-episode split. It is followed by a matched 20k online comparison of the default bounded and unbounded encoders on seeds 3–5. These six online follow-ups are excluded from the original screen ranking. They reuse seeds 3–5 after examining the seed-5 replay, so this activation comparison is exploratory rather than an untouched validation set. They appear in the common-budget table, and the fixed-data findings are reported separately.

Coverage AUC is the exact mean fraction of reachable positions visited after every real interaction, calculated from raw replay. Final coverage and time to coverage thresholds are reported separately. Oracle geometry is confined to evaluation and hyperparameter selection. Baselines are matched by interactions and update schedule, not wall-clock compute. Collection times reflect varying concurrency and are not a controlled speed benchmark. No online checkpoint/model-selection criterion uses coverage.

## Screening results

| Setting | Position AUC | Final positions | Final rooms | Updates | Collection seconds |
|---|---:|---:|---:|---:|---:|
| random | 31.4% | 60.7% | 75.0% | 4,505 | 11 |
| horizon8 | 29.2% | 46.7% | 56.2% | 4,505 | 1286 |
| single_gaussian | 28.4% | 42.7% | 56.9% | 4,505 | 142 |
| wide128 | 27.4% | 37.3% | 45.1% | 4,505 | 173 |
| variance1e-3 | 26.1% | 41.6% | 56.2% | 4,505 | 192 |
| latent2 | 26.1% | 47.6% | 60.4% | 4,505 | 171 |
| repulsion30 | 25.6% | 39.3% | 50.7% | 4,505 | 158 |
| shooting | 25.2% | 42.2% | 58.3% | 4,505 | 172 |
| epsilon30 | 24.8% | 34.2% | 41.0% | 4,505 | 232 |
| horizon1_mc64 | 23.9% | 33.3% | 41.0% | 4,505 | 38 |
| repulsion1 | 22.9% | 38.2% | 47.9% | 4,505 | 155 |
| mc64 | 22.6% | 31.2% | 42.4% | 4,505 | 539 |
| latent2_mc64 | 22.5% | 30.7% | 34.0% | 4,505 | 493 |
| latent16 | 21.2% | 36.1% | 44.4% | 4,505 | 196 |
| base | 20.9% | 37.7% | 53.5% | 4,505 | 176 |
| ensemble5 | 20.6% | 34.0% | 43.8% | 4,505 | 226 |
| horizon1 | 19.9% | 27.7% | 38.2% | 4,505 | 33 |
| lr1e-3 | 19.4% | 38.4% | 47.2% | 4,505 | 259 |
| latent4 | 17.2% | 27.9% | 36.1% | 4,505 | 269 |
| fit4x | 16.0% | 23.9% | 31.2% | 18,020 | 331 |

![Screen](../runs/probabilistic-sweep-summary-20261104/screen.png)

## Matched 20,000-step confirmation

| Setting | Seeds | Mean position AUC | Mean final positions | Mean final rooms | Position range |
|---|---:|---:|---:|---:|---:|
| base | 3 | 32.3% | 45.7% | 52.3% | 38.5–58.9% |
| horizon8 | 3 | 32.9% | 48.2% | 58.1% | 39.8–55.5% |
| random | 3 | 48.7% | 72.5% | 81.7% | 69.3–75.8% |
| random_iid | 3 | 26.3% | 42.2% | 47.2% | 35.4–53.3% |
| single_gaussian | 3 | 38.8% | 54.7% | 60.2% | 51.0–60.9% |
| unbounded | 3 | 31.8% | 45.9% | 52.8% | 32.3–60.1% |
| wide128 | 3 | 35.2% | 49.1% | 56.2% | 37.4–57.2% |

![Matched coverage](../runs/probabilistic-sweep-summary-20261104/matched-coverage.png)

## 100,000-step architecture comparison

| Setting | Seeds | Mean position AUC | Mean final positions | Mean final rooms | Position range |
|---|---:|---:|---:|---:|---:|
| random | 3 | 83.8% | 99.4% | 100.0% | 99.0–99.8% |
| random_iid | 3 | 61.5% | 82.6% | 84.5% | 79.4–86.9% |
| single_gaussian | 3 | 63.1% | 74.2% | 78.2% | 57.3–90.9% |
| wide128 | 3 | 50.8% | 56.3% | 63.2% | 42.7–64.9% |

![Long coverage](../runs/probabilistic-sweep-summary-20261104/long-coverage.png)

### Individual final evaluations (budgets differ)

| Setting | Seed | Steps | Position AUC | Final positions | Steps to 80% positions | Replay JS | Seen / unseen JS | No-motion actions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| random | 3 | 100,000 | 84.1% | 99.4% | 28567 | 0.085 | 0.097 / 0.061 | 59.9% |
| random | 4 | 100,000 | 84.7% | 99.0% | 23014 | 0.189 | 0.203 / 0.133 | 59.5% |
| random | 5 | 100,000 | 82.5% | 99.8% | 25358 | 0.226 | 0.243 / 0.139 | 59.3% |
| random_iid | 3 | 100,000 | 59.8% | 79.4% | not reached | 0.070 | 0.080 / 0.146 | 31.3% |
| random_iid | 4 | 100,000 | 60.3% | 86.9% | 81251 | 0.078 | 0.089 / 0.110 | 32.1% |
| random_iid | 5 | 100,000 | 64.3% | 81.5% | 79690 | 0.121 | 0.123 / 0.124 | 31.7% |
| single_gaussian | 3 | 100,000 | 66.6% | 74.4% | not reached | 0.088 | 0.075 / 0.078 | 27.5% |
| single_gaussian | 4 | 100,000 | 70.9% | 90.9% | 59099 | 0.083 | 0.100 / 0.216 | 33.1% |
| single_gaussian | 5 | 100,000 | 51.9% | 57.3% | not reached | 0.066 | 0.071 / 0.114 | 22.9% |
| wide128 | 3 | 100,000 | 58.0% | 64.9% | not reached | 0.083 | 0.076 / 0.083 | 27.4% |
| wide128 | 4 | 100,000 | 39.2% | 42.7% | not reached | 0.318 | 0.299 / 0.447 | 35.5% |
| wide128 | 5 | 100,000 | 55.1% | 61.4% | not reached | 0.200 | 0.162 / 0.175 | 28.6% |
| horizon8 | 3 | 20,000 | 34.7% | 55.5% | not reached | 0.124 | 0.113 / 0.107 | 34.3% |
| horizon8 | 4 | 20,000 | 30.9% | 39.8% | not reached | 0.330 | 0.275 / 0.366 | 38.9% |
| horizon8 | 5 | 20,000 | 33.0% | 49.2% | not reached | 0.243 | 0.196 / 0.228 | 38.1% |
| base | 3 | 20,000 | 38.4% | 58.9% | not reached | 0.117 | 0.109 / 0.154 | 33.6% |
| base | 4 | 20,000 | 32.3% | 39.8% | not reached | 0.247 | 0.291 / 0.426 | 38.0% |
| base | 5 | 20,000 | 26.3% | 38.5% | not reached | 0.235 | 0.246 / 0.340 | 40.4% |
| unbounded | 3 | 20,000 | 38.0% | 60.1% | not reached | 0.087 | 0.092 / 0.092 | 39.2% |
| unbounded | 4 | 20,000 | 31.2% | 45.1% | not reached | 0.238 | 0.242 / 0.580 | 51.5% |
| unbounded | 5 | 20,000 | 26.2% | 32.3% | not reached | 0.627 | 0.468 / 0.617 | 67.7% |

## Fixed-data activation diagnostic

Two fresh fits use the same 100k-transition uniform-random seed-5 replay and the same whole-episode 80/20 split: 80k training transitions, 20k held out, 50k optimizer updates each. Only the encoder output activation changes. No new interactions are collected. The evaluator retrieves the nearest encoded maze position to the ensemble-mean prediction; the full catalogue is never available to the agent. This is a specific readout, not an optimal decoder. Observed/unobserved refers to physical state-action pairs in the fitting buffer.

| Activation | All-pair retrieval | Unobserved-pair retrieval | Mean physical error | Horizontal / vertical latent step |
|---|---:|---:|---:|---:|
| tanh | 74.3% | 50.9% | 0.1711 | 0.98 |
| identity | 89.3% | 77.3% | 0.0339 | 2.90 |

![Activation comparison](../runs/probabilistic-sweep-summary-20261104/activation-comparison.png)

Both offline fits preserve sensitivity to horizontal and vertical movement. The original online uniform-random seed-5 model was nearly insensitive to vertical movement (horizontal/vertical step ratio about 758), so the output bound alone does not explain that earlier failure. Offline fitting sees the full buffer from the beginning; online fitting has a changing data distribution. The online activation comparison tests whether the fixed-data prediction benefit carries over to exploration.


## What the diagnostics show

### Prediction and exploration can diverge

In the matched 20k online activation follow-up, bounded encoders achieved mean all-state next-position retrieval of **52.6%**, versus **64.5%** for unbounded encoders. Mean physical error fell from **0.4664** to **0.1127** maze units. Each unbounded run improved these prediction metrics relative to its matched seed, although the policies collected different data. Position coverage was **45.7% versus 45.9%**, and coverage AUC **32.3% versus 31.8%**. Better prediction under this readout therefore did not yield better exploration in this comparison.

The paired fixed-data result above isolates activation more cleanly: identical split, initialization seed, and replay-sampling sequence, with 50k updates each. It is one paired seed, not a general claim that unbounded representations always win. A move is 0.25 maze units. All-state retrieval uses a privileged evaluation catalogue and the ensemble-mean prediction; it is not an optimal decoder or an agent capability.

### Stalls persist while the agent moves

Single-Gaussian seed 3 ended at 74.4% coverage and discovered its last new position at step **53,579**. Seed 5 ended at 57.3%, with its last discovery at **75,598**. Neither discovered a new position during the final 20k steps, while they still moved on **84.5%** and **87.4%** of those steps. They visited 368 and 344 distinct, already-known positions in that interval. The failure is not simply standing still or always hitting a wall.

Persistent random repeats its previous action with probability 0.8; otherwise it samples uniformly (which can select the same action again). Its advantage over independent uniform actions is large: 99.4% versus 82.6% final coverage at 100k. This motivates a future controlled test of temporally persistent candidate paths or macro-actions. That proposed policy change was not tested in this batch.

### Global spread is not a sufficiency test

The online uniform-random seed-5 encoder retained substantial variance and effective covariance rank about 2.04, yet responded roughly **758 times more to horizontal than vertical movement**. Its all-state next-position retrieval was only 12.8%. The deterministic encoder still produced distinct floating-point vectors for all evaluated positions; the issue is extremely weak encoding of a navigational direction, not an explicit codebook or exact state merge.

A fresh offline tanh fit on the same dataset preserved both directions, so online training history matters alongside architecture. Conversely, the unbounded offline encoder had a lower effective covariance rank (1.38 versus 2.85) while predicting physical positions better. Neither global variance nor covariance rank is a standalone quality measure.

### Disagreement needs independent validation

Familiar and unfamiliar state-action pairs can have similar final disagreement. For example, single-Gaussian seed 3 has seen/unseen mean JS 0.075/0.078 nats. Other runs separate them more strongly, so this is not a universal failure. The visit-count plot is a cross-sectional diagnostic: transition difficulty and policy selection also affect which pairs are visited. It does not establish a causal relationship between visitation and uncertainty.

![Long-run uncertainty diagnostics](../runs/probabilistic-sweep-summary-20261104/long-diagnostics.png)

An additional estimator probe compared eight-sample immediate JS rankings with a 256-sample reference on 64 states, repeated 20 times. Action agreement was 62.4% for the 10k default checkpoint and 28.0% for the saturated latent-2 checkpoint. The reference is itself approximate. Increasing samples to 64 did not produce a clear exploration win in the seed-0 follow-ups. An independent numerical-integration test verifies the JS implementation's value on overlapping mixtures; this validates the calculation, not epistemic calibration of the learned ensemble.

## Interpretation limits

These are coordinate-observation results on a single deterministic maze layout. They do not test pixel representation learning, nuisance invariance, stochastic navigation, or transfer. Ensemble disagreement is a heuristic, not calibrated epistemic uncertainty. All-state latent errors use each model’s learned coordinate system and should not be treated as directly comparable physical errors. Final probes include seen transitions. NLLs across separately learned encoders are not model-selection scores.

Each run retains its resolved config, exact source snapshot/hashes, training and exploration logs, replay, checkpoint, and final evaluation. The machine-readable `results.json` includes per-run paths and full diagnostics. The manifests record the grid and seeds; the runner isolates RNGs in worker processes and refuses to overwrite incomplete runs.

## Reproduction and validation

From the repository root, each manifest is runnable with:

```bash
.venv/bin/python scripts/run_probabilistic_sweep.py \
  --manifest configs/probabilistic_sweep.json \
  --out runs/new-screen --jobs 3
```

| Manifest | Budget per run | Runs | Purpose |
|---|---:|---:|---|
| `configs/probabilistic_sweep.json` | 10k | 16 | Initial screen, including persistent random |
| `configs/probabilistic_sweep_followup.json` | 10k | 4 | Adaptive latent-width / MC-precision screen |
| `configs/probabilistic_long_random.json` | 100k | 3 | Persistent-random reference |
| `configs/probabilistic_long_iid.json` | 100k | 3 | Independent-action random control |
| `configs/probabilistic_long_gaussian.json` | 100k | 3 | One Gaussian per ensemble member |
| `configs/probabilistic_long_wide.json` | 100k | 3 | Width-128 encoder and dynamics |
| `configs/probabilistic_long_deep.json` | 20k | 3 | Horizon 8 / 48 simulations |
| `configs/probabilistic_online_activation.json` | 20k | 6 | Matched tanh / identity activation follow-up |

For the fixed-data activation fits, run `map_learning train` with the probabilistic base config, the saved uniform-random seed-5 replay, `seed=5`, `train.updates=50000`, and `model.latent_activation=tanh` or `identity`. Dataset geometry is read from replay; whole-episode held-out splitting is automatic. Each actual output directory contains the resolved config and provenance.

`report_probabilistic_sweep.py` generates tables and coverage plots. `evaluate_next_state_retrieval.py`, `plot_disagreement_visits.py`, and `probe_disagreement_noise.py` provide the additional diagnostics. The bundled `results.json` includes all online metrics, fixed-data fits, retrieval results, visitation diagnostics, and the reward-noise probe.

Validation: **30 tests pass**, including mixture likelihood against PyTorch, bimodal fitting, sampling, JS against independent numerical quadrature, joint gradients, checkpoint continuation, stochastic search, replay-sampling equivalence, exact coverage AUC, and legacy activation-config evaluation. All 41 online budgets, update counts, replay lengths, checkpoints, and matching 1,000-step warmups were checked. The iid-random control intentionally uses a different warmup policy. No unresolved runs remain.

One run completed training but hit a legacy-config compatibility error during final evaluation. Its evaluation was recovered from saved artifacts; checkpoint, replay, and config hashes were verified unchanged. The error and resolution are retained in the run's recovery record and sweep's resolved-failure archive. No training was repeated or discarded. Online runs that predated the opt-in activation feature retained their original loaded training code; default tanh behavior remains compatible.

This was an exploratory sequence with adaptive follow-ups and three seeds per confirmed setting, not a preregistered benchmark. All environments used the same maze layout. The results establish limitations and promising diagnostics for this implementation, not a general verdict on learned-model exploration.
