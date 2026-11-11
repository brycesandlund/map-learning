# Deterministic map learning with latent regularization

The current baseline and L2 model need no probabilistic encoder or predictive distribution. Write the encoder as

\[
z=f_\theta(o)\in\mathbb R^d,
\qquad g_m(z,a)=z+h_{\psi_m}(z,\operatorname{onehot}(a)),
\quad m=1,\dots,M.
\]

Each dynamics head predicts a single successor latent. With replay transitions (o,a,o′) and independently sampled replay observations o_i,o_j, the objective is

\[
\mathcal L =
\mathbb E_{(o,a,o')}\frac1{Md}\sum_{m=1}^M
\|g_m(f_\theta(o),a)-f_\theta(o')\|_2^2
+\alpha\,\mathbb E_{o_i,o_j}
[\delta^2-\|f_\theta(o_i)-f_\theta(o_j)\|_2^2]_+
+\lambda\,\mathbb E_o\|f_\theta(o)\|_2^2.
\]

The terms are transition prediction, repulsion, and latent L2. Dynamics loss trains the encoder through both its input and its target; neither is detached. Repulsion and L2 also train the encoder. L2 sums latent coordinates and averages replay observations, using the same observations drawn for repulsion.

The selected noisy-maze configuration is d=8, M=3, α=.03, δ=1, λ=.03. Set λ=0 for the matched unregularized formulation. No posterior, prior, variance head, distribution likelihood, or sampled latent is needed. Observation noise, replay minibatch selection, and search tie-breaking remain random.

The persistent frontier uses

\[
\operatorname{priority}(o)=\min_{v\in V}\|f_\theta(o)-f_\theta(v)\|_2,
\]

where V stores raw observations from expansions. The archive and frontier are re-encoded after training. The optional disagreement planner uses the spread of the M deterministic predictions; this is disagreement across models, not an output distribution.

## Run the clean formulation

```sh
.venv/bin/python scripts/persistent_frontier.py \
  --config configs/deterministic_frontier.json \
  --arm interleaved100 --seed 15 --out runs/deterministic-l2-new
```

For the baseline, add `--latent-l2 0`. The clean config retains the established architecture, learning rates, shared batches, and training cadence. Each dynamics head now has only d output coordinates. The encoder has 136,456 parameters; all three dynamics heads together have 411,672. The old unused encoder-scale head and dynamics logit/scale outputs are absent.

`train.loss=dynamics` is supported by ordinary training/checkpoint loading, replay schedules, MCTS/shooting disagreement planners, reward probes, and evaluation. The historical probabilistic path and historical experiment commands remain available, so old runs can still be reproduced.

The KL comparison remains a distinct historical experiment. Although its dynamics inputs were deterministic, its KL term still trained distribution-scale parameters. Removing those parameters would change that objective. No KL checkpoint is silently converted to L2.

## Convert an existing deterministic checkpoint

```sh
.venv/bin/python scripts/convert_deterministic_checkpoint.py \
  runs/l2-noise-confirm-20261110/l2-0.03-seed15/checkpoint.pt \
  runs/converted-new/l2-seed15.pt
```

The converter strips inactive distribution outputs and preserves encoder/dynamics weights, Adam state, update count, and replay RNG. It refuses active KL, sampled encoder inputs, multi-component dynamics, and an existing destination. This migrates a model checkpoint, not an entire exploration session with its queue and replay.

We converted the trained seed-15 L2 model to `runs/deterministic-conversion-20261110/l2-seed15.pt`. On 2,048 recorded observations, encodings and successor predictions matched exactly; its next training loss and encoder update also matched exactly in this CPU audit. Smaller matrix operations can round differently on other shapes/devices, so long online trajectories are not promised to remain bit-identical. The [audit](assets/deterministic-conversion-audit.json) records the checks.

Fresh initialization uses the ordinary deterministic residual network. Removing unused output rows changes parameter initialization RNG consumption relative to the legacy mixture network. This is the same objective and active network capacity, not a claim of an identical seeded trajectory. Published sweep results remain results of their recorded historical runs.

A new native-config 5,000-action smoke run completed with 20 updates and 5,000 transition presentations/head, retaining all expected artifacts at `runs/deterministic-cleanup-smoke-20261110`. It validates integration, not a new full exploration benchmark. The full test suite has 84 passing tests, including migration/Adam continuity, checkpoint round-trip, rejection of incompatible conversions, and distribution-free training, planning, diagnostics, and evaluation.
