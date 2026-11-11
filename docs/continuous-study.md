# Continuous-coordinate maze exploration

Study in progress, November 11, 2026. Final validation and ablations are still running; the results below are provisional until that section is finalized.

## What changed

The physics already accepted bounded 2D controls, but the previous five fixed actions kept positions on a quarter-cell lattice. The new experiment samples real-valued controls and supplies the **actual control vector** to each dynamics head. Position is not snapped or rounded. This adds action diversity without hidden motor noise: given a position and a control, the physical transition remains deterministic.

At each expansion, four perpendicular directions have a fresh random rotation, and their lengths are independently uniform from .5 to1 in control units. With step_size=.25, nonblocked moves span .125–.25 maze units. A fifth action stays still. Every child comes from an actual simulator step. The existing swept wall-intersection check rejects blocked moves completely, including diagonal paths; it does not slide along walls or teleport through them.

We retain the 12×12 maze, eight independent Gaussian observation-noise coordinates, exact simulator restores, persistent raw-observation frontier and visited archive, and nearest-expanded-latent queue priority. Queued states retain their original noisy observation; all archived/queued observations are reencoded after training. There is **no state deduplication, distance threshold, quantization, or coverage signal in the agent**. Both learned methods still use deterministic8D encoders. Adjacency's auxiliary dynamics are detached and diagnostic only.

## Coverage means finite spatial resolution

There are infinitely many mathematical positions, so unique-coordinate count is not a completion metric. We record occupancy of spatial bins of width1,.5,.25,.125; the main sample-efficiency target is95% occupancy of.25-wide bins. The maze has144unit rooms,2,304quarter-width bins, and9,216eighth-width bins. Walls have zero thickness and every room is connected by construction.

A bin counts when an actual observed endpoint lies inside it; we do not credit the full path segment, latent predictions, or simulator resets. Bins are used only by evaluation, never by the learner/queue. As an additional measure, fixed probes in each room are tested for proximity to an observed endpoint in the **same room**, avoiding spurious coverage through walls. This is conservative at open room boundaries and is not a geodesic metric.

## Initial three-seed comparison

Both dynamics and adjacency reach every room and all quarter-width bins at100k. Fine-bin occupancy is98.8–99.5%. Coordinate-only distance is a privileged diagnostic which discards known nuisance input dimensions. Random queue selection uses the same physical proposals and simulator-reset protocol.

|Method|Actions to95% quarter-bin occupancy, seeds30/31/32|Final quarter-bin occupancy|
|---|---|---|
|Dynamics|51,847 /60,542 /49,529|100% each|
|Adjacency|50,136 /54,277 /72,537|100% each|
|Coordinate-only diagnostic|13,867 /14,204 /13,767|100% each|
|Random queue|Not reached|3.4% /4.0% /3.6%|

![Initial coverage at four spatial resolutions](assets/continuous-screen.png)

First prespecified seed30; the animation displays a deterministic subsample of cumulative **actual continuous endpoints**, not an evaluation lattice. Red marks the latest expansion, not a physically traversed line between simulator restores. Numerical occupancy uses every endpoint, regardless of display subsampling.

![Continuous exploration](assets/continuous-screen.gif)

## Validation and ablations

Loss coefficients were frozen before validation. Both methods reach all 144 rooms and the 95% quarter-bin target in **all 15 validation runs per method** (six new seeds on the original layout, nine runs across three new layouts). At 100k, all but one run per method reach every quarter-width bin; those two runs each miss just one of 2,304 bins.

| Method | Mean actions to 95%, original layout / 6 fresh seeds | Three new layouts / 3 seeds each |
|---|---:|---:|
| Dynamics | 62,715 | 62,149 |
| Adjacency | 55,320 | 60,977 |

Adjacency wins 5/6 paired seeds on the original layout, about 11.8% fewer actions in the mean. An exploratory paired-bootstrap interval for adjacency minus dynamics is approximately [−13,708, −718] actions. This is six seeds on one layout, not six environments.

On new layouts its mean advantage falls to 1.9%, with only 4/9 paired wins. Dynamics wins every seed on layout 23. Fine-bin occupancy at 100k ranges from 97.8–99.6% for dynamics and 96.7–99.7% for adjacency across these layouts. Thus adjacency is useful but not a consistently superior replacement. The default remains dynamics.

Noise, proposal, frozen-encoder and smaller-step follow-ups are being finalized.

## Implementation and reproducibility

Dynamics: coordinate encoder10→256→256→256→8 with SiLU, and three residual dynamics heads with2D continuous controls. Joint transition MSE +.03 random-pair hinge repulsion +.03 latent squared norm. The dynamics MSE is averaged over latent dimensions and heads. Both endpoint encodings retain gradients.

Adjacency replaces encoder dynamics gradients with `.3 E||f(o)-f(o')||²`, keeping the same repulsion/L2. Squared adjacency distance sums over dimensions. Both encoder endpoints train; auxiliary dynamics receive detached embeddings. No posterior distributions or sampling during encoding.

Every100expansions/500actions, training receives a causal slice of one full replay epoch per5000actions. Thus each100k learned run has4,112updates and1,050,000transition presentations per head. This is **not** a full epoch every500actions. All newly collected transitions are trained; old experience stays in replay. Controls without learning have zero updates and are interaction-matched, not compute-matched.

```
.venv/bin/python scripts/continuous_frontier.py \
  --objective dynamics --seed 100 --blocks 20 \
  --out runs/new-continuous-dynamics

.venv/bin/python scripts/continuous_frontier.py \
  --objective adjacency --seed 100 --blocks 20 \
  --out runs/new-continuous-adjacency
```

Use `--proposal iid`, `--nuisance-dims 0`, `--step-size .125`, or `--layout 13` for the corresponding interventions. `--objective oracle_xy|random_queue|frozen` selects diagnostic controls. `configs/continuous_frontier.json` is separate; existing discrete-action defaults and checkpoints remain supported.

The experiment retains coordinate observations and exact resets. It does not establish exploration with pixels, unreliable restores, a continuous-action learned actor, or imagined latent rollouts. Continuous simulation uses floating-point coordinates, not an imposed lattice; stays and blocked actions still revisit exact positions.

[Protocol and staged decisions](continuous-study-protocol.md). Initial machine-readable results: [screen](assets/continuous-screen-results.json). Checkpoint representation probes: [audit](assets/continuous-representation-audit.json). Run roots and frozen arm manifests are recorded in the protocol; each run retains source snapshots, replay, checkpoints, metrics and visualization frames.
