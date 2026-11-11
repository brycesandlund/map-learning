# Continuous-coordinate frontier protocol — November 11, 2026

User request: move to continuous maze coordinates; test existing dynamics and adjacency if useful. Preexisting uncommitted distance-study work is preserved. No subagents, commits, or baseline-default changes.

## Fixed environment and learner

Same 12×12 maze (initial layout seed 7), normalized xy plus eight independent standard-normal nuisance coordinates, free exact simulator restores, persistent raw-observation frontier and visited archive. No physical identity deduplication or latent merging. Priority is Euclidean distance to the nearest expanded latent. Every archive/frontier observation is reencoded and every priority recomputed after training. Coverage never affects the agent.

Each expansion executes four continuous controls plus stay. A fresh uniform rotation in [−π/4, π/4] sets four perpendicular directions; independent control lengths are uniform in [.5, 1]. Maximum displacement is .25 maze units initially. Wall intersections reject the whole move, as before. Dynamics receives the actual 2D control vector: no action binning or unobserved motor randomness. Positions are not snapped; this is not observation jitter applied to a discrete simulator. Stay and wall collisions still yield exact physical revisits.

Native deterministic encoder 10→256→256→256→8, SiLU, three residual dynamics heads with 2D controls. Dynamics objective: transition MSE + .03 repulsion + .03 latent L2. Adjacency: .3 squared endpoint distance + the same regularizers; auxiliary dynamics detached. Both encoder endpoints receive gradients. No posterior or distributional latent sampling.

Shared batches. Train every 100 expansions / 500 actions, using causal slices of one replay epoch per 5,000 actions. Every new transition is included. At 100k: 4,112 updates and 1,050,000 presentations per head. Archive/frontier survive all training. Loss coefficients remain frozen throughout this study.

## Evaluation established before launch

Rooms are connected by construction; walls have zero thickness. Record endpoint occupancy of bins with widths 1, .5, .25, .125. Main finite-resolution target: 95% of .25-wide bins. Also report all 144 rooms entered, 90/99% targets, coverage curves/AUC and finer resolution. No claim of covering infinitely many continuous states. Unique endpoints and fraction on the old lattice are sanity checks only.

Final fill metric uses fixed 8×8 probes per room and their nearest observed endpoint **in the same room**, reporting fractions within radii .05/.1/.2. This avoids credit through walls, is conservative across open room boundaries, and is not geodesic distance. Animation uses actual continuous endpoints and draws no fictitious travel across restores. Figures can subsample points; metrics use all observations.

Initial parameters must match within each seed and observation dimension. Action proposal and observation-noise RNGs are independent of training RNGs. Audit replay controls, actual physical transitions, observed xy, occupancy recounts, training budgets, and checkpoint continuation. Probe positions, noise redraws, and spatial labels are evaluation-only.

## Initial screen, declared before launch

Four methods × seeds 30–32 × 100k actions: dynamics, adjacency, coordinate-only oracle, random queue pop. Oracle discards known nuisance channels; random queue is an untrained diagnostic. Neither is a matched-compute learned baseline. Frozen arm manifest: `configs/continuous-screen-20261111.json`; runs: `runs/continuous-screen-20261111`.

Planned adaptive rule: inspect all three seeds before selecting follow-ups. If successful, confirm fresh seeds/layouts and isolate noise/proposal effects. If failure, inspect controls and fitting before tuning. Preserve all runs; never overwrite.

## Decision after all three screen seeds

Both learned methods reach all quarter-width bins and all rooms at 100k; fine-bin occupancy 98.8–99.5%. Dynamics 95% times: 51,847 / 60,542 / 49,529. Adjacency: 50,136 / 54,277 / 72,537. Oracle: 13,867 / 14,204 / 13,767. Random queue: 3–4% final quarter-bin coverage. No loss retuning justified.

- Six fresh seeds 33–38 on layout 7, both learned losses: `continuous-confirm-20261111`.
- Layouts 13/23/37 × seeds 39–41 × both losses: `continuous-layouts-20261111`.
- Original seeds 30–32, both losses under clean input, IID directions, and half-sized moves: `continuous-ablation-20261111`.
- Before ablation launch, add frozen random encoders with/without nuisance. Clean learned arms alone cannot isolate the value of training.

Corresponding frozen JSON arm lists are under `configs/` with the same stage names. Each stage has an immutable `manifest.json`, base config, per-job command/status, run config, source snapshot and results. These settings were chosen before their outcomes, without selecting favorable seeds.

## Adaptive half-step fitting intervention

Declared after first half-step seed finished: original 100k coverage is 57.9% dynamics / 72.5% adjacency. Add half-step coordinate-only control, plus 1,000 extra replay updates for each learned objective. Compare all extra updates after the first 5,000 actions against the same 1,000 updates spread evenly over 20 blocks (50 per block). Use all preselected seeds 30–32, including those whose original half-step runs had not finished.

Extra updates use existing noisy replay only. Both placements have 5,112 updates and 1,306,000 presentations per head, so they are compute/sample-matched to each other, but have more compute than the original cadence. Arm list: `configs/continuous-smallstep-followup-20261111.json`; run root with the same stage name.

## Bounded longer-budget check

Declared after first half-step dynamics fitting pair finished: extra fitting helps but does not complete coverage (early 78.9%, distributed 66.3%, original 57.9%). Extend the **original** half-step protocol to 200k for both losses and all three preselected seeds, without extra updates or retuning. Run root `runs/continuous-long-20261111`, manifest `configs/continuous-long-20261111.json`.

The first 100k replay and checkpoint must reproduce the original half-step runs exactly. These are extensions, not additional independent seeds. The epoch schedule supplies about four times the cumulative optimization at 200k, so extension results do not isolate additional data from additional training. No further budget extensions planned.

## Reproduction

For a single run, see the main report. For the initial screen:

```
.venv/bin/python scripts/run_continuous_study.py \
  --arms configs/continuous-screen-20261111.json \
  --seeds 30 31 32 --workers 4 --blocks 20 \
  --out runs/reproduce-continuous-screen
```

Use the corresponding arm manifest and seed list above for other stages; `--blocks 40` for the long stage. Output roots must not exist. `scripts/analyze_continuous_study.py` audits completed stages; `scripts/report_continuous_study.py` renders validation/ablation plots; `scripts/report_continuous_smallstep.py` verifies exact prefixes and plots longer runs. `scripts/render_continuous_study.py` renders continuous endpoint animations. `scripts/audit_continuous_representation.py` probes checkpoints without modifying them.
