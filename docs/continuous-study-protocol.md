# Continuous-coordinate frontier study, November 11, 2026

User request: move to continuous maze coordinates; test existing dynamics, try adjacency if needed. Existing uncommitted distance-study work is preserved. No subagents or commits.

## Frozen initial protocol

Same 12×12 maze (layout7), normalized xy plus8 independent Gaussian nuisance coordinates, exact free simulator restores, persistent raw-observation frontier and visited archive. No physical identity deduplication or latent merging. Priority remains Euclidean distance to nearest expanded latent; all priorities are recomputed after training. Coverage has no effect on search.

Every expansion executes four continuous controls plus stay. A fresh uniform rotation in[-pi/4,pi/4] sets four perpendicular directions; independent control lengths uniform[.5,1]. Maximum displacement.25 maze units. Wall intersections reject the entire move, as before. Dynamics receives actual2D action vector (no action binning or unobserved motor randomness). Thus states are off-lattice; this is not coordinate jitter applied to a discrete simulator. Input noise persists. Stay and wall collisions may still produce exact physical revisits.

Native deterministic encoder10→256×3→8, three residual dynamics heads now with2D controls, otherwise same settings. Dynamics objective = transition MSE +.03 repulsion +.03 latent L2. Adjacency objective =.3 squared endpoint distance +same regularizers, auxiliary dynamics detached. Shared batches; train every100expansions/500actions using causal slices of one replay epoch per5000actions. At100k:4112updates/1.05M presentations perhead for learned arms. Archive and frontier survive all updates.

First screen: dynamics, adjacency, coordinate-only oracle (untrained; drops the known nuisance channels), random frontier pop (untrained); seeds30/31/32,100k actions each. These controls isolate the queue/action proposal and value of learning; they are privileged/untrained diagnostics, not matched-compute learned baselines. Initial learned parameter hashes must match within eachseed. Action proposal and observation noise RNGs separated from training RNG.

## Evaluation

All rooms connected by construction; walls zero thickness. Record endpoint occupancy of unit rooms and bins with widths.5,.25,.125; bins are evaluation only. Main finite-resolution target:95% of.25-wide bins. Also report all144 rooms entered,90/99% targets, occupancy curves/AUC, and fine-resolution sensitivity. Never equate a bin target with covering infinitely many continuous states. Continuous endpoint count and fraction on old quarter-cell lattice are sanity checks, not exploration success.

Final fill measure: fixed8×8 probes per room, distance to closest visited endpoint **in the same room**, fractions within radii.05/.1/.2. This avoids credit through walls but is conservative at open room boundaries; not geodesic distance. Figures/animation use coverage bins and physical points for visualization only, drawing no fictitious paths across restores. Model quality/noise probes should be evaluated separately from spatial coverage.

## Planned adaptive decisions

Inspect all three seeds before selecting follow-ups. If learned methods work, confirm on fresh seeds and additional layouts; test noiseless inputs and/or action proposal changes to isolate any bottleneck. If failure, use oracle and random controls plus training/noise diagnostics before tuning. Longer budgets only for a justified unresolved question, recorded before launch. Preserve completed runs; never overwrite.

Implementation tests: continuous actions remain bounded and off-lattice, swept wall collisions/endpoints, resolution accounting, replay vector roundtrip, exact checkpoint continuation, auxiliary-dynamics gradient isolation. Existing discrete experiments remain compatible. No changes to the old baseline defaults.

## Decision after all three initial seeds completed

Both learned methods reach100% quarter-bin occupancy and all144rooms by100k, fine-bin occupancy98.8–99.5%. Dynamics95% times51,847/60,542/49,529; adjacency50,136/54,277/72,537. Oracle95%13,867/14,204/13,767; randomqueue3–4% quarter-bin final. No loss retuning justified. Freeze .3 adjacency and.03 regularizers.

Confirm both learned objectives on six fresh seeds33–38, original layout. Test layouts13/23/37 on seeds39–41, three seeds per layout. Then paired ablations on original seeds30–32: nuisance_dims0, IID action directions, and maximum step.125. Everything else held fixed. These ask whether learning is needed, four-direction stratification is essential, and whether smaller physical motion overwhelms noise or simply slows coverage. No choosing seeds by outcome.

Before ablation launch, include frozen random encoder controls with and without nuisance (same seeds30–32). These directly test the value of learning; clean learned arms alone would not isolate it. No training updates for frozen controls.

## Adaptive smaller-step follow-up (declared before launch)

The first paired half-step seed finishes100k with only57.9% dynamics /72.5% adjacency quarter-bin occupancy. Normal-step arms succeed, and first clean/frozen-clean controls are fast. To test whether this is a coverage-proposal limit or insufficient fitting, add half-step coordinate-only oracle and1,000extra replay updates for each objective. Compare all extra updates after the first5,000actions against the same1,000updates spread evenly over20blocks. Same total compute/sample budget between placements; does not isolate compute from the original baseline. Three seeds30–32, including not-yet-completed half-step seeds; no choosing seeds by outcomes. Extra updates train only collected noisy replay; no clean targets or resampling. Per learned run:5,112updates and1,306,000presentations/head. This is a targeted intervention, not a loss sweep.

## Bounded longer-budget check

Extra fitting in the first half-step dynamics seed improves100k coverage but does not complete it (early78.9%, distributed66.3%, original57.9%). Run the original half-step protocol to200k for both losses and all three preselected seeds30–32. No extra updates or retuning in these long runs. Verify their first100k replay and checkpoint20 match the corresponding original100k run exactly. This tests delayed completion, not whether extra data or extra optimization is the cause; the epoch schedule supplies about4× cumulative training at200k. Do not extend further without a new concrete finding.
