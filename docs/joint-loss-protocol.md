# Combined-objective / loss-scale study — November 11, 2026

Requested: combine dynamics and adjacency, sweep all loss coefficients and the repulsion hinge threshold. Preserve prior work. No subagents or commits.

Encoder objective:

`D * mean_{heads,batch,coordinates}(g(z,a)-z')² + A * E||z-z'||² + R * E[max(m²-||zu-zv||²,0)] + L * E||z||²`.

Both encoder endpoints receive gradients. D>0 for dynamics/hybrid; pure adjacency uses the existing detached auxiliary dynamics implementation (D=0 for encoder). Native dynamics is A=0. Margin m is a distance: the historical `1` inside the hinge is m². Changing m also changes the maximum repulsion value, so margin effects are not a pure change in semantic neighborhood size. Positive scaling of all latents preserves nearest-neighbor ordering; absolute latent scale and optimizer behavior must be distinguished from improved geometry.

Fixed noisy continuous frontier, exact simulator resets, same architecture/initializations, replay schedule and all five real sampled controls. No labels/coverage/reencoded-clean observations in learning. 100k actions, 4,112 updates, 1,050,000 presentations per head. Max move .25 (normal) and .125 (small). Normal evaluation: time to95% quarter-bin coverage and occupancy AUC. Small-step screen: mean final quarter-bin coverage, AUC as tie-breaker; report individual seeds and failures. No hyperparameter chosen on confirmation seeds.

## Stage 1 — mixture strength (declared before launch)

Seeds42–44, both movement scales. D=1; A=.01/.03/.1/.3/1 for hybrids; existing pure dynamics and pure adjacency(.3) controls. R=L=.03, m=1. Total 42 runs. Manifest `configs/joint-loss-mix-20261111.json`. Distinguishes adding adjacency from replacing dynamics. Freeze a promising hybrid mixing value only after all three seeds complete.

## Planned coefficient/margin stage

Use the smaller-step setting where baseline has substantial headroom. Independently vary D, A, R, L and m, including zero-L2 and weak-repulsion controls; combine targeted one-factor sweeps with joint combinations to avoid drawing conclusions only from independent changes. Use two tuning seeds 42/43 and include endpoints; select by mean final coverage, inspecting both seeds. Check margin normalization/scale equivalence explicitly. Record exact immutable arm manifest and decisions before launch.

Then confirm selected methods against both historical baselines on six fresh seeds 60–65 at both move scales. Include a pure-adjacency counterpart with the winning hybrid's regularizers to isolate whether a gain actually needs dynamics. If useful, test held-out layouts with fixed settings. Further work only for a concrete unresolved result; never overwrite runs. Retain partial/failing settings in report.

## Validation

Test weighted loss arithmetic, both-term gradient behavior, old default compatibility, exact checkpoint continuation and squared-margin semantics. Audit real transitions/coverage, matched budgets, initial parameter hashes and action streams. Log weighted loss terms, hinge-active fraction, latent spread/distances and fine-resolution coverage. Produce plots/animations and update training_runs.md. Outcomes under resets/coordinate inputs are not evidence for imagined planning or image observations.

## Coefficient stage launched independently of mixture outcomes

To parallelize independent work, anchor one-factor tests at the historical A=.3, D=1, R=L=.03, m=1 rather than wait for a mixing winner. D=.1/.3/3/10; A=3 (smaller A already covered); R=0/.003/.01/.1/.3; L=0/.003/.01/.1/.3; m=.25/.5/2/4. Add12 balanced joint combinations drawn before outcomes across D=.1/1/10, A=.03/.3/3, R=.003/.03/.3, L=.003/.03/.3, m=.5/1/2. Four pure-adjacency coefficient/margin variants and two normalized-hinge controls complete 37settings×2seeds 42/43. For normalized controls set R=.03/m², holding peak hinge contribution fixed; this also changes relative regularization. Immutable manifest configs/joint-loss-scale-20261111.json. No mixture results used in constructing this design.

## Analytical check while sweeps run

For independent replay samples with latent law Z, `E||Z-Z'||² = 2 Var(Z)` (trace variance). Since `[m²-d²]+ >= m²-d²`, repulsion plus L2 is bounded below by `R*m² + (L-2R)*Var(Z) + L*||E Z||²`. Dynamics and adjacency losses are nonnegative and admit zero at a constant zero encoder with zero dynamics output. Therefore **L>=2R makes zero collapse a global optimum** (not necessarily unique at equality). This is a population objective statement; finite sampling/optimization need not reach it. It is independent of margin and supplies a predicted failure regime for the explicit high-L2/low-repulsion controls. No results were discarded on this basis.

## Margin as a change of units

For identity-output encoders, `f_m=m*f` and `g_m(z,a)=m*g(z/m,a)` are exactly realizable by scaling encoder/dynamics final affine layers and inversely scaling latent-input columns of the dynamics first layer. At margin m, all four terms become m² times their margin1 values; nearest-latent queue order is unchanged. `scripts/audit_joint_margin_units.py` verifies this on a trained hybrid checkpoint at m=.25/.5/2/4. This is a static function-class equivalence, NOT equivalence of Adam trajectories with fixed initialization, learning rates, epsilon and clipping. Raw-margin improvements cannot be interpreted as a uniquely better semantic neighborhood radius. Normalized-hinge controls additionally change relative coefficient weighting.

## Confirmation controls launched independently

Run both historical baselines on untouched seeds 60–65 at both step sizes while coefficient tuning continues. Do not inspect these outcomes to choose candidates; only job completion/failures may be monitored. Candidate confirmation remains matched to these seeds/actions/initialization and is selected exclusively from tuning seeds 42–44. This overlaps independent work without using confirmation outcomes for selection.

Held-out layout controls are also launched blind before candidate selection: layouts13/23 × seeds 66–68 × both historical baselines, small-step setting,12 runs. Only completion/failure is inspected before candidate freeze. Selected settings will be evaluated on these same layouts/seeds without further retuning.

Audit during execution found two accidental duplicates: balanced joint-07 equals the D=10 arm; joint-09 equals the R=.3 arm. All remain in the immutable manifest/runs; count 35 unique coefficient-stage settings, not37 independent settings. Do not treat duplicate jobs as extra independent evidence. Use their checkpoint/replay agreement as a reproducibility check and disclose the duplication.

Freeze mixing-only winner for independent confirmation: D=1,A=.03,R=L=.03,m=1, highest mean final small-step coverage over seeds 42–44 (small gain over pure adjacency, not yet established). Confirm at both scales on seeds 60–65. Keep these outcomes blinded until coefficient candidates freeze, so they cannot influence selection. This tests combining alone, separate from coefficient retuning.

Metadata clarification: early runner result files retain the CLI adjacency-weight argument (.3) even for objective=dynamics, where it is unused. Config.train.adjacency_weight=0 is authoritative. Later runner result metadata reports the effective value; no learning behavior changed. Summary JSON explicitly reconstructs effective D/A/R/L/m from the actual objective/config so baseline weights cannot be misread. Existing raw result files were preserved.

## One bounded margin extension (before fresh outcomes inspected)

Both tuning seeds for hybrid m=.25 are 81.21/81.25%, while m=.5 is 77.56/72.05% and m=2 is 61.15/24.48%. The best raw margin is at the tested lower boundary. Add m=.125/.0625 for hybrid and pure adjacency, plus the missing pure adjacency m=.25 counterpart. Same D=1 for hybrid,A=.3,R=L=.03,small-step,seeds 42/43;10 runs. This is the final tuning extension; do not keep shrinking margins adaptively. It distinguishes combining from a change of optimization units and ensures matched pure-adjacency controls for these candidates. Candidate freeze waits for both original sweep and this extension.

Before unblinding, add a matched pure-dynamics counterpart to the winning hybrid (same D/R/L/m; A=0) if it differs from the historical baseline. This prevents attributing a margin/regularizer gain to adjacency or mixing without comparing dynamics at the same scale. Confirm it on the same fresh seeds and layouts; no additional tuning.

## Final candidate freeze — 2026-11-11 18:45 UTC, before unblinding

All126 tuning jobs completed. Selector uses only common seeds 42/43 for the coefficient decision and deduplicates identical tuples. Selected hybrid D=1,A=.3,R=L=.03,m=.25 (81.21/81.25%); selected pure adjacency D=0,A=.3,R=L=.03,m=.125 (87.24/75.65%). Include matched pure adjacency at m=.25 and matched pure dynamics D=1,A=0,m=.25, both R=L=.03. These are four candidates,48 fresh-seed runs and24 additional-layout runs, no further retuning. Prior mixing-only A=.03,m=1 confirmation remains separate. Exact settings/ranking are in configs/joint-loss-selected-20261111/{selection,confirm-arms,layout-arms}.json. Fresh control and mixing outcomes may now be inspected. Layouts13/23 were used in prior studies but not for this study's tuning.

## Completion

All 246 jobs completed successfully; no additional tuning was launched after unblinding. Main report: joint-loss-study.md. The representative animation uses the prespecified seed 60; its final panel shows the matched m=.25 adjacency control so the last two panels isolate the dynamics contribution. Pure adjacency at m=.125 remains in all tables/plots. Fresh physical probes and a read-only saved-prefix audit were added after outcomes to inspect mechanisms; neither changed training or selection. All 12 pure-objective margin1/.25 pairs match exactly through the 25k checkpoint; later data diverge.
