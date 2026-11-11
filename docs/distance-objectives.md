# Distance objectives for noisy persistent-frontier exploration

## Protocol and question

Can a distance-centric objective replace the dynamics loss and improve environment-interaction efficiency? Fixed: noisy input, persistent observation archive/frontier, exact simulator restores, nearest-visited Euclidean priority, encoder architecture, encoder learning rate, data cadence and interaction budget. No physical-state labels, known coordinates separated from noise, oracle shortest paths, or resampled observations enter training. The 10 raw observation coordinates all enter the same encoder.

The native deterministic control is the current formulation (joint dynamics MSE + .03 repulsion + .03 latent squared norm), freshly initialized. Legacy posterior-based checkpoints do not have identical initialization and are not paired controls. Seeds18–20 screen; seeds21–23 reserved for confirmation. All initial runs use100k interactions regardless of early completion. Each run records exact first full coverage and 50/90/95/99% milestones, not just final coverage. Report failed completions as censored, not as true100k completion times.

Cadence `interleaved100`: every100 expansions (=500 executed actions), causally partition one replay epoch across each1000-expansion (=5000-action) block. At100k actions this is4112 updates and1.05M transition anchors/presentations per head, not a full epoch every500 actions. Temporal objectives sample an ancestor per transition anchor; an anchor appearing does not imply that its one-step edge was used as a temporal target. Each auxiliary dynamics head does receive that edge.

All distance arms have detached auxiliary dynamics heads with the same initialization and optimizer as the control, for diagnostics. Their loss cannot affect the encoder. They are not used by the frontier. Tests explicitly change these heads while checking an identical encoder update. This preserves comparable diagnostic capacity without claiming a dynamics-free compute cost.

## Notation

`z=f(o)` is deterministic,8-dimensional. `d(x,y)=||f(x)-f(y)||_2`. Pairs `(o,o')` are actual one-action transitions. Random pairs `(u,v)` come independently from replay's source/successor observation mixture. `R=E[1-d(u,v)^2]_+`, `N=E||f(u)||²`. All expectations are minibatch estimates. Distance arms minimize `w*T + alpha*R + lambda*N`; coefficients below are frozen before results. Both encoder endpoints receive gradients.

### Adjacent attraction

`T=E d(o,o')²`. Sweep w=.01,.1,1; alpha=.03; lambda=.03. This is a minimal temporal-smoothness control: neighbors attract, random pairs repel, norms are regularized. Actions are not provided to the encoder loss. Blocked and stay transitions still have fresh observation noise, so adjacency naturally includes nuisance-changing observations without identifying which transitions stayed put.

### QRL-inspired upper-bound constrained expansion

Normalize distance by edge cost c. `T=E[20 softplus(-d(u,v)/(20c))] + mu*(E[max(d(o,o')/c-1,0)²] - epsilon²)`. Epsilon=.05. Update mu by projected ascent `mu <- max(0,mu+.1*constraint)` after each primal step; initial mu=1. Sweep c=.05,.1,.25, w=.03, alpha=0, lambda=.03. Log multiplier and violation; finite optimization is not proof the constraint is satisfied.

This adapts [QRL](https://arxiv.org/abs/2304.01203): Euclidean metric instead of an expressive directed IQE, an upper-bound local constraint instead of the paper's practical squared deviation from1, plus norm regularization. The upper-bound form avoids explicitly demanding nonzero distance for every noisy stay transition. It does not inherit exact shortest-path recovery guarantees. The norm penalty and finite Euclidean dimension further change the optimum.

### Time-Myopic-inspired Euclidean temporal stress

A node is an observation occurrence, with its actual parent in the expansion tree. No merging. Source roots and successors are indexed explicitly; queue entry i corresponds to replay transition i and node i+1. Sampling never treats successive frontier pops as consecutive motion.

For a transition's child, sample ancestor lag k uniformly from0 through min(depth,H) for half the minibatch, and0 through min(depth,4H) for the other half. `T=E[(1-exp(-d(ancestor,child)) - min(k/H,1))²]`. Sweep H=4,16,64, w=1, alpha=.03, lambda=.03. Distances follow recorded ancestry, which may include loops and stays: they are observed path lengths, not shortest paths.

The target clipping and output transform follow [Time-Myopic Go-Explore](https://arxiv.org/abs/2301.05635). Replacing its learned directed distance head with a Euclidean norm, adding regularizers and using a persistent continuous-priority frontier are adaptations. The original archive insertion procedure is not reproduced.

### Student-t neighbor affinities

For B actual transition pairs, concatenate their2B endpoint encodings. P has mass1/(2B) on each source/successor pair in both directions, zero elsewhere. Q is proportional to `(1+||zi-zj||²)^(-1)` over all ordered non-diagonal pairs. `T=KL(P||Q)`. Sweep lambda=.003,.03, w=.03, alpha=0 (normalization supplies the repulsive pressure). No perplexity tuning here: P comes directly from observed adjacency.

This borrows the Student-t low-dimensional affinity and KL objective from [t-SNE](https://jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf), but replaces its high-dimensional Gaussian affinities with temporal adjacency and uses a parametric encoder trained online. Different occurrences remain separate candidates, including repeated/noisy observations of the same physical state. This may create false-negative pressure; diagnose rather than assuming noise invariance.

## Reproduction

```
.venv/bin/python -m pytest -q
.venv/bin/python scripts/run_distance_study.py --out runs/distance-screen-20261110 --workers 4
```

The driver writes an immutable manifest, fully expanded objective specs, subprocess logs/status files, source snapshots and fresh run directories. It refuses an existing study directory. Distance checkpoints preserve objective specification, occurrence ancestry and independent pair-sampling RNG; `Trainer.load` dispatches appropriately. Baseline and distance arms have identical initial encoder AND dynamics hashes for a given seed. All physical-state/noise diagnostics use independent RNG streams.

## Analysis planned before results

Primary: interaction efficiency, full-coverage rate, coverage-area summary and milestones across seeds. Secondary: nuisance/movement ratios and latent geometry at matched interaction counts; nearest-queue ranking (top-k genuinely novel physical states, evaluation only); actual geodesic correlations/distortions as diagnostics, not training labels. Retain intermediate checkpoints5k/25k/50k/75k/100k, raw trajectories and visualization frames. Any follow-up and selection decision must be logged before launch. Confirmation must include a fresh matched baseline.

## Adaptive follow-up decision (23:34 PST, before follow-up runs)

Evidence available: completed seeds18/19 show adjacency weight.1 competitive with native dynamics, weight1 slower and.01 unsuccessful; first-seed affinity runs complete; QRL-c.05 completes but slowly, c.1/.25 less reliable. Temporal H4/H16 are poor on both first seeds; H64 is incomplete on seed18. The full initial screen remains in flight. First-seed H16 target audit: mean lag11.73 versus geodesic2.93;9.99% of positive-lag pairs share a physical position;59.55% of target values exceed capped shortest distance by>.25. This is diagnostic evidence of supervision mismatch, not causal proof.

Freeze12 follow-up arms on the same screening seeds18–20 (still tuning, not independent confirmation): adjacency (w,lambda)=(.03,.03),(.3,.03),(.1,.01),(.3,.01),(1,.01),(.1,.003); QRL(c,dual_lr)=(.05,1),(.05,10),(.025,1); temporal-upper H4/16/64. The latter changes squared residual to squared positive residual: `T=E[max(1-exp(-d)-min(k/H,1),0)²]`, with all other temporal settings retained. It preserves upper bounds from observed paths without forcing their endpoints far apart. Long clipped pairs impose no extra constraint. Repulsion remains responsible for anti-collapse. No oracle shortest distances are used as targets.

Manifest input: `configs/distance-followup-20261110.json`. These tests target attraction/regularization balance, lagging QRL multipliers, and the equality-versus-upper-bound distinction. Follow-up choices use initial screen evidence, and will require fresh-seed confirmation with matched controls.

### Additional native dynamics controls (23:44 PST, before runs)

The first tuned adjacency seed improves on the original baseline. To avoid attributing an effect of relative loss weights to the objective family, add9 screening runs on seeds18–20: native dynamics with(alpha,lambda)=(.03,.01),(.015,.015),(.0075,.0075). The latter two strengthen prediction relative to both regularizers while retaining their ratio. Parameters are frozen in configs/distance-dynamics-controls-20261110.json. Select the strongest native control using the same screening criterion before fresh-seed confirmation. These are adaptive controls, not confirmations.

### QRL norm-penalty control (23:49 PST, before runs)

Common-replay geometry at50k shows the successful initial encoders nearly preserve Euclidean physical distances (Spearman about.99), with essentially no extra separation for pairs across walls. QRL-c.05 also shows little wall sensitivity and retains one dominant spatial direction. The added L2 regularizer can compete with QRL's distance maximization; therefore test its removal before judging that objective family: c/dual_lr=(.05,.1),(.05,10),(.025,1), alpha=lambda=0, otherwise unchanged, on screening seeds18–20. This remains the Euclidean upper-bound adaptation, not a faithful IQE reproduction. Config frozen at configs/distance-qrl-unregularized-20261110.json. Geometry diagnostics are evaluation only.

## Frozen fresh-seed confirmation (23:54 PST, before runs)

All adjacency and native-control tuning outcomes are available. Choose by mean coverage AUC, requiring3/3 completions: adjacency-w.3/lambda.03 (AUC.6608), native dynamics regularizers-half (AUC.6794); keep original native baseline(AUC.6444) and best Student-t affinity/lambda.03(AUC.6246). Near-tie adjacency-w.1/lambda.01(AUC.6595) is not advanced; do not select again after fresh results. Run all four unchanged settings on fresh seeds21–26 at100k:24new runs. Config `configs/distance-confirm-20261110.json`. This checks whether screening differences reproduce and whether distance objectives match the retuned native control. No claim of equivalence is warranted from a non-significant difference.

The separately predeclared QRL no-L2 ablation is still pending. Only if a no-L2 setting achieves3/3completions and exceeds original baseline screening AUC.6444 will its single best setting be added to these same six confirmation seeds. QRL outcomes cannot alter the four frozen confirmation choices. Additional maze-layout checks, if run, will keep these settings unchanged.

## Frozen extra-layout checks (23:58 PST, before runs)

To check sensitivity to the one maze, retain original native baseline, retuned native dynamics(alpha=lambda=.015), and selected adjacency(w=.3,alpha=lambda=.03) unchanged. Three new maze-layout seeds13/23/37 crossed with three new model/environment seeds27/28/29:27runs at100k. No tuning on these outcomes. Identical noise dimension, maze size, step size, search and cadence. Arm config configs/distance-layout-arms-20261110.json, environment configs distance-layout{13,23,37}-20261110.json. These are supplementary robustness checks, not nine independent maze layouts (only three layouts).

Also fit selected tuned adjacency and dynamics on the same original baseline streams used by the earlier18fixed-data controls (six additional fixed-data fits). Encoder learning rate and sample order remain fixed; regularizer changes intentionally prevent exact baseline-weight reproduction for tuned dynamics.
