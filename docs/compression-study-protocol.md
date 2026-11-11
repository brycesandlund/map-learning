# Compression and ancestry prediction — November 11, 2026

User authorized L1, finite-precision latent channels and multi-step prediction for both current dynamics and adjacency encoders. No policy/imagination changes or Atari claim. No subagents. Preserve all existing work and immutable run directories.

## Primary design, frozen before outcomes

100k actions, small maximum movement .125, layout7, eight nuisance inputs, persistent raw-observation frontier/archive and exact restores. Same encoder/dynamics architecture, initialization, real controls, successor-noise streams and causal replay cadence as the prior study: 4,112 optimizer updates and 1.05M one-step transition presentations/head. Regularization anchor R=L2=.03, margin m=.25; include the original m=1 baselines. Pure adjacency weight A=.3; dynamics D=1. Two tuning seeds70/71. Select by mean final quarter-bin occupancy, occupancy AUC tie-breaker, then freeze before inspecting new seeds80–85. Retain every failed setting. Multi-step runs have extra differentiable edge presentations/compute; count them explicitly, and include H1 controls.

Initial92 jobs:32 regularizer arms×2seeds plus14 ancestry arms×2seeds. No full Cartesian claim. Manifests configs/compression-20261111/{regularizers,multistep}-arms.json; each intervention has a separate immutable spec.

L1: E sum_j |f(o)_j|, weights .0001/.0003/.001/.003/.01 added to L2, plus .001/.003/.01 replacing L2. Compare both base objectives. This penalizes per-observation amplitudes, not guaranteed removal of entire latent coordinates. L1 has a locally collapse-favoring linear scale term; failures are informative.

Fixed-noise channel: encoder is explicitly bounded coordinatewise by .25*tanh(raw), and training uses f(o)+N(0,sigma²I). Ratios sigma/m=.01/.03/.1/.3. Always keep search/archive encoding deterministic. Noise is added to the source for dynamics/adjacency; targets are clean, jointly differentiated encodings. Noise is also added independently to both random latents for hinge repulsion. L1/L2 act on clean outputs. Compare bounded/noiseless controls and a task-only-noise control at ratio .1 with clean repulsion. Source-only Gaussian noise in squared adjacency changes its expected loss only by a constant; any real population objective change there comes from nonlinear noisy repulsion, not a denoising prediction bottleneck. Dynamics actually has a learned decoder after the noisy channel. This distinction must appear in reporting.

Multi-step: both base objectives retain their existing one-step training. Add an action-conditioned latent rollout endpoint loss with weights .1/1 and maximum horizons1/4/16; add H32 at weight .1. Random lags are uniform1..min(H,depth). Choose up to32 anchors from the current one-step minibatch using a separate RNG; use occurrence parent IDs to reconstruct real action sequences. Feed each head its own predicted latent at subsequent steps; no teacher forcing. Endpoint and source encoders both differentiate. Pure adjacency gains a predictive auxiliary objective; H1 isolates adding dynamics from extending horizon. Its old one-step dynamics remains detached from encoder. Multi-step targets are observed outcomes, not shortest-path distances. Sibling transitions and chronological replay neighbors must never become invented paths.

## Validation and follow-ups

Exact baseline equivalence at zero intervention; exact checkpoint continuation including channel/path RNGs; branching ancestry path/action tests; antithetic adjacency-noise gradient identity; physics/occupancy/budget/paired-stream audits. Instrument L1, norm, repulsion, active hinge fraction, path lengths, extra trajectory/edge presentations. Use real-point animations and multi-seed plots, plus nuisance/geometry/sparsity diagnostics. Candidates must beat tuned m=.25 controls, not just old m=1. After complete initial screen, do bounded coefficient/interaction follow-ups only if justified, record before launch, and confirm on untouched seeds plus additional layouts. Do not select on confirmation outcomes. Update training_runs.md and complete report before ending.

A key-dependent environment is not implemented in this maze. Longer-horizon losses can preserve a key distinction once its consequences appear in replay, but cannot guarantee the frontier discovers those consequences when representations initially alias key/no-key. Do not interpret this experiment as solving that exploration feedback problem.

## Blind controls launched during screen

To overlap independent work, run m=.25 pure-objective baselines on fresh seeds80–85 (layout7,smallstep) and layouts13/23×seeds86–88. Do not inspect outcomes before candidate selection is frozen; only completion/failure status. These data cannot influence tuning.

## Bounded follow-up declared before fresh outcomes

First-seed L1>=.0003 collapses; .0001 remains noncollapsed but underperforms the tuned baseline. Add L1=1e-5/3e-5, L1=.0003 with repulsion .1, and delayed L1=.0003/.001 beginning after50k actions and ramping linearly over10k. Test both objectives and both tuning seeds. This distinguishes early optimization trapping from compression after a map forms. Add bound1/noiseless and bound1/sigma.025 to separate the smaller tanh bound from noise, plus bound.25/sigma.025 with R=.1 to test repulsion compensation. Total16arms×2seeds, fixed before launch; no further wide tuning extension. Default schedule fields preserve prior updates exactly (test included).

## Narrow delayed-noise mechanism check

On seed70, dynamics bounded/noiseless reaches71.7%, sigma=.0025 reaches26.6%, sigma=.0075 collapses to4.7%. Add only these two noise levels introduced after50k with a10k linear ramp, for both base objectives and seeds70/71 (8jobs). This checks compressing an established representation versus inhibiting early learning, analogous to the L1 delay. Candidate freeze must include this stage; fresh controls remain blind. No more coefficient expansion planned.

## Final lower-noise boundary check

Even the lowest immediate noise sigma=.0025 is substantially worse than the bounded/noiseless dynamics control on seed70. Rather than conclude about a sweep whose best nonzero value is its lower boundary, add sigma=.00025/.00075, bound.25, both objectives and tuning seeds (8jobs). This is the final noise grid extension; freeze after all five tuning stages. Total140tuning jobs,70settings. The delayed-noise check remains a separate intervention. All tuning remains on seeds70/71; fresh outcomes are still blinded.

## Reporting choices, before confirmation outcomes

Use seed80 for representative animations, one four-panel animation per base objective (base, selected L1, selected noise, selected multi-step). Show matched controls in quantitative tables/plots; animation is a single fixed seed. Rollout audit uses the same ancestry paths from the corresponding adjacency-base fresh run for each model seed, horizons1/4/16/32; report encoded-target decoding floor and decoded-start and physical-stay baselines. Uniform physical decoder probes are evaluation only.

## Multi-step freeze,20:44UTC, before fresh outcomes

All28multi-step jobs complete. Select dynamics H32/weight.1 (80.38/77.99%,mean79.19), adjacency H16/weight1 (79.56/81.77%,mean80.66). Confirm each plus its H1/same-weight control on seeds80–85. These24jobs can overlap remaining regularizer tuning; KEEP their outcomes blind until all families freeze. Full selector will assert these settings unchanged and omit duplicate confirmations. Additional-layout winner remains undecided until all tuning completes.

## Full candidate freeze,20:57UTC, before unblinding

All140tuning jobs complete. Dynamics: L1=3e-5 immediate; bounded B=.25 noise sigma=.0025 delayed50k/ramp10k; predictive H32/weight.1. Adjacency: L1=.0003 delayed50k/ramp10k; same bounded delayed noise; predictive H16/weight1. Include original tuned baseline, bounded/noiseless and H1/same-weight controls. Six remaining fresh arms×6seeds=36jobs, reusing already-frozen24MS/H1jobs and12baselinejobs. Layout winners from tuning only: dynamics L1 and adjacency H16, plus H1 control;18newjobs reuse12baselinejobs. No coefficient retuning after this point. Exact configs/ranking: configs/compression-selected-20261111. Fresh outcomes may now be inspected.

## Read-only follow-up after confirmation

All 242 training runs complete; no new training or retuning. Adjacency H16 loses to base/H1 on both extra layouts in every paired seed. Apply the same fixed common-ancestry forecast audit and 25k/100k noise/geometry probes to all 30 layout models to distinguish forecast failure from exploration-geometry failure. This is a post-outcome diagnostic, not an independently prespecified hypothesis test. Both audits find improved H16 forecasts despite reduced exploration. Derived all-arm reproduction manifests are in configs/compression-reproduction-20261111; original manifests remain unchanged.
