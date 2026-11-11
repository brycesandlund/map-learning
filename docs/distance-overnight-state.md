# Distance-objective study — COMPLETE

**Stop: all authorized work is complete. Do not launch additional experiments.**

Final report: docs/distance-study.md. Exact formulas/decisions: docs/distance-objectives.md. Training log updated. Final results: docs/assets/distance-study-results.json. 141 online runs of 100k actions + 24 fixed-data fits, all successful; 93 tests passed. All 27 extra-layout checks finished; they reverse the apparent efficiency advantage of adjacency on the original maze. All 15 adjacency validation runs complete, but the new-layout mean is 54.5k actions versus 49.2k for original dynamics. Main result is a viable simpler objective, not a consistent speed winner.

All experiment workers finished. Validation/geometry figures and both GIFs inspected; local links/JSON/diff checks passed. No baseline-default changes, commits, overwrites of previous runs, or modifications of preexisting user work. Heartbeat is confirmed PAUSED in its saved automation configuration; if an already queued wakeup fires, stop without launching work. Automation ID: overnight-distance-objective-maze-experiments.

## Historical working notes (superseded by completion above)

# Distance-objective overnight study — live handoff

Started Nov 10 ~23:14 PST; aim report by Nov 11 07:00 PST. User authorizes autonomous implementation, experiments and adaptive follow-ups replacing dynamics with distance-centric losses. No subagents authorized. Repository initially clean; preserve all subsequent work. Write permission granted for this turn.

Heartbeat: `overnight-distance-objective-maze-experiments`, every 20 minutes, max 24 occurrences. Pause when complete. Initial setup in progress; no runs launched yet. Do not duplicate launches. Read this file plus git status on continuation.

## Fixed protocol

Noisy 12x12 maze, 1784 reachable quarter-cell positions, 8 iid Gaussian nuisance inputs plus xy. Persistent raw-observation frontier/visited archive, simulator restores, no dedup or oracle state identities in search/training. Queue chooses maximum nearest-visited Euclidean latent distance. All 5 actions expanded. Encoder 10→256×3→8, SiLU; native deterministic initialization. Matched native dynamics+L2 baseline needed (legacy initialization differs). Encoder LR 9e-5; batch256; repulsion .03; L2 .03 unless a declared arm changes it. Interleaved100 cadence: train every100 expansions/500 actions, apportion one full replay epoch across each5000-action block; 100k actions=4112updates,1.05M transition presentations. No resetting of learned model/frontier.

Temporal pairs must follow actual ancestry. Frontier pop order is not consecutive motion. Do not use physical position equality or resampled-noise pairs for training. Physical graph and controlled noise samples are evaluation only.

## Planned screen

Euclidean adjacency attraction + repulsion/L2; QRL-inspired constrained expansion (Euclidean, upper-bound edge constraint, explicitly an adaptation); temporal distance stress using ancestry path lengths (observed path ≠ shortest path); t-SNE-inspired Student-t affinity loss with observed-neighbor positives (not original t-SNE). Match encoder/dynamics initialization and environment/action budgets. Dynamics heads for distance arms can be trained on detached encodings strictly as diagnostics, never affect encoder/search. Screen seeds18–20; reserve21–23 for confirmation. Select follow-ups based on completed screen, record decision before running. Baseline and loss coefficients need explicit immutable manifest.

## Remaining

Implement and test, launch immutable sweep, record sessions; inspect failures and completed results; fresh-seed confirmation; matched-data geometry/noise/queue-ranking diagnostics; coverage/latent plots and exploration animations; docs report and training_runs.md. Preserve all results, label adaptations and censor failed completion times. Never infer shortest-path learning from trajectory regression or agreement from correctness.

## 23:23 PST implementation update

Implemented `distance_learning.py`: adjacency, qrl_upper, temporal, affinity; detached auxiliary dynamics, explicit occurrence ancestry. Runner supports `--distance-spec`. Full tests91 passed, including ancestry/no fictitious reset edges, auxiliary gradient isolation, serialization and native matched initialization. Smoke temporal running in tool session30433, output `runs/distance-smoke-20261110/temporal`. Screen driver `scripts/run_distance_study.py`:12arms × seeds18/19/20,100k,4workers. Exact initial formulas/protocol in docs/distance-objectives.md. Launch screen only after smoke completes.

Screen launched successfully: tool session24636, driver PID22919; immutable `runs/distance-screen-20261110/manifest.json`. All36 jobs queued,4parallel. Smoke completed2500actions/10updates. No failures so far. `fit_distance_replay.py` available for identical-stream followups; do not launch before selecting completed screen evidence.

## Adaptive decision recorded (~23:34 PST)

Initial screen still running. 12 follow-up arms frozen in configs/distance-followup-20261110.json; rationale and exact formulas in docs/distance-objectives.md. Temporal_upper added (one-sided temporal stress), legacy branches unchanged; focused tests running session recorded in tool history. Do not launch follow-up until initial screen finishes to avoid excess CPU contention. Command: `.venv/bin/python scripts/run_distance_study.py --out runs/distance-followup-20261110 --workers 4 --arms configs/distance-followup-20261110.json`.

Available new analysis scripts: analyze_distance_study.py (checks budgets and matching init hashes); fit_distance_replay.py and run_distance_fixed.py (identical-stream controls; 5k native reproduction EXACT encoder+dynamics); audit_distance_geometry.py (full8D queue ranking, controlled-noise geometry, optional frozen128-expansion×3 continuations); audit_temporal_targets.py (oracle evaluation only). Geometry smoke baseline25k completed and inspected JSON. Set MPLCONFIGDIR=/tmp/map-learning-mpl and XDG_CACHE_HOME=/tmp/map-learning-cache for plotting. sysctl/ps unavailable sandbox; use logs/status files and recorded sessions, no need escalation.

## Initial screen complete; follow-ups launched (23:38 PST approx)

All36/36 screen runs exited0. Budgets and exact initial parameter hashes audited. Final initial summary: docs/assets/distance-screen-results.json; plot distance-screen.png. Baseline full coverage46884/51192/55908; adjacency-w.1:47808/46949/56134 (comparable, not established improvement). Adjacency-w1:57506/61876/58096. QRL-c.05:73387/82584/73987. All temporal equality arms failed to finish; H4 mean final6.61%,H16 10.15%,H64 48.56%. Affinity-l2.03:53037/50432/59006; l2.003 slower. No failures.

Running:
- Follow-up36runs: `runs/distance-followup-20261110`, tool session49754,4workers, driver.pid inside. Seeds18–20,12adaptive arms frozen earlier. All100k.
- Common-data18fits: `runs/distance-fixed-20261110`, tool session42150,2workers. Source each seed's completed native dynamics baseline; families dynamics-l2,adjacency-w.1,qrl-c.05,temporal-H16,H64,affinity-l2.03. Identical observations/batches/init and4112updates per fit. Source replay is not copied; metadata gives source. Audits/geometry should reconstruct the SOURCE replay, not expect replay.npz in fixedfit folders. Nativecontrol MUST exactly reproduce source weights at100k (enforced).

Next: inspect these as complete; audit common-data25k/50k with audit_distance_geometry.py and frozen continuations; choose best tuned families after followup, freeze choices then native baseline+2–3distance arms on fresh seeds21–26. Optional fresh maze layouts if results justify. Build animations via render_distance_exploration.py (uses saved frames, reset-safe). Initial screen source code changed only to add an unused new objective during its last jobs; existing branches mathematically unchanged, snapshots recorded. Temporal target diagnostics initial JSON still partial; rerun into a final new filename after all screen complete. Focused tests9pass after temporal_upper addition;91full prior to addition.

## 23:49 PST status and additions

Fixed18/18 fits complete, all controls exactly reproduce full100k encoder+dynamics weights. First24 audits complete; remaining12 (seed20) running tool session60444 in runs/distance-fixed-audits-20261110. Audit script resume-safe; names with decimal coefficients handled by appending extensions, not Path.with_suffix. Common-data audits:25k/50k, frozen128-expansion×3 replicates at25k only. Earlier audit session17402 completed24 tasks; all successful.

Additional9native dynamics controls RUNNING: runs/distance-dynamics-controls-20261110, session78382,2workers. Frozen config configs/distance-dynamics-controls-20261110.json, rationale docs/distance-objectives.md. Needed because stronger relative prediction vs regularizers could explain apparent loss-family gains. Screen seeds18–20 only. First two seeds reg-half completes43571/43529 vs original46884/51192.

Planned but NOT YET LAUNCHED:9QRL-no-L2 controls, configs/distance-qrl-unregularized-20261110.json. Test whether norm regularization counteracts distance maximization. Command `.venv/bin/python scripts/run_distance_study.py --out runs/distance-qrl-unregularized-20261110 --workers 3 --arms configs/distance-qrl-unregularized-20261110.json`. Launch after current follow/control workers free capacity. Still need fresh confirmation21–26; no fresh runs launched yet.

Geometry finding (common replay): at50k successful native/adjacency/affinity encoders approximately2D, Euclidean physical-distance Spearman~.99; QRL remains~1D, temporalH16 strongly encodes noise (noise/move~100–180). Equal physical gap(.5 units) across a wall versus open passage has latent distance ratio~1 even when true geodesic ratio8–11. Thus exploration success is not evidence of shortest-path geometry. Scripts audit_distance_barriers.py + report_distance_geometry.py. Current plot docs/assets/distance-geometry-initial.png inspected, but regeneration needed after title clarified that surfaces are noise-averaged centroids (noise is NOT shown by surface). At50k seed18 source is already fully explored, so its queue panels all0productive; use25k for ranking figures and explicitly distinguish completion.

No follow-up conclusions yet; screen followup29/36 complete at this update, all successful. Do not overclaim one-seed tuning gains. Full initial temporal audit now docs/assets/distance-temporal-targets.json (9runs).

## Midnight Nov11 update (08:00 UTC)

All screening complete: initial36 + adaptive36 + native-weight controls9 + QRL-no-L2 controls9 =90online100k runs. No failures. Initial common-data18fits and36geometry/ranking audits complete. Prediction diagnostic36audits complete in docs/assets/distance-prediction-results.json: probes fresh noise on balanced seen/untried physical edges, nearest centroid decoding; NOT exact replay memorization. Final wall-detour audit36models in docs/assets/distance-barrier-results.json. Full test suite93passed.

Fresh confirmation RUNNING: runs/distance-confirm-20261110, session84091,3workers;24runs (4arms×seeds21–26);19finished at last check. Choices frozen before results in configs/distance-confirm-20261110.json: original native, reg-half native, adjacency-w.3/l2.03, affinity-l2.03. Do not retune on these. QRL no-L2 all3variants AUC~.44–.46, below advancement criterion .6444, so none enters confirmation. QRL run session69886 completed9; no-L2 source results plotted docs/assets/distance-qrl-unregularized-results.json/.png.

Tuned common-data fits RUNNING: runs/distance-tuned-fixed-20261110, session64046,2workers;6fits (reg-half dynamics and selected adjacency on original source baseline streams18–20).2done at last check. Script now checks exact source reproduction ONLY when config train/model/maze/seed match, so tuned controls intentionally skip exact weight assertion. After fits finish run run_distance_audits.py into runs/distance-tuned-fixed-audits-20261110, then prediction/barrier diagnostics if useful.

Extra layouts predeclared: seeds13/23/37 × agent seeds27/28/29 × original native/reg-half native/selected adjacency =27runs. Configs distance-layout{13,23,37}-20261110.json and distance-layout-arms-20261110.json. Layout13 RUNNING: runs/distance-layout13-20261110, session85760,1worker. Layout23 and37 NOT YET LAUNCHED. Commands same driver with their config, --seeds27 28 29, respective fresh output dirs. Wait for confirmation/fixed workers to free capacity before launching both. Keep total training processes about6.

Animation rendering RUNNING session61210: docs/assets/distance-exploration-seed21.gif, four frozen confirmation methods, first fresh seed (preselected, not cherry-picked). Inspect sampled frames. Consider supplemental other seeds; main report not yet written. Need readable final comparison, method formulas/protocol linked, geometry noise-centroid caveat, training_runs.md, README, accounting, final tests if further edits, pause heartbeat.

Still no report/completion notification. Fresh best-loss comparison may be close to retuned dynamics; do not turn one-seed gains into a winner claim. Significant already: adjacency-only encoder objective matches dynamics-level exploration; temporal equality targets are badly mismatched; successful geometry is nearly ordinary physical distance, not shortest-path detours.
