# Continuous study — running

User authorized thorough continuous-coordinate maze experiments. Repo has preexisting uncommitted distance-study changes; preserve them. No subagents. No new heartbeat needed. Write permission granted this turn.

Protocol: docs/continuous-study-protocol.md. New runner scripts/continuous_frontier.py; scheduler scripts/run_continuous_study.py. Shared code changes: model config continuous_action_dim default0, dynamics takes real2D controls when2, replay stores vector actions. Legacy tests pass. New config configs/continuous_frontier.json, separate from default. 98 tests passed after implementation.

Completed initial12runs under runs/continuous-screen-20261111 (all success). Three seeds30–32, dynamics/adjacency/oracle-xy/random-queue. Learned both100% quarter-bin occupancy at100k; finer bins98.8–99.5%. Oracle much faster; random queue stays nearorigin. JSON docs/assets/continuous-screen-results.json. Curves and100-frame GIF docs/assets/continuous-screen.{png,gif}, inspected. Smoke2×5000actions completed.

Active schedulers:
- confirmation: session37331, runs/continuous-confirm-20261111,12runs, seeds33–38, two learned arms,2workers.
- new layouts: session44092, runs/continuous-layouts-20261111,18runs, layouts13/23/37×seeds39–41×2arms,2workers.
- ablations: session78637, runs/continuous-ablation-20261111,24runs, seeds30–32×8arms: two learned clean, two IID proposals, two halfsteps, frozen-noisy, frozen-clean;2workers.

All roots immutable manifests, per-job logs/status, source snapshots, replay/config/checkpoints. Do not overwrite or duplicate. Monitor status files and progress.jsonl; analyze only when complete.json exists. Main36? total planned66runs:12+12+18+24=66, of which54trainable and12diagnostic/untrained. Check actual accounting at end.

Audits completed first two mainseeds via session27917, runs/continuous-audits-main-20261111. Thirdseed session25041, runs/continuous-audits-third-20261111. Fixed uniform physical probes+freshnoise, early25k/final100k; not replay accuracy, labels evalonly. At100k noise/open-transition RMS approx.49dynamics,.34adjacency firstseed. Smooth centroid vs xy distance rank~.9, not a perfect shortest-path map. No need overinterpret small sample.

Remaining: monitor all stagedruns, inspectespecially halfstep/cleancontrols; justifiedfollowups only. Auditbudgets/initialization/physics and results. Produce consolidated resultJSON, validation/ablationplots, optionalfailureanimation ifany, final report and training_runs.md/README. Check98tests plus any addedtests. Final message with actualcontinuous endpointanimation, clearfinite-resolution coverage and resets/noise assumptions. No claim of covering infinite states or improved imaginedplanning.

Smaller-step follow-up launched: session84859, runs/continuous-smallstep-followup-20261111,15runs (two objectives × early/distributed1000extra updates + oracle halfstep) × seeds30–32,2workers. First halfstepseed partialcoverage, predeclared in protocol. Extra-update10smoke completed5000actions/30updates/7560presentations. Total planned81mainruns. No settings of already running arms changed.

Last planned stage: runs/continuous-long-20261111, session recorded in chat,6originalhalfstep runs to200k, seeds30–32×2losses,2workers. Verify first100k exactreplay/weights vs ablation originals. Finalmain count87runs,9.3M actions; trained66? recheck accounting. Stop after these unless genuine new unresolved blocker.
