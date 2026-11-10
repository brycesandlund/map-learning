# COMPLETE — overnight study finished November 10, 2026

All55 runs and audits completed without failures. Final synthesis: `docs/overnight-findings.md`; appendix: `docs/horizon-experiments.md`. All9 duration replay prefixes verified, all12 matched-control streams verified, final tests48/48 pass, git diff --check clean. All plots and GIF still frames visually inspected. Overnight heartbeat `overnight-maze-experiments` deleted; temporary caffeinate process4599 stopped. No more work is queued. Changes left uncommitted as requested scope did not include committing. Historical working notes follow.

# Overnight experiment handoff — 2026-11-09

User authorized overnight autonomous experimentation: sweep horizons (beyond 32 if model rollouts remain useful), quantify latent rollout fidelity, compare shared versus independent ensemble batches, iterate and report meaningful results. Repo `/Users/brycesandlund/GitHub/map-learning`. Do not discard preexisting uncommitted work; no commits requested. No agents requested; do not spawn subagents.

## Running work

40 runs total, all 10k actions. Main horizon factorial discount .99, 24 MCTS sims, full epoch every ten actions, horizons 4/8/16/32 × shared/independent × seeds 0/1/2 (24 runs). Matched random streams: epoch or new+20 × shared/independent × three seeds (12 runs). Horizon 64 extension × shared/independent × seeds 0/1 (4 runs). Dynamics/encoder architecture and losses unchanged.

| Root under runs/ | Manifest under configs/ | Exec session | Workers |
|---|---|---|---|
| horizon-pilot-20261109 | horizon_pilot.json | 39376 | 3 |
| horizon-confirm-20261109 | horizon_confirm.json | 87264 | 6 |
| ensemble-matched-20261109 | ensemble_matched.json | 20744 | 3 |
| horizon-long-20261109 | horizon_long.json | 58605 | 4 |

Postprocessing sessions: 12996 (pilot + matched), 37535 (confirm), 86683 (long). These poll complete runs, writing `overnight-audit.json` (nearest-state retrieval, rollout fidelity to depth128, imagined-vs-real candidate reward ranking, map stability). They exit once all matching runs are audited. `summary.json` and per-run `.log` files provide progress. Use OS process inspection if session handles no longer available. Do not rerun active directories! Sweep command skips completed runs but deliberately rejects partial directories. Investigate failures before retrying with a fresh directory. Existing completed runs remain valid.

Full epoch independent controls were verified bit-identical to old random-epoch replay arrays AND model/dynamics weights for all three seeds. Independent RNG behavior preserved by drawing all per-head indices before optionally broadcasting head0 indices; repulsion RNG stays matched. Shared full epochs synchronize shuffled minibatches; both modes visit every transition per head. New+20 also shares replay minibatches in shared mode.

48 tests pass (includes new shared-batch/RNG alignment and perfect-dynamics rollout-audit tests). Full suite `.venv/bin/python -m pytest -q`. MPLCONFIGDIR=/tmp/map-reward-mpl for rendering. Intermediate checkpoints every2000 actions and reward maps every200 actions. Checkpoints save AFTER training; reward snapshots BEFORE next action.

## What remains

1. Monitor runs and failures, finish all 40 and postprocessing. Long H64 runs are expensive (~0.5 s/action under heavy contention as of22:45 local); expect throughput to improve as other jobs finish. Do not cut them short merely to save time.
2. Read `docs/horizon-experiments.md` (currently PARTIAL, generated) and individual `overnight-audit.json` files. Refresh with:

```
MPLCONFIGDIR=/tmp/map-reward-mpl .venv/bin/python scripts/report_horizon.py runs/horizon-pilot-20261109 runs/horizon-confirm-20261109 runs/ensemble-matched-20261109 runs/horizon-long-20261109
```

It writes docs/assets/horizon-results.json and .png, tables in docs/horizon-experiments.md. It verifies exact replay equality across matched random controls. Do not hand-edit conclusions into this generated file until final refresh (or add a separate `docs/overnight-findings.md`).

3. Adaptive followups are authorized and desirable. Depending on outcomes, consider H16/32 with96 MCTS simulations (24 fixed simulations may limit search), batched shooting as an explicitly labeled alternative, H64 third seed if promising, or H128 only if fidelity and actual gains support it. Keep any comparisons controlled; do not change representation/loss arbitrarily. If no exploration gain, report significant negative finding with evidence rather than hunting cherry-picked seeds. Main batch/horizon study already has3 seeds on one maze; H64 only2. Budget available until roughly07:00 local Nov10; this is a stop-to-report preference, not an excuse to leave jobs orphaned.
4. Quantify on/off rails using final audit AND some intermediate checkpoints, especially best vs short horizon. For history:
```
.venv/bin/python scripts/audit_latent_rollouts.py RUN1 RUN2 --intermediate
```
H4 seed0 independent/shared histories already exist. Checkpoint histories cost only seconds. Final replay trajectories always within same episode; cutoff prevents using future data. Curves include teacher-forced vs open-loop MSE, exact nearest-state retrieval, physical error, nearest-manifold distance normalized by median edge length, imagined/real-state disagreement ratio, and known-edge-prefix fraction. Replay paths are all seen edges; persistent-random/greedy imagined-reward paths start at visited states and can go unseen.
5. Generate repo animations with counts and rewards for informative conditions. `scripts/animate_rewards.py` supports arbitrary horizons; exact enumeration bars omitted aboveH5 and labeled accordingly. Existing H4 batch pilot GIF/PNG `/tmp/horizon-batch-pilot.gif` / `.png` are optional; not yet copied to repo. Final select H4/H16/H32/H64 appropriate mode and/or matched epoch/new20 shared-ind to illustrate stability. Use fresh --out paths; renderer refuses overwrite. View still PNG and GIF frames to check legibility. Plot generator `scripts/report_horizon.py` is ready.
6. Finish concise interpretation in docs, update training_runs.md from in-progress to completed with actual findings and followups; link README to report. Report results honestly: genuine gains vs persistent random59.8% prior3-seed mean; horizon4 original .9 discount must not be compared as pure-horizon effect to new .99. No cross-run raw latent disagreement as calibrated universal uncertainty. Strong shared-batch agreement alone does not prove correctness.
7. Inspect git diff --check and changed-code tests; no need rerun unchanged broad suite repeatedly. Final reply should show plot/animation, key findings (coverage, rollout limits, shared batches) with report link. Pause/remove overnight heartbeat when complete; do not leave recurring experiments indefinitely.

## Early findings (NOT final conclusions)

Old full-epoch H4/.9 seed0 model on known replay sequences: exact retrieval depth4 93.8%,16 77.1%,32 67.7%,64 53.1%,128 47.9%; mean position error at128 only3.65 discrete moves. Imperfect but not immediate runaway.
New H4/.99 seed0 coverage independent54.8%, shared57.0%; seen accuracy94.3% vs94.1%. Shared map shape more stable in this online pair, but data differs, so wait matched controls.
Candidate path ranking on these new H4 final models: at32 selected/best true-state return52% independent,67% shared; selected imagined/real return~3×. At64/128 ranking worsens but remains finite. Model selection optimism is important. A “true-state return” here is frozen-ensemble disagreement on actual states, not real environmental novelty. Candidate diagnostic is random shooting, not the agent's actual MCTS plans. Need distinguish.

Heartbeat created: `overnight-maze-experiments`, attached to this chat, every20 minutes for24 occurrences. Pause it after completion via automation_update. A bounded8-hour `caffeinate -i -t 28800` process keeps idle sleep from stopping the overnight computation.

## Follow-up at23:08 local

Matched controls all12 completed and audited; replay equality verified by report. Shared batches reduce mean-normalized map L1 changes0.447→0.374 for full epochs and0.583→0.460 for new+20; rank correlation rises0.758→0.835 and0.582→0.748. Seen accuracy slightly lower (~1pp) with sharing, so stability does not equal better prediction. H4/H8 full3seed results show no clear exploration advantage: independent53.1/57.0%, shared52.0/53.9%, random59.8%. H16/H32/H64 not all finished.

Launching controlled search-budget follow-up (six10k runs, seeds0/1/2): H16 independent MCTS96 simulations vs original24; H32 independent random shooting256 sequences, one deterministic particle, as an explicitly different batched search strategy. Both .99 discount, full epochs, same model/learning. These test whether fixed24 MCTS simulations limit longer-horizon search. Manifest `configs/horizon_search_followup.json`, root `runs/horizon-search-20261110`. Include this root in final report/postprocessing. No need to start new variants until these and the horizon64 runs resolve.

Followup sweep session50607 (6workers), audit session75406. Matched-data conclusions documented in `docs/ensemble-batch-findings.md`, animation in `docs/assets/ensemble-batches.gif`/`.png`. Rendering plus H16 seed0 history audits in session95716; check completion. This heartbeat notified the meaningful completed batch finding, so do not repeat it unless tying into final results. Shared encoder sees less diverse examples per update with shared batches; documented caveat.

Session95716 completed successfully: batch GIF and H16 seed0 intermediate audits saved.

## Follow-up at23:30 local

Still healthy; no failures. All H4/H8/H16 main conditions now3seeds; H32/H64 and MCTS96 ongoing. Shooting256 H32 completed3seeds, coverage42.8% (51.1,43.9,33.4), not a gain. H16 independent mean45.8%, shared58.2%; do not declare monotonic horizon benefit. Report refreshed including `runs/horizon-search-20261110`.

Added detailed audit protocol/failure case in `docs/latent-rollout-audits.md`. Enhanced `scripts/audit_reward_ranking.py` with median/max optimism, ratio-of-sums, and maximum latent radius relative to actual-state range; verified all previous statistics unchanged. Fresh extended audit files exist for all3 shooting runs, separate from their original `overnight-audit.json`. Shooting seed2 shows actual runaway at128: largest selected latent radius383× actual max; ratio-of-sums optimism77,909×. Other2seeds mild. This does NOT establish online128 failure because agentH32; it does justify NOT blindly extending to128. ExistingH64 runs should finish. No new runs launched this heartbeat.

Final aggregate reporter currently reads original overnight-audit ranking fields; for robustness/outlier interpretation use separate reward-ranking-audit.json. Consider re-running enhanced ranking script on H64 final checkpoints when they finish. See /tmp/horizon-behavior-partial.json for initial stationary/action-switch diagnostics; generate fresh final file if needed.

## Follow-up at23:50 local: main horizon factorial complete

All40 original runs completed/audited. H64 independent2seed mean47.7%, shared54.5%; H32 independent48.6%, shared51.8%. H16shared58.2% and H8independent57.0% are strongest learned3seed configurations but neither exceeds random59.8%. MCTS96 H16 still running at~8.6k; no clear gain yet. Rendering sharedH4/H16/H32/H64 animation `docs/assets/horizon-rewards.gif`/`.png` in session13460. EnhancedH64 ranking and H32/H64seed0 intermediate audits in session49475.

Because a10k interaction budget could conceal later benefits as the model fits more data, launching a controlled duration follow-up:30k actions ×3seeds for H8independent, H16shared, and persistent-random full-epoch control. These are the strongest configurations selected from the10k sweep, so selection must be disclosed. Manifest `configs/horizon_duration.json`; root `runs/horizon-duration-20261110`. All other settings unchanged; reward maps every600 actions, checkpoints every6000 (keeps50frames, boundedstorage). Nine additional runs,55total including sixsearch-budgetfollowups. Do not extend to online128 given documented extrapolation failure. Wait for duration results before finalizing; no additional broad sweep needed.

For duration runs audit snapshot stability implementation currently hardcodesframes>=5000 but averages whatever intervals recorded; DO NOT compare their temporal-stability numbers with200-step snapshot runs. The matched-data stability conclusion uses original10k runs only. Final report generator needs to exclude random_epoch_30k from equal-length replay assertion and matched-data stability table; current code compares all randomcontrols acrosslengths and will fail. Fix this before includingduration root. Also graphmain horizon plot should continueusing10k conditions only; duration results getseparate plot/table. Verify new30k runs' first10k replay prefixes against source10k runs to demonstrate matched initial learning, though checkpoints6000 don'tsave10000.

Duration sweep session12453 (6workers), audit session21979. Reporter now excludes duration controls from10k matched-equality/stability checks; verifies30k prefixes against original10k runs, produces separate `horizon-duration.png`, and uses enhanced reward-ranking-audit.json if available. Include duration root when generating final report.

All session49475 audits and session13460 horizon animation completed; still PNG and aggregate chart visually inspected and legible. H64 enhanced ranking shows no large radius runaway (all selectedmaxradius<actualmax), but only~32–54% selected/best return bydepth64; this is misleading prediction/ranking without gross norm explosion. Duration runs proceed; no need to launch more conditions. Finalize when nine duration runs and remaining threeMCTS96 runs complete, then delete heartbeat per current developer instructions (not merely pause).

## Follow-up at00:08 local

All46 ten-thousand-action runs (original40 +6search followups) completed and audited. H16 independent MCTS96 coverage byseed0/1/2 =47.87/47.98/43.95%, mean46.60%, vs45.77% at24simulations. This is a small inconsistent gain (one seed improves, two fall), not a search-budget rescue. Shooting256 H32 mean42.79%. Duration runs healthy: sixlearned30k runs at~13–17k; randomcontrolsqueued. No failures, no new conditions. Await duration completion.

## Follow-up at00:50 local: duration yields a fitting improvement

H8independent30k runs completed3seeds and audited; prefix assertions pass. Coverage81.6/83.9/92.3% (mean85.9%); seen-edge accuracy98.25/99.35/99.13% (mean98.91%). Other6durationrunsstillactive; randomcontrols at~13k initially this heartbeat, H16shared~28k. No final exploration comparison yet.

Added `scripts/audit_training_duration.py` to compare10k and30k models on IDENTICAL first10k replay paths (exactprefix checked). All3 H8 results saved `early-replay-audit.json`. Same-path exact retrieval at32 imagined steps rises44.1→89.9%; at16 60.8→95.1%,64 33.7→73.6%,128 23.6→57.6%. Meanphysicalerror at32 falls1.40→0.36 gridsteps. Additional data AND training change, so do not attribute solely to training or claim generalization onunseenroutes. This is a strong fitting/rollout improvement. Notify user once thisheartbeat; don'trepeatuntilfinal.

When H16shared/random30k finish, run `audit_training_duration.py` for those6 as well. Include matched-path curves in final report alongside unconditioned final-dataset audits. This suggests adapting horizon as model matures may be a future direction; no new runs launchedyet. Prioritize finishing55-run report rather than unbounded additional sweeps.
