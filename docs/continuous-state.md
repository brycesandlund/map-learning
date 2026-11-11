# Continuous-coordinate study — COMPLETE

All authorized experiments, analyses, plots and documentation are complete. Do not launch additional jobs from this handoff.

Main report: `docs/continuous-study.md`. Protocol/staged decisions: `docs/continuous-study-protocol.md`. Run log and README updated. Consolidated results: `docs/assets/continuous-results.json`; accounting: `docs/assets/continuous-validation-accounting.json`.

All six stages completed successfully: `runs/continuous-{screen,confirm,layouts,ablation,smallstep-followup,long}-20261111`. Total 87 jobs, 9.3M executed collection actions, 379,608 updates, 96,972,000 presentations/head. Six long jobs rerun identical 100k prefixes and are not new independent replications. Separate smoke: 15k actions, 70 updates, 17,560 presentations/head. No experiment workers remain active. No heartbeat was created for this study.

Normal maximum move .25: both losses reach all 144 rooms and the 95% quarter-bin evaluation target in all 15 validation runs per method. Fresh-seed mean target times: dynamics62,715 / adjacency55,320. New layouts:62,149 /60,977; adjacency is not consistently faster across layouts. Coordinate-only/frozen-clean controls are fast, frozen-noisy and random-queue controls poor.

Maximum move .125: original100k runs all miss95%; extra1000updates help depending on placement/objective but no full repair. At200k, dynamics90.02/91.58/77.52%; adjacency99.96/95.62/96.40%. Adjacency reaches95%3/3; dynamics0/3. All six extended prefixes match original replay and encoder/dynamics weights exactly. Target attainment is not complete continuous coverage.

99 tests pass. Initial hashes, shared action streams, budgets, replay vector bounds, sampled physical transitions, input xy and independent coverage recounts audited. New local links/JSON and git diff checks pass. Validation, ablation and long-run charts plus both100-frame GIFs inspected. Continuous default remains dynamics; old discrete defaults/checkpoints supported. No commits made by this agent; external commits during work were preserved. No preexisting experiment data overwritten.
