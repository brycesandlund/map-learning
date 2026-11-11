# Joint objective study — COMPLETE (November 11, 2026)

All 246 full jobs finished successfully. No active experiment workers or automation were created for further work. Main report: docs/joint-loss-study.md. Protocol: docs/joint-loss-protocol.md. Run log updated in training_runs.md; README links the implementation and report.

Implemented native hybrid dynamics/adjacency training, independent D/A/R/L/m controls, diagnostics and tests. Defaults unchanged. 102 tests pass. Preserved preexisting uncommitted work; no commit made. Original continuous-study edits belong to earlier work and remain intact.

126 tuning jobs (47 distinct small-step settings; four accidental duplicate jobs retained but not counted as evidence), 84 fresh confirmation jobs, 36 additional-layout jobs. Every run: 100k actions, 4,112 updates, 1.05M presentations/head. Total: 24.6M actions, 1,011,552 updates, 258.3M presentations/head. Audits cover physics, occupancy, budgets, matched initialization/action/noise streams, exact duplicates and saved checkpoint prefixes.

Fresh small-step coverage: dynamics m1 50.2%; adjacency m1 47.9%; hybrid A.03/m1 67.7%; selected hybrid A.3/m.25 62.8%; adjacency m.125 72.7%; adjacency m.25 74.0%; dynamics m.25 67.0%. R=L=.03, D=1 when present, A=.3 for pure adjacency. Margin reduction improves both pure methods 6/6 on fresh small steps, 6/6 normal-step completion times, and 6/6 additional-layout runs. Hybrid loses to matched pure adjacency 6/6 fresh and 5/6 additional layouts. No small-step run reaches 95% by 100k; all normal-step fresh runs do. Do not claim full continuous coverage.

Recommended next working setting: pure adjacency A=.3,R=L=.03,m=.25, with dynamics m=.25 comparator. Frozen selection/configs: configs/joint-loss-selected-20261111. No further tuning after unblinding. Original-margin A=.03 hybrid was not tested on additional layouts.

Mechanism: all 12 fresh pure-objective m1/.25 pairs are bit-identical through 25k; replay diverges around 27k–35.5k as smaller hinge cutoff becomes active. Smaller margins improve physical centroid geometry despite worse noise/transition ratio. Static rescaling preserves queue order/function-class objectives up to m²; optimizer trajectories differ. L>=2R gives a collapse global optimum. Do not equate scaled norm or centroid projections with a proved causal geometry mechanism.

Assets: joint-loss-{confirmation,confirmation-curves,layouts,tuning,margin}.png, joint-loss-seed60.gif (100 frames through 100k, seed fixed before outcomes); all result/probe/effect/audit JSON files under docs/assets. Main accounting joint-loss-accounting.json. Stages under runs/joint-loss-{mix,scale,margin-extension,confirm-controls,confirm-mix,confirm-selected,layout-controls,layout-selected}-20261111. Do not overwrite these immutable roots. Scripts in report reproduce studies and artifacts.
