# How far can the latent model be rolled forward?

This is an evaluation protocol and intermediate finding log. The aggregate final tables live in [the horizon report](horizon-experiments.md).

## Three different questions

1. **Compounding error on known paths.** Start from an encoded replay observation, replay the recorded actions without re-encoding, and compare each imagined state with the encoding of its true successor. Sequences never cross resets. Every transition on these paths was observed before the evaluated checkpoint.
2. **Generalization along new paths.** Start at visited states, then generate persistent-random or greedy imagined-disagreement actions. Execute those same actions in the maze only for scoring. Report the fraction whose entire prefix used observed state-action edges; these paths can leave the training data.
3. **Does planning select the right rewards?** From 16 visited starts, evaluate 64 persistent-random candidate sequences each. Rank candidates by discounted disagreement along imagined latents and compare with disagreement evaluated at the real states reached by the same actions. This diagnoses model-dependent ranking errors independently of the agent's MCTS approximation.

All use frozen encoder/dynamics weights. True maze states and the complete state catalogue enter evaluation only. The candidate bank is not the online MCTS tree; it also differs from the online shooting agent's IID-action bank. These diagnostics cannot alone identify why a particular online action was taken.

## Metrics and their limits

- Exact state retrieval decodes an imagined latent by nearest encoded state among all 1,784 reachable positions. This is a common physical target across encoders, but depends on their learned geometry.
- Physical position error is Euclidean distance in units of one 0.25-unit action. It ignores walls, so a small physical error can still be navigationally important.
- Normalized latent MSE divides by maze-wide latent variance; teacher-forced error re-encodes the true predecessor each step and separates one-step error from accumulated drift.
- Nearest-manifold distance is measured against the same finite catalogue and divided by the median nonzero true-adjacency latent distance. Low distance does not prove correct position: predictions can land near the wrong state.
- Candidate selected/best return compares the true-state disagreement return of the imagined-best candidate with the best in that same bank. It is not exploration gain or an environment reward.
- Optimism compares imagined to true-state disagreement return. Means of ratios can be dominated by outliers; additional median, maximum and ratio-of-sums diagnostics are saved by `scripts/audit_reward_ranking.py`.
- Maximum rollout radius is measured from the mean encoded maze state, normalized by latent standard deviation, then divided by the largest radius among actual encoded maze states. This detects large departures from the encoded range without forcing the representation to stay within a prescribed shape.

## A concrete long-rollout failure

At the final checkpoint of `h32_independent_shoot256-seed2`, the candidate audit becomes dramatically optimistic beyond the agent's training-time planning horizon:

| Audit horizon | Median selected optimism | Ratio of summed imagined / true-state return | Largest selected-path radius / actual-state maximum |
|---|---:|---:|---:|
| 16 | 3.53× | 2.08× | 0.84× |
| 32 | 6.28× | 6.11× | 0.90× |
| 64 | 9.15× | 194.79× | 11.45× |
| 128 | 30.28× | 77,909.21× | 383.41× |

This is an actual latent extrapolation failure, not just a tiny denominator in an uncertainty ratio. The other two shooting checkpoints are much milder: maximum selected-path radius at128 is0.88× and1.20× the actual-state maximum, and ratio-of-sums optimism is2.18× and3.36×. Thus the catastrophic result is one seed, not the typical behavior of every trained model. It also occurs in a post-hoc128-step audit of a32-step agent; do not claim that the online agent planned128 steps.

Based on this evidence, a blind extension to online horizon128 is not justified. The completed64-step online conditions reach47.7% independent /54.5% shared mean coverage, below the59.8% random control. The online shooting condition itself completed all three seeds with42.8% mean coverage, versus59.8% persistent random. Longer candidate searches therefore have not automatically solved exploration.

Run provenance: `runs/horizon-search-20261110/h32_independent_shoot256-seed{0,1,2}/reward-ranking-audit.json`. Added robustness metrics were verified to leave every preexisting candidate-ranking result numerically unchanged.

## More experience and training improve the same known paths

The three H8-independent30k runs reach98.9% mean seen-edge next-state accuracy. To avoid comparing different replay distributions, `scripts/audit_training_duration.py` evaluates the10k and30k checkpoints on the **same first10k replay paths** (96 fixed sequences per seed, prefixes verified exactly).

| Imagined depth | Exact retrieval,10k model | Exact retrieval,30k model |
|---|---:|---:|
| 4 | 85.8% | 98.6% |
| 8 | 71.5% | 97.9% |
| 16 | 60.8% | 95.1% |
| 32 | 44.1% | 89.9% |
| 64 | 33.7% | 73.6% |
| 128 | 23.6% | 57.6% |

At depth32 mean Euclidean position error falls from1.40 to0.36 action steps. This shows that accumulated rollout error is substantially improvable with continued online experience and full-replay training. It does not establish accuracy on unseen routes, and it does not isolate additional optimization from additional collected data. The completed 30k comparison reaches85.9% H8 /85.6% H16-shared coverage versus81.4% random, with mixed per-seed and coverage-AUC results; see the final findings.
