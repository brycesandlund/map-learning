# Latent L2 and noise/search study protocol

November 10, 2026. User authorizes L2 sweep and noise/search investigation. All training uses ordinary noisy replay, no paired-equivalence supervision or control over nuisance inputs.

L = coordinate-mean dynamics MSE +.03 hinge repulsion + λ E||μ(o)||². L2 sums eight coordinates and averages the same 512 replay observations already used for repulsion; it is a latent penalty, not weight decay. Dynamics receives deterministic means. No KL in L2 arms. Sweep λ={0,1e-5,3e-5,1e-4,3e-4,1e-3,3e-3,.01,.03,.1}, seeds9/10/11. Compare noiseless KL β=1e-5 at identical repulsion. Reuse matching prior zero/KL controls, independently reproduce full zero seed9. Same 100k actions,4,112 joint updates,1,050,000 presentations/head, interleaved replay epoch training every500 actions. No preexisting output overwritten.

Select positive λ by mean coverage AUC on seeds9/10/11, then freeze it before confirmation seeds15/16/17 against λ=0 and noiseless KL. Also fit selected L2 on the same β=0 source replay streams used by the previous mechanism study (source6/7/8, model9/10/11) to compare with existing common-data mean/no-KL and mean/KL encoders. This separates data-collection feedback from representation effects.

Noise diagnostics use fixed models and identical raw physical archives at 50k and 75k. Test stored observations, fresh nuisance only on queue, only on visited anchors, or on both; measure top100 overlap, productive ranking, norm/priority correlation, nuisance residuals and top-ranked false positives. These are evaluation-only conditional counterfactuals: resampling also removes archive selection bias and assumes controllable simulator noise, not a proposed general training method. Independently resampled nuisance dimensions remain N(0,1).

Frozen 640-action continuations compare stored inputs, fresh queue inputs, fresh queue+anchors, known-nuisance removal, and radial latent clipping at95th-percentile visited radius. Clipping uses only latent statistics and is a diagnostic of the large-norm-outlier explanation, not a recommended threshold or selected policy. Known nuisance removal is privileged evaluation. All preprocessing persists for new observations during continuation. Use three matched rollout-noise replicates; no learning during these tests. More ranking-only resamples assess stability. Report all arms, censored completions, and seed variation; do not treat noise replicates as independent trained seeds.

## Completed adaptive decisions

All screening arms completed before selecting λ=.03 by the stated coverage-AUC criterion. The selection was saved to `docs/assets/l2-noise-selection.json` before confirmation seeds 15/16/17 were launched. Selected-L2 common-data fits ran alongside confirmation; no coefficients changed after inspecting confirmation. The selected model received the same predeclared noise interventions as the controls at both checkpoints. A descriptive nearest-memory identity audit was added after observing the queue instability, to distinguish entry churn from physical-state confusion. No additional coefficient was selected from those diagnostics.

All training, confirmation, frozen evaluations, and audits completed. See [the final report](l2-noise.md) for results, assumptions, budgets, and reproduction commands.
