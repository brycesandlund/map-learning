"""Report novelty/error/disagreement diagnostics; seeds are aggregation units."""
from pathlib import Path
from collections import defaultdict
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LABELS={'epoch_independent':'Epoch independent (10k)','epoch_shared':'Epoch shared (10k)',
 'new_then_replay_independent':'New+20 independent (10k)','new_then_replay_shared':'New+20 shared-all (10k)',
 'new20_shared_replay_only':'New+20 shared replay (10k)','h8_independent_30k':'H8 independent (30k)',
 'h16_shared_30k':'H16 shared (30k)','random_epoch_30k':'Random / epoch (30k)'}


def mean(rs,fn):
    vals=[fn(r) for r in rs];vals=[v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('runs/novelty-audit-v2-20261110'));args=p.parse_args()
    rows=[json.loads(f.read_text()) for f in sorted(args.root.glob('*.json'))]
    assert len(rows)==72,f'Expected 72 checkpoint audits; got {len(rows)}'
    groups=defaultdict(list);history=defaultdict(list)
    for r in rows:
        history[r['group']].append(r)
        if 'paths' in r:
            assert r['paths']['starts']==64 and r['paths']['candidates']==128
            groups[r['group']].append(r)
    for seed in range(3):
        matched=[r for r in rows if r['seed']==seed and 'paths' in r and r['cutoff']==10000]
        for family in ['iid','persistent']:
            for i in range(5):
                for key in ['random_unique_new_states','best_unique_new_states','starts_with_novel_candidate','starts_with_reachable_novelty']:
                    assert len({r['paths']['families'][family][i][key] for r in matched})==1
    asset=Path('docs/assets');(asset/'novelty-results.json').write_text(json.dumps(rows,indent=2)+'\n')
    names=list(LABELS)
    names=[n for n in names if n!='new_then_replay_shared']
    def e(r,scope,score):return r['edges']['groups'][scope]['new_successor_auc'][score]
    def path(r,family,h=32):return next(x for x in r['paths']['families'][family] if x['horizon']==h)
    lines=['# Does prediction error or disagreement identify novel exploration?','',
      '72 saved-checkpoint audits across 24 runs, three seeds per condition. No retraining or new agent experience. Novelty labels use only the replay prefix available at the checkpoint; simulator geometry supplies evaluation labels, never agent inputs.','',
      '## Findings','',
      '**Prediction error and ensemble disagreement both contain information about novel exploration.** At the learned agents’ 30k checkpoints, latent error identifies transitions into new states with AUROC 0.80–0.83, and disagreement with AUROC 0.82–0.84. This remains above chance when considering only untried actions at visited states. The earlier near-chance result used a different label: whether an action had ever been tried, not whether it enters a new state.','',
      'This association is useful but incomplete. At a frontier, the largest-disagreement action enters novelty only 41–44% of the time, versus 23–24% for a random action. About 88% of transitions into new states are nevertheless predicted correctly by nearest-state retrieval: increased latent error does not necessarily imply a wrong physical successor.','',
      'On matched 10k random-behavior data, full epochs give a stronger disagreement–novelty association than new-once + 20 replay updates (AUROC 0.67 versus 0.56, independent batches). Sharing replay batches changes the latter to 0.58; after restricting to untried actions, the values are essentially unchanged (0.606 versus 0.607). The earlier improvement in temporal reward stability therefore does not establish a corresponding improvement in novelty discrimination. Path-selection results are mixed.','',
      'Long-path diagnostics separate three potential limitations: candidate paths may never reach new territory, accumulated reward may rank an unproductive path highest, and imagined rollouts may distort that ranking. All three occur in these diagnostics; their relative importance varies across checkpoints. They do not establish the causal contribution of each limitation to the online MCTS runs.','',
      '## One-step novelty','',
      'Here “novel” means the action enters a previously unvisited physical state. AUROC ranks novel against non-novel successors; 0.5 is chance. Rows below consider actions whose **source state was already visited**. This avoids confusing distant unvisited regions with choices the agent could make at a known state. Errors are ensemble-mean latent MSE or nearest-latent-decoded physical position error.','',
      '| Condition | Disagreement→novelty AUROC | Latent error→novelty AUROC | Physical error→novelty AUROC | Disagreement↔latent error Spearman |','|---|---:|---:|---:|---:|']
    for name in LABELS:
        rs=groups[name]
        vals=[mean(rs,lambda r:e(r,'visited_source',score)) for score in ['disagreement','latent_error','physical_error']]
        rho=mean(rs,lambda r:r['edges']['groups']['visited_source']['disagreement_error_spearman'])
        lines.append(f'| {LABELS[name]} | '+' | '.join(f'{v:.3f}' for v in vals)+f' | {rho:.3f} |')
    lines+=['','The earlier action-history label and the actual novelty label give different answers:','',
      '| Condition | Disagreement→untried action AUROC | Disagreement→new successor AUROC (seed range) |',
      '|---|---:|---:|']
    for name in LABELS:
        rs=groups[name]
        vs=[e(r,'visited_source','disagreement') for r in rs]
        untried=mean(rs,lambda r:r['edges']['untried_edge_auc_at_visited']['disagreement'])
        lines.append(f'| {LABELS[name]} | {untried:.3f} | {min(vs):.3f}–{max(vs):.3f} |')
    lines+=['','Physical mistake rates use the nearest latent in the full maze catalogue; the true successor is available solely for evaluation. These are ensemble-mean predictions, not a requirement that all individual heads be correct.','',
      '| Condition | Wrong successor when entering a new state | Wrong successor when entering a visited state |',
      '|---|---:|---:|']
    for name in LABELS:
        rs=groups[name]
        novel=mean(rs,lambda r:r['edges']['groups']['visited_source']['physical_mistake_rate_novel'])
        familiar=mean(rs,lambda r:r['edges']['groups']['visited_source']['physical_mistake_rate_non_novel'])
        lines.append(f'| {LABELS[name]} | {novel:.1%} | {familiar:.1%} |')
    lines+=['','A second control compares only **untried actions at visited states**, so “untried” itself cannot explain the separation:','',
      '| Condition | Disagreement AUROC | Latent-error AUROC |','|---|---:|---:|']
    for name in LABELS:
        lines.append(f'| {LABELS[name]} | {mean(groups[name],lambda r:e(r,"untried_at_visited","disagreement")):.3f} | {mean(groups[name],lambda r:e(r,"untried_at_visited","latent_error")):.3f} |')
    lines+=['','## A useful score is not a reliable direction','',
      'A frontier state is visited but has at least one action entering an unvisited state. Compare a uniform random action with the maximum-disagreement or maximum-error action, averaging states uniformly then seeds equally. Ties are broken uniformly in expectation. This is an evaluation of a greedy one-step selector, not the online MCTS policy.','',
      '| Condition | Frontier states / seed | Random enters novelty | Max disagreement | Max latent error |','|---|---:|---:|---:|---:|']
    for name in LABELS:
        rs=groups[name];fn=lambda r:r['edges']['direction']['frontier']
        lines.append(f'| {LABELS[name]} | {mean(rs,lambda r:fn(r)["n"]):.0f} | {mean(rs,lambda r:fn(r)["random_toward_rate"]):.1%} | {mean(rs,lambda r:fn(r)["disagreement"]):.1%} | {mean(rs,lambda r:fn(r)["latent_error"]):.1%} |')
    lines+=['','Directional scores at 2–4, 5–16, and 17+ actions from the closest unvisited state are retained in JSON. These use true graph distances as labels only. Some early far-distance strata contain very few states, so they should not be overinterpreted.','',
      '## Does accumulated reward select paths that explore?','',
      'For each final model: 64 uniformly sampled visited starts × 128 candidate paths, horizons 4/8/16/32/64. Two path banks: IID actions and persistent actions (80% probability to keep the previous action). Every candidate is executed in the true maze for evaluation. Outcome is **number of distinct previously unvisited states encountered**, counting revisits only once. All three scoring rules use the same candidate bank:','',
      '- Real-state disagreement: discounted disagreement evaluated along the actual states reached by that candidate. This uses privileged true transitions and is diagnostic, not a deployable planner.',
      '- Real-state error: discounted latent prediction MSE on those actual transitions; also privileged diagnostic information.',
      '- Imagined disagreement: discounted disagreement along recursive ensemble-mean latent predictions—the same kind of imagined reward used by our planner.',
      '', 'Scores use discount 0.99; novelty count is undiscounted. Selection maximizes one score per start; uniform tie breaking. Random means selecting a candidate uniformly from the same bank. The candidate banks are not actual MCTS trees and the models do not learn during these hypothetical paths.','']
    for family in ['iid','persistent']:
        lines += ['',f'### {family.upper()} paths, horizon 32','',
          '| Condition | Random novel states | Max real disagreement | Max real error | Max imagined disagreement | Best novelty in bank |',
          '|---|---:|---:|---:|---:|---:|']
        for name in LABELS:
            rs=groups[name]
            vals=[mean(rs,lambda r:path(r,family)['random_unique_new_states'])]
            vals += [mean(rs,lambda r:path(r,family)['scores'][key]['selected_unique_new_states']) for key in ['real_disagreement','real_error','imagined_disagreement']]
            vals += [mean(rs,lambda r:path(r,family)['best_unique_new_states'])]
            lines.append(f'| {LABELS[name]} | '+' | '.join(f'{v:.2f}' for v in vals)+' |')
    lines+=['','### Within-start path correlations','',
      'Spearman correlation between discounted score and the number of distinct new states, across the 128 persistent candidates at horizon 32. Correlations are computed separately for each start, omitting starts with constant outcomes or scores, then averaged within seed and across seeds. The last columns condition on at least one candidate actually reaching novelty; they measure selecting a zero-novelty path despite that opportunity.','',
      '| Condition | Real disagreement correlation | Real error correlation | Imagined disagreement correlation | Zero novelty: max real disagreement | Zero novelty: max imagined disagreement |',
      '|---|---:|---:|---:|---:|---:|']
    for name in LABELS:
        rs=groups[name]
        vals=[mean(rs,lambda r:path(r,'persistent')['scores'][key]['within_start_spearman']) for key in ['real_disagreement','real_error','imagined_disagreement']]
        zeros=[mean(rs,lambda r:path(r,'persistent')['scores'][key]['zero_novelty_choice_rate_eligible']) for key in ['real_disagreement','imagined_disagreement']]
        lines.append(f'| {LABELS[name]} | '+' | '.join(f'{v:.3f}' for v in vals)+' | '+' | '.join(f'{v:.1%}' for v in zeros)+' |')
    lines+=['','### Candidate generation misses reachable novelty','',
      'Counts pooled over three seeds (192 sampled starts per condition). “Reachable” uses the true shortest distance to any unvisited state; “found” means at least one of the 128 candidate paths visits a new state. The persistent bank keeps its previous action with probability 0.8, otherwise drawing uniformly. Counts measure candidate coverage, before any reward-based selection.','',
      '| Condition, horizon 32 | Starts with reachable novelty | IID bank finds novelty | Persistent bank finds novelty |',
      '|---|---:|---:|---:|']
    for name in LABELS:
        rs=groups[name]
        feasible=sum(path(r,'iid')['starts_with_reachable_novelty'] for r in rs)
        iid=sum(path(r,'iid')['starts_with_novel_candidate'] for r in rs)
        persistent=sum(path(r,'persistent')['starts_with_novel_candidate'] for r in rs)
        lines.append(f'| {LABELS[name]} | {feasible}/192 | {iid}/{feasible} | {persistent}/{feasible} |')
    lines+=['','For H8/H16 at 30k, all 192 sampled starts can reach novelty within 64 actions, yet the IID bank finds such a path from only 121/126 starts, and the persistent bank from 150/160. A longer horizon alone does not make random candidate paths cover reachable destinations. These banks do not reproduce MCTS expansion, so these figures are not an MCTS failure rate.','',
      'At horizon 32 with persistent paths on the matched independent full-epoch models, selecting by real-state disagreement yields 4.95 new states per path versus 1.85 for random selection; imagined disagreement yields 2.88. This illustrates lost reward-ranking utility under imagined rollouts. It is not universal: H8 at 30k yields about 1.57 with either real-state or imagined disagreement.','']
    lines+=['','![Novelty diagnostics](assets/novelty-diagnostics.png)','','Bars show three-seed means; black dots in the upper panels show individual seeds. The lower-left panel divides the mean selected yield by mean random yield. The first four plotted conditions are matched random-behavior models at 10k interactions; H8/H16 are active-agent models at 30k.','',
      '## Interpretation limits','',
      'The statistics are means across three model seeds, not independent samples across thousands of adjacent states. One maze, deterministic observations and exact discrete-state novelty; they do not establish behavior with continuous non-repeating observations. Different active-agent runs have different visited sets; only the matched-random controls hold the data fixed. Checkpoint comparisons change both fitting and the definition of what remains unvisited.','',
      'A state can be unvisited yet predicted correctly. Tiny latent errors can rank novelty well even when nearest-state retrieval stays correct; therefore latent-error association is not equivalent to frequent physical mistakes. AUROC also does not establish that the single best action is novel, nor that navigating to it is easy.','',
      'Path outcomes are counterfactual frozen-model diagnostics. They do not measure closed-loop exploration performance or prove the cause of previous MCTS failures. Real-state-scored paths show what reward ranking could do with exact rollouts, not an achievable policy with the current model. No learning-progress experiment was performed.','',
      'Scores are normalized by one full-catalogue latent-variance scalar per checkpoint. This differs from the online replay-based normalization but cannot change the AUROC, Spearman correlations, or maximizing action/path within that checkpoint. The per-seed values and sample counts, including undefined correlations, are retained in the portable JSON.','',
      '## Reproduction','',
      '`scripts/analyze_novelty.py` with `configs/novelty_matched.json` and `configs/novelty_duration.json`, output `runs/novelty-audit-v2-20261110`; then `scripts/report_novelty.py`. Portable per-checkpoint output: `docs/assets/novelty-results.json`. The initial smaller 24-start/64-candidate pilot is retained separately in `runs/novelty-audit-20261110`; only the expanded 64-start/128-candidate audit is reported here.','']
    Path('docs/novelty-diagnostics.md').write_text('\n'.join(lines))
    fig,axes=plt.subplots(2,2,figsize=(14,10));x=np.arange(len(names));short=['Epoch\nind.','Epoch\nshared','New+20\nind.','New+20\nshared replay','H8\n30k','H16 shared\n30k','Random\n30k']
    for offset,score,label in [(-.14,'disagreement','Disagreement'),(.14,'latent_error','Latent prediction error')]:
        axes[0,0].bar(x+offset,[mean(groups[n],lambda r:e(r,'visited_source',score)) for n in names],width=.28,label=label)
        for j,n in enumerate(names):
            axes[0,0].scatter(j+offset+np.linspace(-.045,.045,3),[e(r,'visited_source',score) for r in groups[n]],s=10,color='black',alpha=.5,zorder=3)
    axes[0,0].axhline(.5,color='gray',ls='--');axes[0,0].set(ylim=(0,1),ylabel='AUROC',title='Does the score identify a novel successor?');axes[0,0].legend(fontsize=8)
    for offset,key,label in [(-.22,'random_toward_rate','Random action'),(0,'disagreement','Max disagreement'),(.22,'latent_error','Max error')]:
        axes[0,1].bar(x+offset,[mean(groups[n],lambda r:r['edges']['direction']['frontier'][key]) for n in names],width=.22,label=label)
        for j,n in enumerate(names):
            axes[0,1].scatter(j+offset+np.linspace(-.035,.035,3),[r['edges']['direction']['frontier'][key] for r in groups[n]],s=10,color='black',alpha=.5,zorder=3)
    axes[0,1].set(ylim=(0,1),ylabel='Probability of entering a new state',title='At a frontier: which action gets selected?');axes[0,1].legend(fontsize=8)
    for offset,key,label in [(-.22,'real_disagreement','Real-state disagreement'),(0,'real_error','Real-state error'),(.22,'imagined_disagreement','Imagined disagreement')]:
        vals=[mean(groups[n],lambda r:path(r,'persistent')['scores'][key]['selected_unique_new_states'])/mean(groups[n],lambda r:path(r,'persistent')['random_unique_new_states']) for n in names]
        axes[1,0].bar(x+offset,vals,width=.22,label=label)
    axes[1,0].axhline(1,color='gray',ls='--');axes[1,0].set(ylabel='Novel-state yield / random candidate',title='32-step persistent paths: reward vs actual novelty');axes[1,0].legend(fontsize=8)
    for name in ['epoch_independent','new_then_replay_independent','new20_shared_replay_only','random_epoch_30k']:
        times=sorted({r['cutoff'] for r in history[name]})
        axes[1,1].plot(times,[mean([r for r in history[name] if r['cutoff']==t],lambda r:e(r,'visited_source','disagreement')) for t in times],'o-',label=LABELS[name])
    axes[1,1].axhline(.5,color='gray',ls='--');axes[1,1].set(xlabel='Recorded interactions',ylabel='Novel-successor AUROC',ylim=(.4,1),title='Novelty ranking at saved checkpoints');axes[1,1].legend(fontsize=8)
    for ax in [axes[0,0],axes[0,1],axes[1,0]]:ax.set_xticks(x,short,fontsize=8)
    for ax in axes.flat:ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(asset/'novelty-diagnostics.png',dpi=160);plt.close(fig)
    print('docs/novelty-diagnostics.md')

if __name__=='__main__':main()
