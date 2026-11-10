"""Summarize the horizon / ensemble-data factorial and rollout audits."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('roots',nargs='+',type=Path);p.add_argument('--out',type=Path,default=Path('docs/horizon-experiments.md'));args=p.parse_args()
    rows=[]
    for root in args.roots:
        for r in json.loads((root/'summary.json').read_text()) if (root/'summary.json').exists() else []:
            run=Path(r['run']);audit=run/'overnight-audit.json'
            if not audit.exists():continue
            r['audit']=json.loads(audit.read_text());
            if (run/'reward-ranking-audit.json').exists():r['audit']['ranking']=json.loads((run/'reward-ranking-audit.json').read_text())
            r['config']=json.loads((run/'config.json').read_text());rows.append(r)
    groups=defaultdict(list)
    for r in rows:groups[r['variant']].append(r)
    def avg(rs,fn):return float(np.mean([fn(r) for r in rs]))
    lines=['# Horizon and ensemble-data experiments','',f'{len(rows)} completed, audited runs. All use the same maze layout, 10,000 real interactions and center resets every 2,000 actions unless noted. Three initialization/action seeds are used for the main factorial.','',
        'Joint MSE dynamics + 0.01 repulsion, width 256 × three layers, eight unbounded latents, three independently initialized heads. Full replay epoch every ten actions for horizon experiments. Shared batches synchronize transition indices across heads; independent batches permute each epoch separately. Both see every replay transition once per epoch. Random-control streams isolate sampling from agent-induced changes in data.','',
        'Main horizon sweep: discount 0.99, 24 MCTS simulations per action, 10% persistent-random epsilon. Horizon changes hold simulation count fixed, not total model evaluations or wall time. The prior horizon-4 baseline used discount 0.9, so compare horizons within this sweep.','',
        '## Exploration and fitting','',
        '| Condition | Seeds | Coverage mean (range) | Coverage AUC | Seen next-state accuracy | Seen disagreement | Collection minutes |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name,rs in sorted(groups.items()):
        c=[r['coverage_lattice_stats']['final'] for r in rs]
        lines.append(f"| {name} | {len(rs)} | {np.mean(c):.1%} ({min(c):.1%}–{max(c):.1%}) | {avg(rs,lambda r:r['coverage_lattice_stats']['auc']):.1%} | {avg(rs,lambda r:r['audit']['retrieval']['seen']['exact_next_position_accuracy']):.1%} | {avg(rs,lambda r:r['metrics']['oracle_seen_disagreement']):.3g} | {avg(rs,lambda r:r['collection_seconds'])/60:.1f} |")
    lines+=['','## Rollout faithfulness','',
        'Final frozen models, 96 sampled reset-free replay trajectories per run. Every edge on these paths has been observed. Open-loop mean predictions are decoded by nearest latent among all 1,784 reachable states; the oracle catalogue is evaluation-only. Accuracy is exact coordinate retrieval, not latent-distance thresholding. Position error is Euclidean distance in units of one 0.25-unit action and ignores walls.','',
        '| Condition | Exact @4 | @16 | @32 | @64 | @128 | Mean position error @32 | @128 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    def curve(r,mode,d):return next(x for x in r['audit']['rollout']['curves'][mode] if x['depth']==d)
    for name,rs in sorted(groups.items()):
        vals=' | '.join(f"{avg(rs,lambda r:curve(r,'replay',d)['nearest_state_accuracy']):.1%}" for d in [4,16,32,64,128])
        lines.append(f"| {name} | {vals} | {avg(rs,lambda r:curve(r,'replay',32)['position_error_steps_mean']):.2f} | {avg(rs,lambda r:curve(r,'replay',128)['position_error_steps_mean']):.2f} |")
    lines+=['','## Does imagined reward select useful candidate paths?','',
        '16 visited starts × 64 persistent-random candidate sequences. Each sequence is scored with the same frozen ensemble twice: once along imagined mean latents and once along encoded true maze states reached by its actions. No true transitions enter planning. “Real-state return” here means model disagreement evaluated at true states, not environmental novelty or exploration gain. Selected/best measures the true-state return of the imagined-best candidate divided by the best candidate in that same bank.','',
        '| Condition | Selected/best @4 | @16 | @32 | @64 | @128 | Random/best @32 | Optimism @32 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    def rankrow(r,d):return next(x for x in r['audit']['ranking']['rows'] if x['horizon']==d)
    for name,rs in sorted(groups.items()):
        vals=' | '.join(f"{avg(rs,lambda r:rankrow(r,d)['selected_real_return_over_best']):.1%}" for d in [4,16,32,64,128])
        lines.append(f"| {name} | {vals} | {avg(rs,lambda r:rankrow(r,32)['random_real_return_over_best']):.1%} | {avg(rs,lambda r:rankrow(r,32)['selected_imagined_over_real_return']):.2f}× |")
    lines+=['','## Reward-map stability on identical random trajectories','',
        'Adjacent snapshots every 200 actions in the second half of training. Spatial L1 compares maps divided by their own mean (lower is more stable); rank correlation and top-decile Jaccard ignore overall reward scale (higher is more stable). This separates shape changes from latent/replay-normalization scale changes. Maps include all state-action pairs.','',
        '| Condition | Spatial L1 change | Rank correlation | Top-decile overlap | Raw scale log-change |',
        '|---|---:|---:|---:|---:|']
    for name,rs in sorted(groups.items()):
        if rs[0]['config']['planner']['behavior']!='random' or rs[0]['config']['collect_steps']!=10000:continue
        vals=' | '.join(f"{avg(rs,lambda r:r['audit']['stability'][k]):.3f}" for k in ['spatial_relative_l1_change','spatial_rank_correlation','top_decile_jaccard','raw_scale_log_change'])
        lines.append(f'| {name} | {vals} |')
    # Audit that the shared/independent controls really received identical streams.
    controls=[r for r in rows if r['config']['planner']['behavior']=='random' and r['config']['collect_steps']==10000]
    verified=0
    for r in controls:
        others=[x for x in controls if x['seed']==r['seed'] and x['variant']!=r['variant']]
        with np.load(Path(r['run'])/'replay.npz') as a:
            for other in others:
                with np.load(Path(other['run'])/'replay.npz') as b:
                    for k in ['actions','observations','next_observations','positions','next_positions']:np.testing.assert_array_equal(a[k],b[k])
        verified+=1
    lines+=['',f'Verified identical replay arrays across all available random-control conditions for each seed ({verified} runs).','',
        '## Duration follow-up', '', 'The strongest 10k learned conditions (H8 independent, H16 shared) were selected for a 30k-action follow-up against persistent random, three seeds each. This adaptive selection is exploratory. Every completed 30k run is checked against its original 10k replay prefix. Reward snapshots are 600 rather than 200 actions apart, so duration-run stability statistics are excluded from the matched-data stability comparison above.', '', '![Duration follow-up](assets/horizon-duration.png)', '', '## Limitations and reproduction','',
        'Three seeds on one maze are exploratory evidence, not a broad benchmark. Final-checkpoint audits describe the final model; intermediate checkpoint audits are separate artifacts. Full replay training still weights repeated transitions by visit frequency. Shared data do not make the heads a calibrated posterior. Long-horizon random candidate banks diagnose model ranking, not exact MCTS optimality. Nearest-latent decoding can conceal or exaggerate model error; raw latent MSE, teacher-forced error, off-manifold distance and known-prefix fractions are retained in JSON.','',
        'Raw runs include configurations, source hashes/snapshots, checkpoints every 2,000 actions, replay, and reward snapshots. Each `overnight-audit.json` contains fitting, rollout, candidate-ranking and temporal-stability measurements. Run manifests in `configs/horizon_*.json` and `configs/ensemble_matched.json` with `scripts/run_probabilistic_sweep.py`; audit with `scripts/postprocess_horizon.py`; regenerate this report with `scripts/report_horizon.py`.','']
    duration=[r for r in rows if r['config']['collect_steps']==30000]
    for r in duration:
        seed=r['seed'];name=r['variant'].removesuffix('_30k')
        if name=='random_epoch':old=Path('runs/ensemble-matched-20261109')/f'epoch_independent-seed{seed}'
        else:old=Path('runs')/('horizon-pilot-20261109' if seed==0 else 'horizon-confirm-20261109')/f'{name}-seed{seed}'
        with np.load(old/'replay.npz') as a,np.load(Path(r['run'])/'replay.npz') as b:
            for key in ['actions','observations','next_observations','positions','next_positions']:
                np.testing.assert_array_equal(a[key],b[key][:len(a[key])])
    asset=args.out.parent/'assets';asset.mkdir(exist_ok=True)
    fig,ax=plt.subplots(figsize=(8,4.5))
    for name in sorted({r['variant'] for r in duration}):
        rs=[r for r in duration if r['variant']==name];curves=[]
        for r in rs:
            log=[json.loads(s) for s in (Path(r['run'])/'exploration.jsonl').read_text().splitlines()]
            steps=[x['environment_step'] for x in log];curves.append([x['visited_positions']/1784*100 for x in log])
        values=np.array(curves);line,=ax.plot(steps,values.mean(0),label=f'{name} (n={len(rs)})')
        ax.fill_between(steps,values.min(0),values.max(0),color=line.get_color(),alpha=.12)
    ax.set(xlabel='Real interactions',ylabel='Visited states (%)',title='30k follow-up: mean and seed range')
    ax.axvline(10000,color='gray',ls='--');ax.grid(alpha=.2)
    if duration:ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(asset/'horizon-duration.png',dpi=160);plt.close(fig)
    (asset/'horizon-results.json').write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n')
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for mode,color in [('independent','#3676ac'),('shared','#e78032')]:
        selected=[r for r in rows if r['variant'] in [f'h{h}_{mode}' for h in [4,8,16,32,64,128]]]
        horizons=sorted({r['config']['planner']['horizon'] for r in selected})
        means=[]
        for h in horizons:
            rs=[r for r in selected if r['config']['planner']['horizon']==h]
            yy=[r['coverage_lattice_stats']['final']*100 for r in rs];means.append(np.mean(yy))
            axes[0,0].scatter([h]*len(yy),yy,color=color,alpha=.4,s=15)
        axes[0,0].plot(horizons,means,'o-',color=color,label=mode)
    axes[0,0].axhline(59.8,color='gray',ls='--',label='Persistent random (prior 3-seed mean)')
    axes[0,0].set(xlabel='Planning horizon',ylabel='Final visited states (%)',title='Exploration: dots are individual seeds');axes[0,0].legend(fontsize=8)
    for name,rs in sorted(groups.items()):
        if name not in ['h4_independent','h32_independent','h32_shared','h64_independent','h64_shared']:continue
        depths=[1,4,8,16,32,64,128]
        axes[0,1].plot(depths,[avg(rs,lambda r:curve(r,'replay',d)['nearest_state_accuracy'])*100 for d in depths],'o-',label=name)
        depths=[4,8,16,32,64,128]
        axes[1,0].plot(depths,[avg(rs,lambda r:rankrow(r,d)['selected_real_return_over_best'])*100 for d in depths],'o-',label=name)
    axes[0,1].set(xlabel='Imagined rollout depth',ylabel='Exact state retrieval (%)',title='Known replay paths: compounding prediction error');axes[0,1].legend(fontsize=8)
    axes[1,0].set(xlabel='Candidate path horizon',ylabel='Selected / best real-state return (%)',title='Does imagined reward rank paths correctly?');axes[1,0].legend(fontsize=8)
    names=[n for n,rs in sorted(groups.items()) if rs[0]['config']['planner']['behavior']=='random' and rs[0]['config']['collect_steps']==10000]
    axes[1,1].bar(range(len(names)),[avg(groups[n],lambda r:r['audit']['stability']['spatial_relative_l1_change']) for n in names],color=['#3676ac' if 'independent' in n else '#e78032' for n in names])
    axes[1,1].set_xticks(range(len(names)),[n.replace('new_then_replay','new+20').replace('_','\n') for n in names],fontsize=8)
    axes[1,1].set(ylabel='Mean-normalized map L1 change',title='Matched-data reward volatility (lower is better)')
    for ax in axes.flat:ax.grid(alpha=.2)
    for ax in [axes[0,0],axes[0,1],axes[1,0]]:ax.set_xscale('log',base=2)
    fig.tight_layout();fig.savefig(asset/'horizon-results.png',dpi=160);plt.close(fig)
    lines.insert(4,'![Results](assets/horizon-results.png)\n')
    args.out.write_text('\n'.join(lines));print(args.out)

if __name__=='__main__':main()
