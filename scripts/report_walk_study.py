"""All-seed exploration/efficiency plots and readable tables."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('results',type=Path,nargs='+');p.add_argument('--out',type=Path,required=True);p.add_argument('--compact-curves',action='store_true');a=p.parse_args()
reports=[r for f in a.results for r in json.loads(f.read_text())];runs=[r for report in reports for r in report['runs']];groups=[]
for name in dict.fromkeys(r['name'] for r in runs):
 g=sorted([r for r in runs if r['name']==name],key=lambda x:x['seed']);groups.append(dict(name=name,objective=g[0]['objective'],step=g[0]['step_size'],layout=g[0]['layout'],n=len(g),seeds=[r['seed'] for r in g],coverage=[r['coverage_k4'] for r in g],auc=[r['auc_k4'] for r in g],fine=[r['coverage_k8'] for r in g],times95=[r['milestones'].get('k4_95') for r in g],capped95=[r['milestones'].get('k4_95',r['interactions']) for r in g],resets=[r['reset_count'] for r in g],updates=[r['updates'] for r in g]))
colors={'walk':'#26866c','dynamics':'#4175b2','adjacency':'#dc8a35','random_queue':'#999999'}
steps=sorted({r['step'] for r in groups});height=max(5,max(sum(g['step']==s for g in groups) for s in steps)*.32+1.3)
fig,axes=plt.subplots(1,len(steps),figsize=(8*len(steps),height),squeeze=False)
for ax,step in zip(axes[0],steps):
 gg=sorted([g for g in groups if g['step']==step],key=lambda g:np.mean(g['auc']))
 for i,g in enumerate(gg):
  ax.barh(i,np.mean(g['auc'])*100,color=colors[g['objective']],alpha=.65)
  ax.scatter(np.array(g['auc'])*100,np.full(g['n'],i)+np.linspace(-.12,.12,g['n']),color='#222',s=15,zorder=3)
 ax.set(yticks=range(len(gg)),yticklabels=[g['name'].replace(f'-s{step}','') for g in gg],xlabel='Coverage AUC (%) — higher is faster exploration',xlim=(0,100),title=f'Maximum move {step}')
 ax.tick_params(axis='y',labelsize=8);ax.grid(axis='x',alpha=.2)
fig.suptitle('Random walks, reset controls and learned frontiers • Every seed shown')
fig.tight_layout();fig.savefig(str(a.out)+'.png',dpi=160);plt.close(fig)
fig,axes=plt.subplots(1,len(steps),figsize=(7*len(steps),5),squeeze=False)
for ax,step in zip(axes[0],steps):
 selected=[g for g in groups if g['step']==step]
 if a.compact_curves:selected=[g for g in selected if not any(t in g['name'] for t in ['reset-policy-noreset','openloop-policy-blockturn','-p0.0-r0','random-queue','-p0.98-r10000'])]
 for index,g in enumerate(selected):
  curves=[]
  for r in [r for r in runs if r['name']==g['name']]:
   rows=[json.loads(s) for s in (Path(r['run'])/'progress.jsonl').read_text().splitlines()];curves.append([x['coverage_k4']*100 for x in rows])
  label=g['name'].replace(f'-s{step}','');style='-'
  if a.compact_curves:
   if '-base-' in g['name']:label=g['objective'].capitalize()+' frontier';style=':'
   elif g['objective'] in ['dynamics','adjacency']:label='20k walk → '+g['objective'];style='--'
   elif g['name'].startswith('continue-'):label='Two early resets, keep walking'
   elif g['name'].startswith('noreset-policy-reset'):label='Persistent walk, reset every10k'
   elif 'turn0' in g['name']:label='Persistent walk, no feedback'
   else:label='Persistent walk, no resets'
  ax.plot(np.array([x['interactions'] for x in rows])/1000,np.mean(curves,axis=0),label=label,linestyle=style,color=plt.get_cmap('tab10')(index%10))
 ax.set(title=f'Maximum move {step}',xlabel='Actions (thousands)',ylabel='Quarter-bin occupancy (%)',ylim=(0,102));ax.grid(alpha=.2);ax.legend(fontsize=7)
fig.tight_layout();fig.savefig(str(a.out)+'-curves.png',dpi=160);plt.close(fig)
Path(str(a.out)+'.json').write_text(json.dumps(groups,indent=2)+'\n')
lines=['# Walk study: every seed','', 'Coverage and AUC are percentages. Time to95 is censored at100k; a dash means the target was not reached.','']
for g in groups:
 lines += ['## '+g['name'],'','| Seed | Final coverage | Fine coverage | Coverage AUC | Actions to95 | Random resets | Updates |','|---|---:|---:|---:|---:|---:|---:|']
 for i,s in enumerate(g['seeds']):
  lines.append(f"| {s} | {100*g['coverage'][i]:.2f} | {100*g['fine'][i]:.2f} | {100*g['auc'][i]:.2f} | {g['times95'][i] if g['times95'][i] is not None else '—'} | {g['resets'][i]} | {g['updates'][i]} |")
 lines+=['',f"Mean final: {np.mean(g['coverage'])*100:.2f}%; mean AUC: {np.mean(g['auc'])*100:.2f}%; mean capped time to95: {np.mean(g['capped95']):.0f} actions.",'']
Path(str(a.out)+'.md').write_text('\n'.join(lines)+'\n')
for g in groups:print(g['name'],'final',round(np.mean(g['coverage'])*100,2),'AUC',round(np.mean(g['auc'])*100,2),'capped95',round(np.mean(g['capped95'])))
