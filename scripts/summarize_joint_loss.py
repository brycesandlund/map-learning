"""Compact sweep summaries. Partial mode is for monitoring, never final inference."""
import argparse,json
from pathlib import Path
import numpy as np

def collect(root,partial=False):
 manifest=json.loads((root/'manifest.json').read_text());groups=[];runs=[]
 for arm in manifest['arms']:
  group=[]
  for seed in manifest['seeds']:
   p=root/f"{arm['name']}-seed{seed}"
   if not (p/'result.json').exists():
    if partial:continue
    raise FileNotFoundError(p)
   r=json.loads((p/'result.json').read_text());cfg=json.loads((p/'config.json').read_text())
   assert r['updates']==sum((i*5000+255)//256 for i in range(1,manifest['blocks']+1))
   assert r['samples_per_head']==sum(i*5000 for i in range(1,manifest['blocks']+1))
   lines=[json.loads(s) for s in (p/'progress.jsonl').read_text().splitlines()]
   train=[json.loads(s) for s in (p/'training.jsonl').read_text().splitlines()]
   r.update(effective_coefficients={'D':0. if r['objective']=='adjacency' else cfg['train'].get('dynamics_weight',1.),'A':r['adjacency_weight'] if r['objective']=='adjacency' else cfg['train'].get('adjacency_weight',0.),'R':cfg['train']['repulsion_weight'],'L':cfg['train']['latent_l2'],'m':cfg['train']['repulsion_margin']},name=arm['name'],run=str(p),config=cfg,auc=float(np.mean([x['coverage_k4'] for x in lines])),
      tail_metrics={k:float(np.mean([x[k] for x in train[-10:]])) for k in ['repulsion','repulsion_active_fraction','random_pair_distance_mean','latent_squared_norm']},
      restricted95=r['milestones'].get('k4_95',r['interactions']))
   runs.append(r);group.append(r)
  if group:
   groups.append({'arm':arm,'n':len(group),'mean_final':float(np.mean([r['coverage_k4'] for r in group])),
      'final':[r['coverage_k4'] for r in group],'mean_auc':float(np.mean([r['auc'] for r in group])),
      'times95':[r['milestones'].get('k4_95') for r in group],
      'mean_restricted95':float(np.mean([r['restricted95'] for r in group])),
      'mean_norm':float(np.mean([r['tail_metrics']['latent_squared_norm'] for r in group]))})
 return {'root':str(root),'manifest':manifest,'groups':groups,'runs':runs,'partial':partial}

def main():
 p=argparse.ArgumentParser();p.add_argument('roots',type=Path,nargs='+');p.add_argument('--out',type=Path);p.add_argument('--partial',action='store_true');a=p.parse_args()
 reports=[collect(root,a.partial) for root in a.roots]
 if a.out:a.out.write_text(json.dumps(reports,indent=2)+'\n')
 for report in reports:
  print(report['root'],len(report['runs']),'runs')
  for g in sorted(report['groups'],key=lambda x:(x['mean_final'],x['mean_auc']),reverse=True):
   print(g['arm']['name'],g['n'],'final',np.round(g['final'],3).tolist(),'auc',round(g['mean_auc'],3),'95',g['times95'],'norm',round(g['mean_norm'],4))
if __name__=='__main__':main()
