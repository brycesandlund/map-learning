"""Summaries and parent-pointer checks for compression experiments."""
import argparse,json
from pathlib import Path
import numpy as np
from summarize_joint_loss import collect


def main():
 p=argparse.ArgumentParser();p.add_argument('roots',type=Path,nargs='+');p.add_argument('--out',type=Path);p.add_argument('--partial',action='store_true');p.add_argument('--ancestry-audit',action='store_true');a=p.parse_args()
 reports=[]
 for root in a.roots:
  report=collect(root,a.partial)
  for r in report['runs']:
   path=Path(r['run']);specpath=path/'compression-spec.json';r['compression_spec']=json.loads(specpath.read_text()) if specpath.exists() else {}
   logs=[json.loads(s) for s in (path/'training.jsonl').read_text().splitlines()]
   r['final_diagnostics']={k:float(np.mean([x[k] for x in logs[-10:]])) for k in ['latent_l1','effective_l1_weight','effective_noise_std','mean_rollout_lag','prediction_loss','multistep_loss'] if all(k in x for x in logs[-10:])}
   r['path_edges']=logs[-1].get('path_edges',0);r['path_presentations']=logs[-1].get('path_presentations',0)
   if a.ancestry_audit and (path/'ancestry.npz').exists():
    anc=np.load(path/'ancestry.npz');replay=np.load(path/'replay.npz');parents=anc['parents'][1:]
    assert np.all(parents<np.arange(1,len(parents)+1))
    assert np.array_equal(anc['observations'][parents],replay['observations'])
    assert np.array_equal(anc['observations'][1:],replay['next_observations'])
    assert np.array_equal(anc['depth'][1:],anc['depth'][parents]+1)
    r['ancestry_audited']=True
  reports.append(report)
  print(root,len(report['runs']))
  for g in sorted(report['groups'],key=lambda x:(x['mean_final'],x['mean_auc']),reverse=True):
   print(g['arm']['name'],g['n'],'coverage',np.round(np.array(g['final'])*100,1).tolist(),'auc',round(g['mean_auc'],3),'norm',round(g['mean_norm'],7))
 if a.out:a.out.write_text(json.dumps(reports,indent=2)+'\n')
if __name__=='__main__':main()
