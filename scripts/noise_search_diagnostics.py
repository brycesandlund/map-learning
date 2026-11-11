"""Evaluation-only nuisance interventions on common archives, frozen models."""
import argparse,json,subprocess,sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.stats import spearmanr,rankdata
from map_learning.training import Trainer,seed_all
from map_learning.maze import Maze,reachable_graph
from audit_frontier_efficiency import reconstruct,encode,priority_stats
from audit_queue_density import live_sequence
from kl_mechanisms import data_source


class EncoderView:
 def __init__(self,t,mode,center=None,radius=None):self.t=t;self.mode=mode;self.center=center;self.radius=radius
 def encode(self,x):
  if self.mode=='zero':x=x.clone();x[:,2:]=0
  z=self.t.model.encode(x)
  if self.mode=='cap':
   center=torch.as_tensor(self.center,dtype=z.dtype,device=z.device);delta=z-center
   z=center+delta*torch.clamp(self.radius/delta.norm(dim=-1,keepdim=True).clamp_min(1e-12),max=1)
  return z


def case(job,out):
 seed_all(0,1);t=Trainer.load(job['checkpoint']);data=np.load(Path(job['source'])/'replay.npz');prefix=job['prefix'];seed=job['seed']
 obs,pos,vo,vp=reconstruct(data,prefix);points,adj,index=reachable_graph(Maze(t.config.maze))
 ids=lambda ps:np.array([index[tuple(np.round(p,5))] for p in ps])
 fid=ids(pos);seen=np.zeros(len(points),bool);seen[ids(data['positions'][:1])]=True;seen[ids(data['next_positions'][:prefix])]=True
 expanded=np.zeros(len(points),bool);expanded[ids(vp)]=True;productive=(~seen[adj]).any(1)
 z=encode(t,obs);v=encode(t,vo);scores=cKDTree(v).query(z)[0];order=np.argsort(-scores,kind='stable');top=order[:100]
 center=v.mean(0);radius=float(np.quantile(np.linalg.norm(v-center,axis=1),.95))
 zero=obs.copy();zero[:,2:]=0;residual=np.linalg.norm(z-encode(t,zero),axis=1)
 norms=np.linalg.norm(z,axis=1);radii=np.linalg.norm(z-center,axis=1)
 pct=lambda x:rankdata(x,method='average')/len(x)
 corr=lambda a,b:float(spearmanr(a,b).statistic) if np.std(a)>0 and np.std(b)>0 else None
 def stats(s):
  o=np.argsort(-s,kind='stable');a=set(fid[top]);b=set(fid[o[:100]])
  return {**priority_stats(s,fid,productive,expanded),'top1_productive':bool(productive[fid[o[0]]]),
          'top100_entry_overlap':len(set(top)&set(o[:100]))/100,'top100_physical_jaccard':len(a&b)/len(a|b),
          'score_spearman_vs_raw':corr(scores,s)}
 raw=stats(scores)
 raw.update(norm_priority_spearman=corr(norms,scores),radius_priority_spearman=corr(radii,scores),residual_priority_spearman=corr(residual,scores),
            top100_norm_percentile=float(pct(norms)[top].mean()),top100_radius_percentile=float(pct(radii)[top].mean()),
            top100_noise_residual_percentile=float(pct(residual)[top].mean()),
            repeated_top100_noise_percentile=float(pct(residual)[top][expanded[fid[top]]].mean()) if expanded[fid[top]].any() else None)
 ranking=[];live=[]
 # Ten paired ranking-only nuisance redraws; raw archive membership is fixed.
 for replicate in range(10):
  rng=np.random.default_rng(seed+prefix+1040000+replicate*1000)
  oq=obs.copy();va=vo.copy();oq[:,2:]=rng.standard_normal(oq[:,2:].shape);va[:,2:]=rng.standard_normal(va[:,2:].shape)
  zq=encode(t,oq);vv=encode(t,va)
  for mode,oo,vvo,zz,vs in [('queue',oq,vo,zq,v),('anchors',obs,va,z,vv),('both',oq,va,zq,vv)]:
   ss=cKDTree(vs).query(zz)[0];ranking.append({'mode':mode,'replicate':replicate,**stats(ss)})
   if replicate<3 and mode in ['queue','both']:
    rr=live_sequence(t,oo,pos,vvo,ss,zz,vs,seen.copy(),expanded,points,index,128,seed+1050000+replicate*100)
    live.append({'mode':mode,'replicate':replicate,'new_states':rr[-1]['new_states'],'rows':rr})
 for mode in ['raw','zero','cap']:
  view=t if mode=='raw' else SimpleNamespace(config=t.config,tensor=t.tensor,model=EncoderView(t,mode,center,radius))
  zz=encode(view,obs);vs=encode(view,vo);ss=cKDTree(vs).query(zz)[0]
  if mode!='raw':ranking.append({'mode':mode,'replicate':0,**stats(ss)})
  for replicate in range(3):
   rr=live_sequence(view,obs,pos,vo,ss,zz,vs,seen.copy(),expanded,points,index,128,seed+1050000+replicate*100)
   live.append({'mode':mode,'replicate':replicate,'new_states':rr[-1]['new_states'],'rows':rr})
 result={**job,'seen_states':int(seen.sum()),'queue_entries':len(obs),'expanded_entries':len(vo),'productive_entries':int(productive[fid].sum()),
         'cap_radius':radius,'raw':raw,'ranking':ranking,'live':live}
 with (out/f'{job["kind"]}-seed{seed}-at{prefix}.json').open('x') as f:json.dump(result,f,indent=2)


def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--l2-root',type=Path);p.add_argument('--worker',type=int);p.add_argument('--jobs',type=int,default=4);a=p.parse_args()
 if a.worker is not None:
  case(json.loads((a.out/'manifest.json').read_text())['jobs'][a.worker],a.out);return
 a.out.mkdir(exist_ok=False,parents=True);jobs=[]
 kinds=['l2'] if a.l2_root else ['none','kl']
 for kind in kinds:
  for seed in [9,10,11]:
   d=a.l2_root/f'seed{seed}' if a.l2_root else Path(f'runs/kl-mechanisms-fixed-20261110/zero-{"mean_kl" if kind=="kl" else "mean_zero"}-seed{seed}')
   for prefix in [50000,75000]:jobs.append(dict(kind=kind,seed=seed,prefix=prefix,checkpoint=str(d/f'checkpoint-{prefix//5000}.pt'),source=str(data_source('zero',seed-3))))
 (a.out/'manifest.json').write_text(json.dumps({'jobs':jobs,'ranking_replicates':10,'live_replicates':3,'expansions':128},indent=2)+'\n')
 def run(i):
  j=jobs[i]
  with (a.out/f'worker-{i}.log').open('x') as log:r=subprocess.run([sys.executable,__file__,'--out',str(a.out),'--worker',str(i)],stdout=log,stderr=subprocess.STDOUT)
  print(json.dumps({**j,'exit_code':r.returncode}),flush=True);return r.returncode
 with ThreadPoolExecutor(max_workers=a.jobs) as pool:codes=list(pool.map(run,range(len(jobs))))
 if any(codes):raise SystemExit(1)
 rows=[json.loads((a.out/f'{j["kind"]}-seed{j["seed"]}-at{j["prefix"]}.json').read_text()) for j in jobs]
 (a.out/'results.json').write_text(json.dumps(rows,indent=2)+'\n')

if __name__=='__main__':main()
