"""Common real ancestry paths, deterministic forecasts, evaluation-only xy decoder."""
import argparse,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from map_learning.training import Trainer
from map_learning.maze import Maze
from map_learning.compression_study import ancestry_sequences
class MaxLag:
 def integers(self,low,high):return np.asarray(high)-1

@torch.no_grad()
def audit(run,reference,horizons=(1,4,16,32)):
 t=Trainer.load(Path(run['run'])/'checkpoint.pt');env=Maze(t.config.maze,81319)
 rng=np.random.default_rng(81311)
 # Decoder labels are evaluation-only and never enter the learned models/search.
 xy=rng.uniform(.001,env.config.size-.001,(2048,2)).astype('float32')
 obs=np.stack([env.observe(x) for x in xy]);z=t.model.encode(t.tensor(obs)).numpy()
 mean=z.mean(0);std=np.maximum(z.std(0),1e-8)
 design=np.column_stack([(z-mean)/std,np.ones(len(z))]);penalty=np.eye(design.shape[1])*1e-3;penalty[-1,-1]=0
 decoder=np.linalg.solve(design.T@design+penalty,design.T@xy)
 def decode(z):return np.column_stack([(z-mean)/std,np.ones(len(z))])@decoder
 data=np.load(reference/'replay.npz');ancdata=np.load(reference/'ancestry.npz')
 anc=SimpleNamespace(**{k:ancdata[k] for k in ancdata.files},size=len(ancdata['parents']))
 ends=rng.integers(len(data['actions']),size=256);rows=[]
 for h in horizons:
  src,children,valid,lengths=ancestry_sequences(anc,ends,h,MaxLag())
  z0=t.model.encode(t.tensor(anc.observations[src]));target=t.model.encode(t.tensor(anc.observations[ends+1]))
  pred=[z0 for _ in t.dynamics.heads];actions=t.tensor(data['actions'][children-1])
  for j in range(children.shape[1]):
   mask=torch.as_tensor(valid[:,j,None]);pred=[torch.where(mask,t.dynamics.predict_head(k,z,actions[:,j]),z) for k,z in enumerate(pred)]
  predicted=torch.stack(pred).mean(0).numpy();actual=target.numpy();initial=z0.numpy()
  endxy=data['next_positions'][ends]
  startxy=np.stack([env.start if i==0 else data['next_positions'][i-1] for i in src])
  mse=float(np.mean((predicted-actual)**2));stay=float(np.mean((initial-actual)**2))
  rows.append({'horizon':h,'mean_actual_lag':float(lengths.mean()),'prediction_mse':mse,'stay_latent_mse':stay,
    'relative_to_latent_stay':mse/max(stay,1e-20),'predicted_xy_rmse':float(np.sqrt(np.mean(np.sum((decode(predicted)-endxy)**2,axis=1)))),
    'decoded_start_xy_rmse':float(np.sqrt(np.mean(np.sum((decode(initial)-endxy)**2,axis=1)))),
    'encoded_target_xy_rmse':float(np.sqrt(np.mean(np.sum((decode(actual)-endxy)**2,axis=1)))),
    'physical_stay_rmse':float(np.sqrt(np.mean(np.sum((startxy-endxy)**2,axis=1))))})
 return {'run':run['run'],'name':run['name'],'seed':run['seed'],'reference':str(reference),'metrics':rows}

def main():
 p=argparse.ArgumentParser();p.add_argument('results',type=Path,nargs='+');p.add_argument('--reference-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--all',action='store_true');a=p.parse_args();torch.set_num_threads(1)
 runs=[r for source in a.results for report in json.loads(source.read_text()) for r in report['runs']]
 rows=[]
 for r in runs:
  if a.all or 'multistep' in r['name'] or r['name'].endswith('-base'):
   layout_suffix='' if r['layout']==7 else f"-layout{r['layout']}"
   rows.append(audit(r,a.reference_root/f"adjacency-base{layout_suffix}-seed{r['seed']}"))
 a.out.write_text(json.dumps({'note':'Identical recorded ancestry paths from adjacency baseline per seed. Models have different own replays; mixed seen/unseen forecast evaluation. Decoder fit uses privileged xy only for evaluation, not model/search. Target-decoding error is reported to expose representation/decoder limitations.','runs':rows},indent=2)+'\n')
 print('audited',len(rows),'models')
if __name__=='__main__':main()
