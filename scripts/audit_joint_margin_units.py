"""Static change-of-units audit; no fitting or extra environment interactions."""
import argparse,copy,json
from pathlib import Path
import numpy as np
import torch
from map_learning.training import Trainer
from map_learning.replay import Replay
from map_learning.probabilistic import repulsion_loss


def rescale(t,factor):
 out=copy.deepcopy(t)
 assert out.config.model.latent_activation=='identity'
 with torch.no_grad():
  layer=out.model.encoder[-1];layer.weight.mul_(factor);layer.bias.mul_(factor)
  for head in out.dynamics.heads:
   head[0].weight[:,:out.config.model.latent_dim].div_(factor)
   head[-1].weight.mul_(factor);head[-1].bias.mul_(factor)
 return out


@torch.no_grad()
def main():
 p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 torch.set_num_threads(1);t=Trainer.load(a.run/'checkpoint.pt');replay,_=Replay.load(a.run/'replay.npz')
 obs,actions,nxt=replay.transitions_at(np.random.default_rng(42).integers(len(replay),size=256))
 def terms(model,margin):
  z=model.model.encode(model.tensor(obs));target=model.model.encode(model.tensor(nxt));pred=model.dynamics(z,model.tensor(actions))
  return z,pred,torch.stack([(pred-target).square().mean(),(z-target).square().sum(-1).mean(),repulsion_loss(z,target.flip(0),margin),z.square().sum(-1).mean()])
 z,pred,base=terms(t,1.)
 rows=[]
 for m in [.25,.5,2.,4.]:
  changed=rescale(t,m);zm,pm,loss=terms(changed,m)
  torch.testing.assert_close(zm,z*m,atol=1e-6,rtol=1e-5)
  torch.testing.assert_close(pm,pred*m,atol=1e-6,rtol=1e-5)
  torch.testing.assert_close(loss,base*m*m,atol=1e-6,rtol=1e-5)
  rows.append({'margin':m,'components':loss.tolist(),'expected_components':(base*m*m).tolist(),
               'max_prediction_rescaling_error':float((pm-pred*m).abs().max())})
 a.out.write_text(json.dumps({'run':str(a.run),'base_components':base.tolist(),'rows':rows,
  'meaning':'All four loss terms scale by m² under a realizable change of units; Euclidean queue order is unchanged. This does not make finite Adam training with fixed initialization/LR scale-invariant.'},indent=2)+'\n')
 print('All static unit-rescaling checks pass')
if __name__=='__main__':main()
