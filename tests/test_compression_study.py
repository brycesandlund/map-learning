import numpy as np
import pytest
import torch
from map_learning.config import load_config
from map_learning.training import Trainer,seed_all
from map_learning.distance_learning import DistanceTrainer,Ancestry
from map_learning.compression_study import CompressionTrainer,ancestry_sequences
from map_learning.replay import Replay


def setup(objective='dynamics',**spec):
 c=load_config('configs/continuous_frontier.json');c.model.hidden_dim=16;c.model.hidden_layers=1;c.train.batch_size=8;c.collect_steps=50
 if spec.get('bound'):c.model.latent_activation='tanh';c.model.latent_bound=spec['bound']
 seed_all(17);t=CompressionTrainer(c,(10,),dict(objective=objective,**spec));r=Replay();rng=np.random.default_rng(6)
 root=rng.normal(size=10).astype('float32');t.ancestry.root(root)
 for i in range(20):
  parent=int(rng.integers(i+1));obs=t.ancestry.observations[parent].copy();nxt=rng.normal(size=10).astype('float32')
  r.add(obs,rng.uniform(-1,1,2),nxt,i,[0,0],[0,0]);t.ancestry.add(parent,nxt)
 return t,r


@pytest.mark.parametrize('objective',['dynamics','adjacency'])
def test_zero_spec_exact_baseline(objective):
 t,r=setup(objective)
 seed_all(17)
 old=Trainer(t.config,(10,)) if objective=='dynamics' else DistanceTrainer(t.config,(10,),{'kind':'adjacency','weight':.3})
 for _ in range(5):
  batch=[np.arange(8)]*3;t.update_batch(r,batch);old.update_batch(r,batch)
  for module in ['model','dynamics']:
   for x,y in zip(getattr(t,module).parameters(),getattr(old,module).parameters()):assert torch.equal(x,y)


def test_ancestry_sequences_are_paths_not_chronological_rows():
 t,r=setup();a=t.ancestry
 for h in [1,4,32]:
  sources,children,valid,lengths=ancestry_sequences(a,np.arange(20),h,np.random.default_rng(42))
  for i in range(20):
   cursor=sources[i]
   for child in children[i,valid[i]]:
    assert a.parents[child]==cursor
    assert np.array_equal(r.observations[child-1],a.observations[cursor]);cursor=child
   assert cursor==i+1
   assert lengths[i]<=h


@pytest.mark.parametrize('objective',['dynamics','adjacency'])
def test_noise_l1_multistep_exact_resume_and_gradients(objective,tmp_path):
 t,r=setup(objective,latent_l1=.001,noise_std=.01,bound=.25,multistep_weight=.3,horizon=4)
 t.update_batch(r,[np.arange(8)]*3);t.save(tmp_path/'model.pt');restored=Trainer.load(tmp_path/'model.pt')
 m=t.update_batch(r,[np.arange(8,16)]*3);n=restored.update_batch(r,[np.arange(8,16)]*3)
 assert m==n and m['path_presentations']==16 and m['path_edges']>=16
 assert m['multistep_loss']>0 and m['latent_l1']>0
 for x,y in zip(t.model.parameters(),restored.model.parameters()):assert torch.equal(x,y)
 assert any(p.grad is not None and p.grad.abs().sum()>0 for p in t.dynamics.parameters())


def test_adjacency_antithetic_noise_only_adds_constant():
 z=torch.randn(10,8,requires_grad=True);target=torch.randn(10,8,requires_grad=True);eps=torch.randn(10,8)
 clean=(z-target).square().mean()
 noisy=((z+eps-target).square().mean()+(z-eps-target).square().mean())/2
 for a,b in zip(torch.autograd.grad(clean,(z,target),retain_graph=True),torch.autograd.grad(noisy,(z,target))):assert torch.allclose(a,b,atol=1e-7)


def test_delayed_l1_is_exactly_off_until_data_threshold():
 t,r=setup('dynamics',latent_l1=.1,l1_after=20,l1_ramp=10)
 seed_all(17);old=Trainer(t.config,(10,))
 a=t.update_batch(r,[np.arange(8)]*3);old.update_batch(r,[np.arange(8)]*3)
 assert a['effective_l1_weight']==0
 for x,y in zip(t.model.parameters(),old.model.parameters()):assert torch.equal(x,y)
 r.add(r.observations[0],r.actions[0],r.next_observations[0],0,[0,0],[0,0])
 assert t.update_batch(r,[np.arange(8)]*3)['effective_l1_weight']==pytest.approx(.01)


def test_delayed_noise_matches_bounded_noiseless_training():
 t,r=setup('adjacency',noise_std=.01,bound=.25,noise_after=20,noise_ramp=10)
 seed_all(17);old=DistanceTrainer(t.config,(10,),{'kind':'adjacency','weight':.3})
 a=t.update_batch(r,[np.arange(8)]*3);old.update_batch(r,[np.arange(8)]*3)
 assert a['effective_noise_std']==0
 for x,y in zip(t.model.parameters(),old.model.parameters()):assert torch.equal(x,y)
