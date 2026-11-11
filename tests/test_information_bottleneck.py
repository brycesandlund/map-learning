import numpy as np
import torch
from map_learning.config import load_config
from map_learning.training import Trainer,seed_all
from map_learning.information_bottleneck import BottleneckTrainer,LearnedPrior
from map_learning.replay import Replay


def setup():
    cfg=load_config('configs/noisy_frontier.json',['model.hidden_dim=16','model.hidden_layers=2','train.batch_size=8'])
    replay=Replay();rng=np.random.default_rng(1)
    for i in range(32):
        obs=rng.normal(size=10).astype('float32');nxt=obs.copy();nxt[:2]+=.1;nxt[2:]=rng.normal(size=8)
        replay.add(obs,i%5,nxt,i,np.zeros(2),np.ones(2))
    return cfg,replay


def test_analytic_kl_and_gradients():
    prior=LearnedPrior(3,1).double();mu=torch.randn(10,3,dtype=torch.double,requires_grad=True)
    ls=torch.full((10,3),-3.,dtype=torch.double,requires_grad=True)
    kl=prior.kl(mu,ls,None)
    target=torch.distributions.kl_divergence(torch.distributions.Normal(mu,ls.exp()),torch.distributions.Normal(prior.means[0],prior.log_std[0].exp())).sum(-1).mean()
    torch.testing.assert_close(kl,target)
    kl.backward();assert mu.grad.abs().sum()>0 and ls.grad.abs().sum()>0 and prior.means.grad.abs().sum()>0


def test_mixture_mc_matches_identical_components():
    prior=LearnedPrior(2,4).double()
    with torch.no_grad():prior.means.zero_()
    mu=torch.zeros(10000,2,dtype=torch.double);ls=torch.full_like(mu,-3.)
    mc=prior.kl(mu,ls,torch.Generator().manual_seed(2),samples=32)
    exact=torch.distributions.kl_divergence(torch.distributions.Normal(mu,ls.exp()),torch.distributions.Normal(prior.means[0],prior.log_std[0].exp())).sum(-1).mean()
    assert abs(float((mc-exact).detach()))<.01


def test_mean_encoder_and_dynamics_initialization_match_original():
    cfg,replay=setup();seed_all(2);base=Trainer(cfg,(10,));seed_all(2);b=BottleneckTrainer(cfg,(10,),beta=1e-4)
    x=b.tensor(replay.sample_states(8,np.random.default_rng(0)))
    torch.testing.assert_close(base.model.encode(x),b.model.encode(x),rtol=0,atol=0)
    for x,y in zip(base.dynamics.parameters(),b.dynamics.parameters()):torch.testing.assert_close(x,y,rtol=0,atol=0)
    state=b.sample_rng.get_state().clone();y=b.model.encode(b.tensor(np.array(replay.observations[:8])));b.model.encode(b.tensor(np.array(replay.observations[:8])))
    assert torch.equal(state,b.sample_rng.get_state())
    assert torch.equal(y,b.posterior(b.tensor(np.array(replay.observations[:8])))[0])


def test_checkpoint_restores_next_update_and_mean_inference(tmp_path):
    cfg,replay=setup();seed_all(0);t=BottleneckTrainer(cfg,(10,),beta=1e-4,prior_components=4)
    t.update(replay);path=tmp_path/'checkpoint.pt';t.save(path)
    expected=t.update(replay);loaded=Trainer.load(path);actual=loaded.update(replay)
    assert expected==actual
    for x,y in zip(t.model.parameters(),loaded.model.parameters()):torch.testing.assert_close(x,y,rtol=0,atol=0)
    assert t.model.posterior_log_std.weight.grad.abs().sum()>0


def test_zero_beta_leaves_prior_unchanged_but_trains_variance():
    cfg,replay=setup();seed_all(0);t=BottleneckTrainer(cfg,(10,),beta=0)
    before=[p.detach().clone() for p in t.prior.parameters()];ls=t.model.posterior_log_std.bias.detach().clone()
    t.update(replay)
    for p,q in zip(before,t.prior.parameters()):assert torch.equal(p,q)
    assert not torch.equal(ls,t.model.posterior_log_std.bias)


def test_warm_start_matches_zero_beta_until_activation():
    cfg,replay=setup();seed_all(0);base=BottleneckTrainer(cfg,(10,),beta=0)
    seed_all(0);warm=BottleneckTrainer(cfg,(10,),beta=1e-4,kl_warmup_updates=2)
    for _ in range(2):
        base.update(replay);stats=warm.update(replay)
        assert stats['active_beta']==0
    for p,q in zip(base.model.parameters(),warm.model.parameters()):torch.testing.assert_close(p,q,rtol=0,atol=0)
    before=warm.prior.means.detach().clone();stats=warm.update(replay)
    assert stats['active_beta']==1e-4
    assert not torch.equal(before,warm.prior.means)
