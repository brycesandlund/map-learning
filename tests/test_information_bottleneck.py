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


def test_full_replay_moments_include_both_endpoints_and_posterior_variance():
    cfg,replay=setup();seed_all(0)
    t=BottleneckTrainer(cfg,(10,),beta=1e-4,prior_mode='replay_moments')
    with torch.no_grad():
        mu,ls=t.posterior(t.tensor(np.concatenate([replay.observations,replay.next_observations])))
        mu=mu.double();ls=ls.double()
        expected_mean=mu.mean(0);expected_variance=mu.var(0,unbiased=False)+(2*ls).exp().mean(0)
    rng=t.sample_rng.get_state().clone();m=t.replay_posterior_moments(replay,chunk_size=7)
    torch.testing.assert_close(m['mean'],expected_mean,rtol=1e-5,atol=1e-8)
    torch.testing.assert_close((2*m['logstd']).exp(),expected_variance,rtol=1e-5,atol=1e-8)
    assert m['count']==2*len(replay) and torch.equal(rng,t.sample_rng.get_state())
    t.fit_prior(replay)
    torch.testing.assert_close(t.prior.means[0].double(),expected_mean,rtol=1e-5,atol=1e-8)
    torch.testing.assert_close(t.prior.log_std[0].double(),.5*expected_variance.log(),rtol=1e-5,atol=1e-8)
    assert not any(p.requires_grad for p in t.prior.parameters())


def test_moment_prior_zero_beta_exactly_preserves_training():
    cfg,replay=setup();seed_all(0);old=BottleneckTrainer(cfg,(10,),beta=0)
    seed_all(0);new=BottleneckTrainer(cfg,(10,),beta=0,prior_mode='replay_moments')
    for _ in range(5):
        old.update(replay);new.fit_prior(replay);new.update(replay)
    for p,q in zip(old.model.parameters(),new.model.parameters()):torch.testing.assert_close(p,q,rtol=0,atol=0)
    for p,q in zip(old.dynamics.parameters(),new.dynamics.parameters()):torch.testing.assert_close(p,q,rtol=0,atol=0)
    assert old.rng.bit_generator.state==new.rng.bit_generator.state
    assert torch.equal(old.sample_rng.get_state(),new.sample_rng.get_state())


def test_moment_checkpoint_preserves_frozen_prior_and_next_update(tmp_path):
    cfg,replay=setup();seed_all(0);t=BottleneckTrainer(cfg,(10,),beta=1e-4,prior_mode='replay_moments')
    t.fit_prior(replay);before=[p.clone() for p in t.prior.parameters()];t.update(replay)
    for p,q in zip(before,t.prior.parameters()):torch.testing.assert_close(p,q,rtol=0,atol=0)
    t.save(tmp_path/'moment.pt');loaded=Trainer.load(tmp_path/'moment.pt')
    assert loaded.prior_fit_stats==t.prior_fit_stats
    expected=t.update(replay);actual=loaded.update(replay);assert expected==actual
    for p,q in zip(t.model.parameters(),loaded.model.parameters()):torch.testing.assert_close(p,q,rtol=0,atol=0)


def test_detached_moment_prior_matches_gradient_of_profiled_kl():
    cfg,_=setup();t=BottleneckTrainer(cfg,(10,),beta=1e-4,prior_mode='batch_moments')
    t.prior.double()
    mu=torch.randn(64,8,dtype=torch.double,requires_grad=True)
    ls=torch.randn(64,8,dtype=torch.double,requires_grad=True)-2
    t.fit_batch_prior(mu,ls);kl=t.prior.kl(mu,ls,None)
    profiled=(.5*(mu.var(0,unbiased=False)+(2*ls).exp().mean(0)).log()-ls.mean(0)).sum()
    torch.testing.assert_close(kl,profiled)
    g1=torch.autograd.grad(kl,(mu,ls),retain_graph=True);g2=torch.autograd.grad(profiled,(mu,ls))
    for p,q in zip(g1,g2):torch.testing.assert_close(p,q)


def test_delayed_replay_moment_kl_keeps_zero_beta_prefix():
    cfg,replay=setup();seed_all(0);base=BottleneckTrainer(cfg,(10,),beta=0)
    seed_all(0);warm=BottleneckTrainer(cfg,(10,),beta=1e-4,prior_mode='replay_moments',kl_warmup_updates=2)
    for _ in range(2):
        base.update(replay);warm.fit_prior(replay);metrics=warm.update(replay);assert metrics['active_beta']==0
    for p,q in zip(base.model.parameters(),warm.model.parameters()):torch.testing.assert_close(p,q,rtol=0,atol=0)
    warm.fit_prior(replay);before=warm.prior.means.clone();metrics=warm.update(replay)
    assert metrics['active_beta']==1e-4
    torch.testing.assert_close(before,warm.prior.means,rtol=0,atol=0)


def test_scheduled_input_is_prescribed_and_variance_head_cannot_turn_it_off():
    cfg,replay=setup();seed_all(0)
    schedule={'updates':[0,10],'std':[.02,.08]}
    t=BottleneckTrainer(cfg,(10,),beta=0,dynamics_input='scheduled',noise_schedule=schedule)
    t.updates=5
    with torch.no_grad():t.model.posterior_log_std.bias.fill_(-10)
    before=[p.clone() for p in t.model.posterior_log_std.parameters()]
    stats=t.update(replay)
    assert abs(stats['dynamics_input_std']-.05)<1e-7
    for p,q in zip(before,t.model.posterior_log_std.parameters()):torch.testing.assert_close(p,q,rtol=0,atol=0)


def test_prior_only_steps_leave_first_joint_update_identical():
    cfg,replay=setup();seed_all(0);one=BottleneckTrainer(cfg,(10,),beta=1e-5)
    seed_all(0);five=BottleneckTrainer(cfg,(10,),beta=1e-5,prior_steps=5)
    one.update(replay);five.update(replay)
    for p,q in zip(one.model.parameters(),five.model.parameters()):torch.testing.assert_close(p,q,rtol=0,atol=0)
    for p,q in zip(one.dynamics.parameters(),five.dynamics.parameters()):torch.testing.assert_close(p,q,rtol=0,atol=0)
    assert one.prior_updates==1 and five.prior_updates==5
    assert not torch.equal(one.prior.means,five.prior.means)
    assert torch.equal(one.sample_rng.get_state(),five.sample_rng.get_state())
    assert one.rng.bit_generator.state==five.rng.bit_generator.state


def test_mean_input_does_not_depend_on_posterior_scales():
    cfg,replay=setup();seed_all(0);one=BottleneckTrainer(cfg,(10,),beta=0,dynamics_input='mean')
    seed_all(0);two=BottleneckTrainer(cfg,(10,),beta=0,dynamics_input='mean')
    with torch.no_grad():two.model.posterior_log_std.bias.fill_(1)
    s=one.update(replay);r=two.update(replay)
    assert s['prediction_loss']==r['prediction_loss'] and s['dynamics_input_std']==0
    for p,q in zip(one.model.encoder.parameters(),two.model.encoder.parameters()):torch.testing.assert_close(p,q,rtol=0,atol=0)


def test_mechanism_checkpoint_exact_continuation(tmp_path):
    cfg,replay=setup();seed_all(0)
    t=BottleneckTrainer(cfg,(10,),beta=1e-5,prior_steps=5,dynamics_input='scheduled',noise_schedule={'updates':[0,10],'std':[.01,.03]})
    t.update(replay);t.save(tmp_path/'ablation.pt');other=Trainer.load(tmp_path/'ablation.pt')
    assert t.update(replay)==other.update(replay)
    for p,q in zip(t.model.parameters(),other.model.parameters()):torch.testing.assert_close(p,q,rtol=0,atol=0)
    assert t.prior_updates==other.prior_updates==10


def test_latent_l2_is_sum_over_coordinates_and_checkpointed(tmp_path):
    cfg,replay=setup();seed_all(0);base=BottleneckTrainer(cfg,(10,),dynamics_input='mean')
    seed_all(0);reg=BottleneckTrainer(cfg,(10,),dynamics_input='mean',latent_l2=.003)
    b=base.update(replay);r=reg.update(replay)
    assert abs(r['joint_loss']-b['joint_loss']-.003*r['latent_squared_norm'])<1e-7
    assert r['prediction_loss']==b['prediction_loss'] and r['repulsion']==b['repulsion']
    assert any(not torch.equal(p,q) for p,q in zip(base.model.parameters(),reg.model.parameters()))
    reg.save(tmp_path/'l2.pt');restored=Trainer.load(tmp_path/'l2.pt')
    assert restored.spec['latent_l2']==.003 and reg.update(replay)==restored.update(replay)
