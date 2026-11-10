from types import SimpleNamespace
import numpy as np
import torch
from torch.distributions import Categorical, Normal, Independent, MixtureSameFamily
from map_learning.config import load_config, PlannerConfig
from map_learning.training import Trainer, seed_all
from map_learning.probabilistic import (mixture_log_prob, mixture_sample, ensemble_js,
                                       mixture_mean, mixture_variance, MixtureDynamicsEnsemble,
                                       repulsion_loss)
from map_learning.uncertainty_planning import UncertaintyPlanner
from test_learning import line_replay


def config():
    return load_config(overrides=['train.loss=probabilistic', 'planner.search=mcts',
                'model.latent_dim=4', 'model.hidden_dim=32', 'model.ensemble_size=3',
                'model.mixture_components=2', 'train.batch_size=64', 'train.lr=0.0005'])


def test_mixture_likelihood_matches_torch_and_samples_both_modes():
    seed_all(0)
    logits = torch.randn(3, 7, 2, dtype=torch.float64)
    means = torch.randn(3, 7, 2, 4, dtype=torch.float64)
    variances = torch.rand_like(means)+.1
    target = torch.randn(7, 4, dtype=torch.float64, requires_grad=True)
    ours = mixture_log_prob((logits, means, variances), target)
    reference = MixtureSameFamily(Categorical(logits=logits), Independent(Normal(means, variances.sqrt()), 1))
    torch.testing.assert_close(ours, reference.log_prob(target))
    (-ours.mean()).backward()
    assert torch.isfinite(target.grad).all() and target.grad.norm()>0
    params = (torch.tensor([.25, .75]).log().expand(5000, -1),
              torch.tensor([[-2.], [2.]]).expand(5000, -1, -1), torch.full((5000, 2, 1), .0001))
    draws = mixture_sample(params)
    assert abs(float((draws>0).float().mean())-.75)<.03
    torch.testing.assert_close(mixture_mean(params)[0], torch.tensor([1.]))
    torch.testing.assert_close(mixture_variance(params)[0], torch.tensor([3.0001]))


def test_disagreement_ignores_shared_aleatoric_noise_and_component_labels():
    seed_all(7)
    logits = torch.tensor([[.2, .8], [.8, .2]]).log().unsqueeze(1)
    means = torch.tensor([[-2., 2.], [2., -2.]]).reshape(2, 1, 2, 1)
    var = torch.ones_like(means)*2
    js = ensemble_js((logits, means, var), samples=1000)
    assert js.abs().max()<1e-6  # identical broad mixtures, permuted labels
    separated = means.clone()
    separated[1] += 20
    assert ensemble_js((logits, separated, var), samples=1000).item()>.65
    # Same mean but distinct variances is also distribution disagreement.
    logits = torch.zeros(2, 1, 1)
    means = torch.zeros(2, 1, 1, 1)
    var = torch.tensor([.01, 4.]).reshape(2, 1, 1, 1)
    assert ensemble_js((logits, means, var), samples=1000).item()>.25


def test_joint_training_noncollapse_and_checkpoint_continuation(tmp_path):
    seed_all(3)
    trainer = Trainer(config(), (2,))
    replay = line_replay()
    before = {k:v.clone() for k, v in trainer.model.state_dict().items()}
    for _ in range(220):
        metrics = trainer.update(replay)
    assert all(np.isfinite(v) for v in metrics.values())
    assert any(not torch.equal(v, before[k]) for k,v in trainer.model.state_dict().items())
    z = trainer.model.encode(trainer.tensor([[0., 0.], [1., 0.]]))
    assert (z[0]-z[1]).norm()>.5
    obs, a, nxt = replay.sample_transitions(256, np.random.default_rng(2))
    z0, z1 = trainer.model.encode(trainer.tensor(obs)), trainer.model.encode(trainer.tensor(nxt))
    means = trainer.dynamics(z0, torch.as_tensor(a))
    assert (means.mean(0)-z1).square().mean() < (z0-z1).square().mean()
    trainer.save(tmp_path/'checkpoint.pt')
    expected = trainer.update(replay)
    restored = Trainer.load(tmp_path/'checkpoint.pt')
    assert restored.update(replay) == expected


def test_likelihood_backpropagates_through_both_encoder_branches():
    seed_all(0)
    model = MixtureDynamicsEnsemble(config().model)
    z = torch.randn(6, 4, requires_grad=True)
    target = torch.randn(6, 4, requires_grad=True)
    loss = -mixture_log_prob(model.parameters_head(0, z, torch.arange(6)%5), target).mean()
    loss.backward()
    assert z.grad.norm()>0 and target.grad.norm()>0
    left = torch.tensor([[0., 0.]], requires_grad=True)
    right = torch.tensor([[.2, 0.]], requires_grad=True)
    repulsion_loss(left, right).backward()
    assert left.grad[0, 0]>0 and right.grad[0, 0]<0


def test_probabilistic_mode_accepts_pixels_and_noisy_features():
    from map_learning.maze import Maze
    from map_learning.replay import Replay
    seed_all(0)
    for kind, noise in [('pixels', 0), ('features', 3)]:
        cfg = config()
        cfg.maze.observation, cfg.maze.nuisance_dims, cfg.maze.size = kind, noise, 3
        cfg.validate()
        env, replay = Maze(cfg.maze), Replay()
        obs = env.reset()
        for action in [0, 2, 1, 3]*3:
            position = env.position.copy()
            nxt, _ = env.step(action)
            replay.add(obs, action, nxt, 0, position, env.position)
            obs = nxt
        trainer = Trainer(cfg, env.observation_shape)
        assert np.isfinite(trainer.update(replay)['joint_loss'])


def test_mixture_learns_two_outcomes_without_averaging_them():
    seed_all(5)
    cfg = config()
    dynamics = MixtureDynamicsEnsemble(cfg.model)
    optimizer = torch.optim.Adam(dynamics.heads[0].parameters(), lr=.005)
    z = torch.zeros(128, 4)
    action = torch.zeros(128, dtype=torch.long)
    target = torch.zeros_like(z)
    target[:64, 0], target[64:, 0] = -1., 1.
    for _ in range(300):
        loss = -mixture_log_prob(dynamics.parameters_head(0, z, action), target).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    logits, means, var = dynamics.parameters_head(0, z[:1], action[:1])
    torch.testing.assert_close(means[0, :, 0].sort().values, torch.tensor([-1., 1.]), atol=.1, rtol=0)
    assert logits.softmax(-1).min()>.35 and var[0, :, 0].max()<.05


def test_mcts_averages_chance_outcomes_instead_of_choosing_lucky_branch():
    class Identity:
        def encode(self, x): return x
    trainer = SimpleNamespace(config=SimpleNamespace(train=SimpleNamespace(prediction_objective='nll')), device='cpu', model=Identity(), dynamics=SimpleNamespace(actions=2),
                              tensor=lambda x:torch.as_tensor(x, dtype=torch.float32))
    cfg = PlannerConfig(behavior='disagreement', search='mcts', horizon=2, simulations=1200,
                        epsilon_random=0, uct_exploration=.3, widening_coefficient=2.)
    class ToyPlanner(UncertaintyPlanner):
        def distribution_reward(self, state, actions):
            # At root, action 0: 10% chance of reward 1, else zero.
            # Action 1: guaranteed reward .3 on the next step. Prefer expected .3.
            b = len(actions)
            logits = torch.tensor([.9, .1]).log().repeat(2, b, 1)
            mu = torch.zeros(2, b, 2, 1)
            mu[:, :, 1, 0] = 10
            mu[:, actions==1, :, 0] = 3
            var = torch.ones_like(mu)*1e-10
            reward = (state[:, 0]/10).clamp_min(0)
            return (logits, mu, var), reward
    planner = ToyPlanner(trainer, cfg, seed=8)
    assert planner.act(np.array([0.])) == 1


def test_shooting_selects_distribution_disagreement():
    class Identity:
        def encode(self, x): return x
    class Dynamics:
        actions = 5
        def distribution(self, state, actions):
            mu = state[None, :, None, :].repeat(2, 1, 1, 1)
            mu[1, actions==2] += 2
            return torch.zeros(2, len(state), 1), mu, torch.ones_like(mu)*.001
    trainer = SimpleNamespace(config=SimpleNamespace(train=SimpleNamespace(prediction_objective='nll')), device='cpu', model=Identity(), dynamics=Dynamics(),
                              tensor=lambda x:torch.as_tensor(x, dtype=torch.float32))
    cfg = PlannerConfig(behavior='disagreement', search='shooting', horizon=1, epsilon_random=0)
    planner = UncertaintyPlanner(trainer, cfg, seed=3)
    assert planner.act(np.zeros(2)) == 2


def test_js_matches_independent_density_quadrature_for_overlapping_mixtures():
    # Check the actual divergence value, not just zero/positive edge cases.
    seed_all(19)
    logits = torch.tensor([[[.8, .2], [.3, .7]],
                           [[.4, .6], [.6, .4]],
                           [[.7, .3], [.5, .5]]], dtype=torch.float64).log()
    means = torch.tensor([[[-1., 1.], [-.5, 1.5]],
                          [[-.6, 1.4], [-1.2, .2]],
                          [[-1.5, .5], [-.8, .8]]], dtype=torch.float64).unsqueeze(-1)
    variances = torch.full_like(means, .35)
    distribution = MixtureSameFamily(Categorical(logits=logits),
                                     Independent(Normal(means, variances.sqrt()), 1))
    grid = torch.linspace(-8., 8., 16001, dtype=torch.float64)
    density = distribution.log_prob(grid[:, None, None, None]).exp()
    pooled = density.mean(1, keepdim=True)
    integrand = (density*(density.log()-pooled.log())).mean(1)
    reference = torch.trapezoid(integrand, grid, dim=0)
    measured = ensemble_js((logits, means, variances), samples=40000)
    torch.testing.assert_close(measured, reference, atol=.01, rtol=.02)


def test_unbounded_encoder_is_opt_in_and_legacy_config_stays_bounded():
    from dataclasses import asdict
    from map_learning.config import config_from_dict
    seed_all(6)
    cfg = config()
    old = asdict(cfg)
    del old['model']['latent_activation']
    assert config_from_dict(old).model.latent_activation == 'tanh'
    trainer = Trainer(cfg, (2,))
    obs = torch.tensor([[0., 0.], [1., 1.]])
    with torch.no_grad():
        trainer.model.encoder[-1].bias.fill_(3.)
        raw = trainer.model.encoder(obs)
        torch.testing.assert_close(trainer.model.encode(obs), raw.tanh())
        trainer.model.latent_activation = 'identity'
        torch.testing.assert_close(trainer.model.encode(obs), raw)
        assert trainer.model.encode(obs).max()>1


def test_evaluation_accepts_live_legacy_model_config_without_activation(tmp_path):
    from dataclasses import asdict
    from map_learning.maze import Maze
    from map_learning.replay import Replay
    from map_learning.probabilistic_evaluation import evaluate_probabilistic
    seed_all(12)
    cfg = config()
    cfg.maze.size = 2
    env, replay = Maze(cfg.maze), Replay()
    obs = env.reset()
    for action in [0, 2, 1, 3]*3:
        before = env.position.copy()
        nxt, _ = env.step(action)
        replay.add(obs, action, nxt, 0, before, env.position)
        obs = nxt
    trainer = Trainer(cfg, env.observation_shape)
    trainer.update(replay)
    legacy = asdict(cfg.model)
    legacy.pop('latent_activation')
    # A long-lived worker can hold an old config while evaluation is imported later.
    trainer.config.model = SimpleNamespace(**legacy)
    metrics = evaluate_probabilistic(trainer, replay, tmp_path)
    assert metrics['encoder_saturation_fraction'] is not None
    assert metrics['oracle_sample_outside_encoder_bounds_fraction'] is not None
