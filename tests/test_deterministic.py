from copy import deepcopy
import numpy as np
import pytest
import torch

from map_learning.config import load_config
from map_learning.training import Trainer, seed_all
from map_learning.information_bottleneck import BottleneckTrainer
from map_learning.deterministic import without_distributions
from map_learning.replay import Replay
from map_learning.cadence import TrainingSchedule, CadenceDiagnostics
from map_learning.uncertainty_planning import make_planner
from map_learning.reward_snapshots import mean_reward
from map_learning.evaluation import evaluate
from map_learning.maze import Maze


def setup():
    cfg = load_config('configs/noisy_frontier.json', ['model.hidden_dim=16', 'model.hidden_layers=2',
        'train.batch_size=8', 'maze.size=2', 'planner.horizon=2', 'planner.simulations=10',
        'planner.candidates=5', 'planner.epsilon_random=0'])
    replay = Replay(); env = Maze(cfg.maze, 4)
    obs = env.reset()
    for i in range(40):
        position = env.position.copy(); nxt, _ = env.step(i % 5)
        replay.add(obs, i % 5, nxt, i, position, env.position.copy()); obs = nxt
    return cfg, replay


def test_stripping_distributions_preserves_function_and_adam_update(tmp_path):
    cfg, replay = setup(); seed_all(5)
    old = BottleneckTrainer(cfg, (10,), beta=0, dynamics_input='mean', latent_l2=.03)
    for _ in range(5): old.update(replay)
    clean = without_distributions(old)
    assert not hasattr(clean.model, 'posterior_log_std') and not hasattr(clean, 'prior')
    assert not hasattr(clean.dynamics, 'distribution')
    assert all(head[-1].out_features == cfg.model.latent_dim for head in clean.dynamics.heads)
    assert clean.config.train.loss == 'dynamics' and clean.config.train.latent_l2 == .03
    x = clean.tensor(replay.arrays()['observations']); actions = torch.tensor(replay.arrays()['actions'])
    torch.testing.assert_close(clean.model.encode(x), old.model.encode(x), rtol=0, atol=0)
    torch.testing.assert_close(clean.dynamics(clean.model.encode(x), actions), old.dynamics(old.model.encode(x), actions), rtol=1e-5, atol=1e-7)
    for _ in range(3):
        a, b = old.update(replay), clean.update(replay)
        for key in ['joint_loss', 'prediction_loss', 'repulsion', 'latent_squared_norm']:
            assert a[key] == pytest.approx(b[key], rel=1e-5, abs=1e-7)
        for p, q in zip(clean.model.encoder.parameters(), old.model.encoder.parameters()):
            torch.testing.assert_close(p, q, rtol=1e-5, atol=1e-7)
        assert clean.rng.bit_generator.state == old.rng.bit_generator.state
    path = tmp_path/'clean.pt'; clean.save(path); restored = Trainer.load(path)
    assert clean.update(replay) == restored.update(replay)
    for p, q in zip(clean.model.parameters(), restored.model.parameters()):
        torch.testing.assert_close(p, q, rtol=0, atol=0)


@pytest.mark.parametrize('kwargs', [{'beta': 1e-5, 'dynamics_input': 'mean'}, {'beta': 0, 'dynamics_input': 'posterior'}])
def test_conversion_rejects_active_distribution_losses(kwargs):
    cfg, _ = setup()
    with pytest.raises(ValueError, match='cannot be discarded'):
        without_distributions(BottleneckTrainer(cfg, (10,), **kwargs))


def test_deterministic_training_search_and_evaluation_need_no_distributions(tmp_path):
    cfg, replay = setup()
    cfg.train.loss = 'dynamics'; cfg.train.latent_l2 = .03; cfg.validate()
    with pytest.raises(ValueError, match='legacy probabilistic'):
        BottleneckTrainer(cfg, (10,))
    seed_all(7); trainer = Trainer(cfg, (10,))
    before = deepcopy(trainer.model.state_dict())
    metrics = TrainingSchedule(trainer).train(replay)
    assert metrics['explicit_transitions_per_head'] == len(replay)
    assert metrics['weighted_latent_l2'] > 0
    assert any(not torch.equal(p, before[k]) for k, p in trainer.model.state_dict().items())
    x = np.asarray(replay.observations[0]); z = trainer.model.encode(trainer.tensor(x[None]))
    reward, successor = mean_reward(trainer, z, torch.tensor([0]), 1.)
    assert reward.shape == (1,) and successor.shape == z.shape
    for search in ['mcts', 'shooting']:
        cfg.planner.search = search
        planner = make_planner(trainer, cfg.planner, 10)
        assert 0 <= planner.act(x) < 5
    diag = CadenceDiagnostics(trainer)
    assert 'mse' in diag.measure(diag.batches(replay), 0)['probe']
    result = evaluate(trainer, replay, tmp_path/'evaluation')
    assert result['oracle_all_state_actions'] > 0 and result['replay_normalized_mse'] >= 0
    assert not any('nll' in key or 'predictive_variance' in key for key in result)
