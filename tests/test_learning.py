from dataclasses import asdict
import numpy as np
import torch
from map_learning.config import load_config
from map_learning.losses import qrl_loss, time_myopic_loss, dual_ascent
from map_learning.models import PrototypeBottleneck
from map_learning.replay import Replay
from map_learning.training import Trainer, seed_all
from map_learning.evaluation import split_replay, correlation_or_none


def line_replay():
    replay = Replay()
    for episode in range(8):
        for x in range(5):
            obs = np.array([x/5, 0], dtype=np.float32)
            nxt = np.array([(x+1)/5, 0], dtype=np.float32)
            replay.add(obs, 0, nxt, episode, obs, nxt)
        for x in range(5, 0, -1):
            obs = np.array([x/5, 0], dtype=np.float32)
            nxt = np.array([(x-1)/5, 0], dtype=np.float32)
            replay.add(obs, 1, nxt, episode, obs, nxt)
    return replay


def test_qrl_primal_and_dual_directions():
    pair = torch.tensor([2.], requires_grad=True)
    edge = torch.tensor([3.], requires_grad=True)
    multiplier = torch.tensor(2.)
    loss, stats = qrl_loss(pair, edge, multiplier, .1, 0)
    loss.backward()
    assert pair.grad < 0 and edge.grad > 0
    dual_ascent(multiplier, stats["constraint"].detach(), .1)
    assert multiplier > 2
    dual_ascent(multiplier, torch.tensor(-100.), 1)
    assert multiplier == 0
    _, stats = qrl_loss(pair, torch.tensor([.5]), multiplier, .1)
    assert stats["violation"] == 0


def test_time_myopic_caps_at_horizon():
    assert time_myopic_loss(torch.tensor([0., .25, 1., 1.]), torch.tensor([0., 5., 20., 100.]), 20) == 0


def test_collapsed_predictions_have_no_correlation_instead_of_nan():
    assert correlation_or_none(np.full(2048, 7.6, dtype=np.float32), np.arange(2048)) is None
    assert correlation_or_none([0, 1, 2], [2, 4, 6]) == 1


def test_pixel_and_nuisance_observation_training():
    from map_learning.maze import Maze
    for observation, nuisance in [("pixels", 0), ("features", 3)]:
        cfg = load_config(overrides=["maze.size=3", f"maze.observation={observation}",
                                    f"maze.nuisance_dims={nuisance}", "train.batch_size=8", "model.hidden_dim=16"])
        env = Maze(cfg.maze)
        replay = Replay()
        obs = env.reset()
        for action in [0, 2, 1, 3]*3:
            position = env.position.copy()
            nxt, _ = env.step(action)
            replay.add(obs, action, nxt, 0, position, env.position)
            obs = nxt
        trainer = Trainer(cfg, env.observation_shape)
        assert np.isfinite(trainer.update(replay)["map_loss"])


def test_pairs_never_cross_reset_and_match_elapsed_time():
    replay = Replay()
    for episode, length in [(0, 7), (1, 4), (2, 9)]:
        for step in range(length):
            x, y = np.array([episode, step]), np.array([episode, step+1])
            replay.add(x, 0, y, episode, x, y)
    x, y, k = replay.sample_pairs(2000, np.random.default_rng(0), 3, 10)
    np.testing.assert_array_equal(x[:, 0], y[:, 0])
    np.testing.assert_array_equal(y[:, 1]-x[:, 1], k)
    assert (k == 0).any() and (k > 3).any()
    train, test = split_replay(replay)
    assert set(train.episodes).isdisjoint(test.episodes)


def test_exact_revisit_minimum_is_observation_based():
    replay = line_replay()
    x, y, k = replay.sample_pairs(1000, np.random.default_rng(0), 5, 10, exact_min=True)
    np.testing.assert_allclose(k, np.rint(np.abs(x[:, 0]-y[:, 0])*5))


def test_replay_roundtrip(tmp_path):
    replay = line_replay()
    config = load_config()
    replay.save(tmp_path/"data.npz", config.maze)
    loaded, maze = Replay.load(tmp_path/"data.npz")
    assert maze == asdict(config.maze)
    for key, value in replay.arrays().items():
        np.testing.assert_array_equal(value, loaded.arrays()[key])


def test_codebook_hard_forward_and_encoder_gradient():
    bottle = PrototypeBottleneck(2, 2, .1)
    z = torch.tensor([[0., 0.], [0.1, 0.], [3., 2.]], requires_grad=True)
    bottle.initialize(z.detach())
    q, losses, ids = bottle(z)
    assert len(torch.unique(q, dim=0)) <= 2
    assert ids[0] == ids[1] and ids[0] != ids[2]
    (q.sum()+sum(losses.values())).backward()
    assert z.grad.abs().sum() > 0 and bottle.codes.grad.abs().sum() > 0


def test_qrl_and_temporal_learn_nonconstant_line_distances():
    replay = line_replay()
    for kind in ["qrl", "time_myopic"]:
        config = load_config(overrides=[f"train.loss={kind}", "model.latent_dim=16", "model.hidden_dim=32",
                                       "model.ensemble_size=2", "train.batch_size=64", "train.horizon=5",
                                       "train.max_pair_steps=10", "train.exact_revisit_min=true"])
        seed_all(0)
        trainer = Trainer(config, (2,))
        for _ in range(300):
            trainer.update(replay)
        with torch.no_grad():
            z = trainer.model.encode(trainer.tensor([[0., 0.], [1., 0.]]))
            distance = trainer.model.distance(z[:1], z[1:]).item()
        assert distance > 2.5, (kind, distance)


def test_checkpoint_continuation_matches_uninterrupted(tmp_path):
    config = load_config(overrides=["model.codebook_size=4", "model.hidden_dim=32", "train.batch_size=16"])
    seed_all(5)
    trainer = Trainer(config, (2,))
    replay = line_replay()
    for _ in range(3):
        trainer.update(replay)
    path = tmp_path/"checkpoint.pt"
    trainer.save(path)
    expected = trainer.update(replay)
    restored = Trainer.load(path)
    actual = restored.update(replay)
    assert actual == expected
    for key, value in trainer.model.state_dict().items():
        torch.testing.assert_close(value, restored.model.state_dict()[key], rtol=0, atol=0)
