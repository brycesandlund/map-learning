import importlib.util
from pathlib import Path
import numpy as np
from map_learning.replay import Replay


def test_replay_online_sampling_matches_materialized_cache_after_appending():
    replay = Replay()
    for i in range(10):
        replay.add(np.ones((2, 3))*i, i%5, np.ones((2, 3))*(i+1), 0, [i, 0], [i+1, 0])
        a = np.random.default_rng(i)
        transitions = replay.sample_transitions(32, a)
        states = replay.sample_states(32, a)
        assert replay._arrays is None
        replay.arrays()
        b = np.random.default_rng(i)
        for actual, expected in zip(replay.sample_transitions(32, b), transitions):
            np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(replay.sample_states(32, b), states)


def test_coverage_auc_counts_every_interaction_and_resets():
    spec = importlib.util.spec_from_file_location('sweep', Path(__file__).parents[1]/'scripts/run_probabilistic_sweep.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Coverage after each action: 2/5, 3/5, 3/5, 4/5. Last source is a reset.
    positions = np.array([[0, 0], [1, 0], [2, 0], [3, 0]])
    next_positions = np.array([[1, 0], [2, 0], [2, 0], [3, 0]])
    result = module.coverage_statistics(positions, next_positions, 5)
    assert result['auc'] == .6 and result['final'] == .8
    assert result['to_50'] == 2 and result['to_80'] == 4 and result['to_95'] is None
