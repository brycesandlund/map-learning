import copy
import json
import numpy as np
import pytest
import torch
from map_learning.cadence import TrainingSchedule, CadenceDiagnostics
from map_learning.config import load_config
from map_learning.training import Trainer, seed_all
from test_learning import line_replay


def config(schedule):
    return load_config('configs/probabilistic_maze.json', [f'training_schedule={schedule}',
        'train.batch_size=13', 'model.hidden_dim=16', 'maze.size=3'])


@pytest.mark.parametrize('mode', ['epoch', 'new_then_replay'])
def test_every_transition_reaches_every_head_including_partial_batches(mode):
    seed_all(4)
    trainer = Trainer(config(mode), (2,))
    replay = line_replay()
    schedule = TrainingSchedule(trainer)
    recorded = [[] for _ in trainer.dynamics.heads]
    original = trainer.update_probabilistic
    def record(replay, head_indices=None):
        if head_indices is not None:
            for dst, indices in zip(recorded, head_indices):
                dst.extend(indices.tolist())
        return original(replay, head_indices)
    trainer.update_probabilistic = record
    metrics = schedule.train(replay)
    for indices in recorded:
        assert sorted(indices) == list(range(80))
    assert recorded[0] != recorded[1]
    assert metrics['block_updates'] == 7+(2 if mode == 'new_then_replay' else 0)
    for indices in recorded:
        indices.clear()
    for _ in range(3):
        replay.add(np.array([0., 0.]), 0, np.array([.2, 0.]), 8, [0., 0.], [.2, 0.])
    metrics = schedule.train(replay)
    for indices in recorded:
        assert sorted(indices) == list(range(0 if mode == 'epoch' else 80, 83))
    assert metrics['guaranteed_trained_through'] == 83


def test_diagnostics_do_not_change_training_or_random_state():
    seed_all(2)
    trainer = Trainer(config('replay'), (2,))
    replay = line_replay()
    state = torch.get_rng_state().clone()
    rng = copy.deepcopy(trainer.rng.bit_generator.state)
    weights = {k: v.clone() for k, v in trainer.model.state_dict().items()}
    diagnostics = CadenceDiagnostics(trainer)
    batches = diagnostics.batches(replay)
    assert diagnostics.measure(batches, 80) == diagnostics.measure(batches, 80)
    assert torch.equal(state, torch.get_rng_state())
    assert trainer.rng.bit_generator.state == rng
    assert trainer.updates == 0 and len(replay) == 80
    for k, v in trainer.model.state_dict().items():
        assert torch.equal(weights[k], v)


@pytest.mark.parametrize('mode', ['epoch', 'new_then_replay'])
def test_online_flushes_final_partial_block_before_warmup(monkeypatch, tmp_path, mode):
    from map_learning import experiment
    monkeypatch.setattr(experiment, 'evaluate', lambda *a, **k: {})
    cfg = config(mode)
    cfg.collect_steps, cfg.warmup_steps, cfg.train_every = 23, 100, 10
    cfg.cadence_diagnostics = True
    path = experiment.online(cfg, tmp_path/mode)
    rows = [json.loads(s) for s in (path/'cadence.jsonl').read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]['environment_step'] == 23
    assert rows[0]['guaranteed_trained_through'] == 23
    assert rows[0]['explicit_transitions_per_head'] == 23
    assert rows[0]['block_updates'] == 2+(2 if mode == 'new_then_replay' else 0)


def test_recovery_budget_matches_actual_schedules_with_final_partial_block():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location('recover', Path(__file__).parents[1]/'scripts/recover_evaluation.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for mode in ('replay', 'epoch', 'new_then_replay'):
        cfg = config(mode)
        cfg.collect_steps, cfg.warmup_steps, cfg.train_every = 83, 21, 10
        trainer = Trainer(cfg, (2,))
        schedule = TrainingSchedule(trainer)
        from map_learning.replay import Replay
        replay = Replay()
        for step in range(1, 84):
            replay.add(np.array([0., 0.]), 0, np.array([.2, 0.]), 0, [0., 0.], [.2, 0.])
            if (step >= 21 and step % 10 == 0) or (step == 83 and mode != 'replay'):
                schedule.train(replay)
        assert module.expected_updates(cfg) == trainer.updates
