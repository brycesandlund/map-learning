import copy
import numpy as np
import torch
import pytest
from map_learning.cadence import TrainingSchedule
from map_learning.training import Trainer, seed_all
from test_cadence import config
from test_learning import line_replay


@pytest.mark.parametrize('schedule', ['epoch', 'replay', 'new_then_replay'])
def test_shared_batches_and_rng_alignment(schedule):
    states=[]
    for mode in ['independent', 'shared']:
        seed_all(4)
        cfg=config(schedule);cfg.train.ensemble_batches=mode
        trainer=Trainer(cfg,(2,));replay=line_replay();seen=[]
        original=replay.transitions_at
        def record(idx):
            seen.append(np.array(idx));return original(idx)
        replay.transitions_at=record
        scheduler=TrainingSchedule(trainer);scheduler.train(replay)
        heads=len(trainer.dynamics.heads)
        assert len(seen)%heads==0
        for offset in range(0,len(seen),heads):
            same=all(np.array_equal(seen[offset],seen[offset+i]) for i in range(heads))
            assert same == (mode=='shared')
        states.append((copy.deepcopy(trainer.rng.bit_generator.state),copy.deepcopy(scheduler.rng.bit_generator.state)))
        assert not all(torch.equal(a,b) for a,b in zip(trainer.dynamics.heads[0].parameters(),trainer.dynamics.heads[1].parameters()))
    assert states[0]==states[1]


def test_shared_replay_only_preserves_explicit_new_batches():
    seed_all(4)
    cfg=config('new_then_replay');cfg.train.ensemble_batches='shared_replay'
    trainer=Trainer(cfg,(2,));replay=line_replay();seen=[]
    original=replay.transitions_at
    def record(idx):
        seen.append(np.array(idx));return original(idx)
    replay.transitions_at=record
    supplied=[np.arange(i,i+8) for i in range(len(trainer.dynamics.heads))]
    trainer.update_probabilistic(replay,head_indices=supplied)
    for got,expected in zip(seen,supplied):np.testing.assert_array_equal(got,expected)
    seen.clear();trainer.update_probabilistic(replay)
    for got in seen:np.testing.assert_array_equal(got,seen[0])
