import importlib.util
from pathlib import Path
import numpy as np
import pytest
import torch

spec=importlib.util.spec_from_file_location('memorize',Path(__file__).parents[1]/'scripts/memorize_dynamics.py')
memorize=importlib.util.module_from_spec(spec)
spec.loader.exec_module(memorize)


def test_conflicting_transition_targets_are_not_silently_merged():
    obs=np.array([[0.,0.],[0.,0.],[0.,0.]])
    actions=np.array([0,0,1])
    nxt=np.array([[1.,0.],[1.,0.],[-1.,0.]])
    first,inverse,counts=memorize.unique_transitions(obs,actions,nxt)
    np.testing.assert_array_equal(nxt[first][inverse],nxt)
    assert sorted(counts)==[1,2]
    nxt[1]=[0.,1.]
    with pytest.raises(ValueError,match='conflicting'):
        memorize.unique_transitions(obs,actions,nxt)


def test_mse_objective_cannot_hide_residual_in_variance():
    means=torch.ones(3,1,2,requires_grad=True)
    params=(torch.zeros(3,1),means,torch.ones_like(means))
    target=torch.zeros(3,2)
    mse=memorize.prediction_loss(params,target,'mse')
    assert mse.item()==1
    assert memorize.prediction_loss((params[0],means,params[2]*100),target,'mse')==mse
    mse.backward()
    assert torch.all(means.grad>0)
    assert memorize.prediction_loss(params,target,'nll')!=memorize.prediction_loss((params[0],means,params[2]*100),target,'nll')


def test_deeper_models_train_jointly_and_old_checkpoint_shapes_are_preserved():
    from map_learning.config import load_config
    from map_learning.training import Trainer, seed_all
    from map_learning.models import mlp
    from test_learning import line_replay
    seed_all(7)
    legacy=mlp(2,64,8)
    assert list(legacy.state_dict())==['0.weight','0.bias','2.weight','2.bias','4.weight','4.bias']
    cfg=load_config('configs/probabilistic_maze.json',['model.hidden_layers=3','model.hidden_dim=32'])
    trainer=Trainer(cfg,(2,))
    assert len([m for m in trainer.model.encoder if isinstance(m,torch.nn.Linear)])==4
    assert len([m for m in trainer.dynamics.heads[0] if isinstance(m,torch.nn.Linear)])==4
    before=[p.detach().clone() for p in trainer.model.parameters()]
    trainer.update(line_replay())
    assert any(not torch.equal(a,b) for a,b in zip(before,trainer.model.parameters()))
