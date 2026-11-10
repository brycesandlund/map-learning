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
