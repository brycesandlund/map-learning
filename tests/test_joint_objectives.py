import numpy as np
import torch
from map_learning.config import load_config
from map_learning.replay import Replay
from map_learning.training import Trainer,seed_all
from map_learning.probabilistic import repulsion_loss


def make():
    c=load_config('configs/continuous_frontier.json');c.model.hidden_dim=16;c.model.hidden_layers=1;c.train.batch_size=8
    seed_all(42);t=Trainer(c,(10,));r=Replay();rng=np.random.default_rng(3)
    for i in range(20):r.add(rng.normal(size=10),rng.uniform(-1,1,2),rng.normal(size=10),0,[0,0],[0,0])
    return t,r


def test_joint_loss_formula_and_exact_resume(tmp_path):
    t,r=make();c=t.config.train;c.dynamics_weight=.2;c.adjacency_weight=.7;c.repulsion_weight=.1;c.latent_l2=.03;c.repulsion_margin=.5
    t.save(tmp_path/'checkpoint.pt');restored=Trainer.load(tmp_path/'checkpoint.pt')
    m=t.update_batch(r,[np.arange(8)]*3);n=restored.update_batch(r,[np.arange(8)]*3)
    assert m==n
    expected=.2*m['prediction_loss']+.7*m['adjacency_loss']+.1*m['repulsion']+.03*m['latent_squared_norm']
    assert np.isclose(m['joint_loss'],expected)
    for a,b in zip(t.model.parameters(),restored.model.parameters()):assert torch.equal(a,b)


def test_adjacency_changes_encoder_without_changing_initial_dynamics_gradient(tmp_path):
    t,r=make();t.save(tmp_path/'base.pt');joint=Trainer.load(tmp_path/'base.pt');joint.config.train.adjacency_weight=.3
    t.update_batch(r,[np.arange(8)]*3);joint.update_batch(r,[np.arange(8)]*3)
    assert any(not torch.equal(a,b) for a,b in zip(t.model.parameters(),joint.model.parameters()))
    # Adjacency does not directly differentiate through dynamics parameters.
    for a,b in zip(t.dynamics.parameters(),joint.dynamics.parameters()):assert torch.equal(a,b)


def test_repulsion_margin_is_a_distance_squared_inside_hinge():
    x=torch.tensor([[0.,0.]]);y=torch.tensor([[.5,0.]])
    assert repulsion_loss(x,y,1).item()==.75
    assert repulsion_loss(x,y,.5).item()==0
    assert repulsion_loss(x,y,2).item()==3.75
