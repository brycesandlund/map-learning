import copy
import numpy as np
import pytest
import torch
from map_learning.config import load_config
from map_learning.distance_learning import Ancestry, DistanceTrainer, student_neighbor_loss
from map_learning.training import Trainer, seed_all
from map_learning.replay import Replay


def setup(kind):
    cfg = load_config('configs/deterministic_frontier.json')
    cfg.collect_steps = 30; cfg.model.hidden_dim = 16; cfg.model.hidden_layers = 1
    cfg.train.batch_size = 8
    seed_all(18)
    t = DistanceTrainer(cfg, (10,), {'kind': kind})
    replay = Replay(); rng = np.random.default_rng(3)
    root = rng.normal(size=10).astype(np.float32); t.ancestry.root(root)
    for i in range(20):
        parent = i//3
        obs = t.ancestry.observations[parent].copy()
        nxt = rng.normal(size=10).astype(np.float32)
        t.ancestry.add(parent, nxt)
        replay.add(obs, i%5, nxt, i, [0,0], [0,0])
    return t, replay


def test_ancestry_follows_parent_not_expansion_order():
    tree = Ancestry(6, (1,)); tree.root([0])
    tree.add(0,[1]); tree.add(0,[2]); tree.add(1,[3]); tree.add(2,[4])
    class FixedRng:
        def random(self,n): return np.full(n,.999)
    a,b,k=tree.pairs(np.array([2,3]),FixedRng(),1)
    assert np.array_equal(a[:,0], [1,0])
    assert np.array_equal(b[:,0], [3,4])
    assert np.array_equal(k, [1,2])
    with pytest.raises(ValueError):tree.add(99,[9])


def test_affinity_probabilities_and_gradient():
    x = torch.tensor([[0.,0.],[2.,0.]],requires_grad=True)
    y = torch.tensor([[.1,0.],[2.1,0.]],requires_grad=True)
    good = student_neighbor_loss(x,y)
    bad = student_neighbor_loss(x,y.flip(0))
    assert 0 <= good < bad
    good.backward()
    assert torch.isfinite(x.grad).all() and torch.isfinite(y.grad).all()
    assert student_neighbor_loss(torch.zeros(2,3), torch.zeros(2,3)) == pytest.approx(np.log(3))


@pytest.mark.parametrize('kind',['adjacency','qrl_upper','temporal','temporal_upper','affinity'])
def test_resume_exact_and_auxiliary_cannot_shape_encoder(kind,tmp_path):
    t,r = setup(kind)
    idx = [np.arange(8)]*3
    path = tmp_path/'test.pt'; t.save(path)
    restored=Trainer.load(path)
    # Corrupt only auxiliary heads. Encoder's next update must be identical.
    with torch.no_grad():
        for p in restored.dynamics.parameters():p.add_(.1)
    m=t.update_batch(r,idx); restored.update_batch(r,idx)
    assert np.isfinite(list(m.values())).all()
    for a,b in zip(t.model.parameters(), restored.model.parameters()):assert torch.equal(a,b)
    t.save(path); restored=Trainer.load(path)
    m1=t.update_batch(r,idx);m2=restored.update_batch(r,idx)
    assert m1==m2
    for a,b in zip(t.model.parameters(), restored.model.parameters()):assert torch.equal(a,b)


def test_initialization_matches_native_control():
    t,_=setup('adjacency')
    seed_all(18); base=Trainer(t.config,(10,))
    for a,b in zip(t.model.parameters(),base.model.parameters()):assert torch.equal(a,b)
    for a,b in zip(t.dynamics.parameters(),base.dynamics.parameters()):assert torch.equal(a,b)


def test_temporal_upper_does_not_push_apart_shorter_paths():
    from map_learning.distance_learning import temporal_stress
    distance=torch.tensor([.1],requires_grad=True);lag=torch.tensor([8.])
    exact=temporal_stress(distance,lag,16);exact.backward()
    assert distance.grad.item()<0  # Regression would increase their separation.
    distance.grad.zero_()
    upper=temporal_stress(distance,lag,16,True);upper.backward()
    assert upper.item()==0 and distance.grad.item()==0
    assert temporal_stress(torch.tensor([2.]),torch.tensor([1.]),16,True)>0
