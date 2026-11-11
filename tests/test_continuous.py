import numpy as np
import pytest
import torch
from map_learning.config import load_config, MazeConfig
from map_learning.continuous import sample_controls, SpatialCoverage, within_cell_fill
from map_learning.distance_learning import DistanceTrainer
from map_learning.maze import Maze
from map_learning.replay import Replay
from map_learning.training import Trainer, seed_all


def test_controls_are_bounded_continuous_and_reproducible():
    for mode in ['stratified','iid']:
        rng=np.random.default_rng(9)
        controls=np.concatenate([sample_controls(rng,mode) for _ in range(100)])
        lengths=np.linalg.norm(controls,axis=1)
        assert np.all((lengths==0)|((lengths>=.5)&(lengths<=1.000001)))
        assert not np.allclose(controls*4,np.rint(controls*4))
        assert np.array_equal(controls[:5],sample_controls(np.random.default_rng(9),mode))
        assert np.array_equal(controls[4::5],np.zeros((100,2)))


def test_continuous_swept_collision_and_free_motion():
    env=Maze(MazeConfig(size=3,step_size=.25))
    env.walls=np.array([[[1,0],[1,2]]],dtype=np.float32)
    p=np.array([.9,1.],dtype=np.float32)
    assert np.array_equal(env.transition(p,[.8,.7]),p)
    assert np.allclose(env.transition(p,[-.3,.7]),[.825,1.175])
    # Crossing a wall endpoint is a collision; just above is free.
    p=np.array([.9,2.],dtype=np.float32)
    assert np.array_equal(env.transition(p,[.8,0]),p)
    p=np.array([.9,2.01],dtype=np.float32)
    assert np.allclose(env.transition(p,[.8,0]),[1.1,2.01])
    assert np.allclose(env.transition([.01,1],[-.5,0]),[.01,1])


def test_coverage_is_multiresolution_not_unique_coordinate_count():
    c=SpatialCoverage(2)
    for i,p in enumerate([[.1,.1],[.11,.11],[1.1,.1],[.1,1.1],[1.1,1.1]]):c.add(p,i)
    assert c.metrics()['coverage_k1']==1
    assert c.metrics()['coverage_k4']==4/64
    assert c.milestones['k1_100']==4
    assert within_cell_fill([[.99,.5]],2)['fill_radius_0.2']<=.25


@pytest.mark.parametrize('distance',[False,True])
def test_vector_action_training_and_checkpoint_exact(distance,tmp_path):
    cfg=load_config('configs/continuous_frontier.json')
    cfg.collect_steps=20;cfg.model.hidden_dim=16;cfg.model.hidden_layers=1;cfg.train.batch_size=8
    seed_all(30)
    t=DistanceTrainer(cfg,(10,),{'kind':'adjacency','weight':.3}) if distance else Trainer(cfg,(10,))
    env=Maze(cfg.maze,1);r=Replay();rng=np.random.default_rng(6);obs=env.reset()
    if distance:t.ancestry.root(obs)
    for i in range(10):
        action=sample_controls(rng)[i%5];pos=env.position.copy();nxt,_=env.step(action)
        r.add(obs,action,nxt,0,pos,env.position)
        if distance:t.ancestry.add(i,nxt)
        obs=nxt
    r.save(tmp_path/'r.npz',cfg.maze);rr,_=Replay.load(tmp_path/'r.npz')
    assert rr.arrays()['actions'].shape==(10,2)
    assert np.array_equal(rr.arrays()['actions'],r.arrays()['actions'])
    idx=[np.arange(8)]*3
    t.update_batch(r,idx);t.save(tmp_path/'checkpoint.pt');restored=Trainer.load(tmp_path/'checkpoint.pt')
    assert t.update_batch(r,idx)==restored.update_batch(rr,idx)
    assert all(torch.equal(a,b) for a,b in zip(t.model.parameters(),restored.model.parameters()))
    z=torch.zeros(2,8);u=torch.tensor([[.5,.2],[-.5,-.2]])
    assert not torch.equal(t.dynamics(z,u)[:,0],t.dynamics(z,u)[:,1])
    if distance:
        with torch.no_grad():
            for p in restored.dynamics.parameters():p.add_(1.)
        t.update_batch(r,idx);restored.update_batch(r,idx)
        assert all(torch.equal(a,b) for a,b in zip(t.model.parameters(),restored.model.parameters()))
