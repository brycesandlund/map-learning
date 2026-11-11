import numpy as np
import pytest
from map_learning.random_walk import RandomWalk


def test_walk_controls_and_stays():
    w=RandomWalk(9, .9, .2)
    actions=np.stack([w.action() for _ in range(10000)])
    assert actions.dtype == np.float32
    lengths=np.linalg.norm(actions,axis=1)
    assert .18 < np.mean(lengths==0) < .22
    assert np.all((lengths==0)|((lengths>=.499999)&(lengths<=1.000001)))


def test_collision_uses_observable_coordinates_not_nuisance():
    w=RandomWalk(4,.9,0,True);action=w.action()
    obs=np.zeros(10);nxt=np.r_[0.,0.,np.ones(8)]
    w.observe(obs,nxt,action);assert w.heading is None
    action=w.action();heading=w.heading;nxt[0]=.01
    w.observe(obs,nxt,action);assert w.heading==heading
    w.observe(obs,obs,np.zeros(2));assert w.heading==heading
    w.reset();assert w.heading is None


def test_walk_reproducibility_and_invalid_probabilities():
    a,b=RandomWalk(5,.9),RandomWalk(5,.9)
    assert np.array_equal([a.action() for _ in range(50)],[b.action() for _ in range(50)])
    for p in [-1,1,float('nan')]:
        with pytest.raises(ValueError):RandomWalk(1,p)
