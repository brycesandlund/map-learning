import numpy as np
from map_learning.persistent_frontier import PersistentDistanceFrontier, epoch_slices


def test_refresh_changes_geometry_preserves_all_entries():
    q=PersistentDistanceFrontier(20,2,(2,),np.random.default_rng(0))
    encode=lambda x:x*np.array([1,10])
    q.visit(np.array([0,0]),np.array([0,0]))
    q.add(np.array([3,0]),np.array([3,0]),'horizontal')
    q.add(np.array([0,2]),np.array([0,2]),'vertical')
    q.refresh(encode,batch_size=1)
    assert q.n_visited==1 and q.active.sum()==2
    obs,payload,z,score=q.pop()
    assert payload=='vertical' and score==20
    q.visit(obs,z)
    q.refresh(encode)
    assert q.n_visited==2 and q.active.sum()==1
    assert q.pop()[1]=='horizontal'


def test_refreshed_and_incremental_priorities_equal_brute_force():
    rng=np.random.default_rng(10)
    q=PersistentDistanceFrontier(100,4,(4,),rng)
    encode=lambda x: x @ np.array([[1,2,0,0],[0,1,0,0],[0,0,3,0],[0,0,0,1.]])
    v=rng.normal(size=4);q.visit(v,v)
    for i in range(30):
        x=rng.normal(size=4);q.add(x,x,i)
    q.refresh(encode)
    for step in range(20):
        ids=np.flatnonzero(q.active)
        exact=np.linalg.norm(q.latents[ids,None]-q.visited[None,:q.n_visited],axis=-1).min(1)
        np.testing.assert_allclose(q.priority[ids],exact,rtol=1e-12,atol=1e-12)
        obs,_,z,score=q.pop()
        assert np.isclose(score,exact.max())
        q.visit(obs,z)
        x=rng.normal(size=4);q.add(x,encode(x),100+step)
        if step%5==0:q.refresh(encode)


def test_interleaved_epoch_exact_exposure_and_gradient_budgets():
    rng=np.random.default_rng(0)
    for block in range(1,21):
        old_count=(block-1)*5000
        old_order=rng.permutation(old_count)
        batches=[]
        for j in range(10):
            parts=epoch_slices(old_count,old_count+500*j,old_count+500*(j+1),j,10,rng,old_order,block*5000,256)
            indices=np.concatenate(parts)
            assert indices.max()<old_count+500*(j+1)
            assert set(range(old_count+500*j,old_count+500*(j+1)))<=set(indices)
            batches.extend(parts)
        assert len(batches)==(block*5000+255)//256
        np.testing.assert_array_equal(np.sort(np.concatenate(batches)),np.arange(block*5000))
