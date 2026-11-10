from types import SimpleNamespace
import numpy as np
import torch
from map_learning.reward_snapshots import exhaustive_q


def test_exhaustive_search_matches_analytic_two_step_problem():
    class Dynamics:
        actions=5
        def distribution(self, z, a):
            reward=z[:,0]+a
            successor=z+a[:,None]
            delta=(1.5*reward).sqrt()[:,None]
            means=torch.stack([successor-delta,successor,successor+delta])[:,:,None,:]
            return torch.zeros(3,len(a),1),means,torch.ones_like(means)
    trainer=SimpleNamespace(dynamics=Dynamics())
    q=exhaustive_q(trainer,torch.zeros(1,1),2,.9,1.)
    torch.testing.assert_close(q[0],torch.arange(5)*1.9+3.6)


def test_recording_does_not_change_actions_training_or_rng(tmp_path, monkeypatch):
    from map_learning.experiment import online
    from map_learning.config import load_config
    import map_learning.experiment as exp
    monkeypatch.setattr(exp,'evaluate',lambda *args,**kwargs:{})
    cfg=load_config('configs/joint_exploration_base.json',['maze.size=3','model.hidden_dim=16',
        'collect_steps=43','warmup_steps=10','train_every=5','training_schedule=new_then_replay'])
    plain=online(cfg,tmp_path/'plain')
    cfg.reward_snapshot_every=10
    recorded=online(cfg,tmp_path/'recorded')
    with np.load(plain/'replay.npz') as a,np.load(recorded/'replay.npz') as b:
        for k in ['actions','observations','next_observations','positions']:
            np.testing.assert_array_equal(a[k],b[k])
    a=torch.load(plain/'checkpoint.pt',weights_only=True);b=torch.load(recorded/'checkpoint.pt',weights_only=True)
    for section in ['model','dynamics']:
        for key in a[section]:torch.testing.assert_close(a[section][key],b[section][key],rtol=0,atol=0)
    torch.testing.assert_close(a['torch_rng_state'],b['torch_rng_state'])
    assert a['rng_state']==b['rng_state']
    with np.load(recorded/'reward_snapshots.npz') as data:
        np.testing.assert_array_equal(data['step'],[0,10,20,30,40])
        assert np.isfinite(data['reward']).all() and np.isfinite(data['exact_q']).all()
        assert data['decision'][0]=='warmup'
        assert data['reward'].shape[2]==5
