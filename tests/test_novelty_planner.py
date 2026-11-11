from types import SimpleNamespace
import numpy as np
import torch
from map_learning.novelty_planning import NoveltyPlanner
from map_learning.uncertainty_planning import SearchNode


def stub(mode):
    planner=object.__new__(NoveltyPlanner)
    planner.trainer=SimpleNamespace(dynamics=SimpleNamespace(actions=3),device='cpu')
    planner.mode=mode;planner.reward_variance=1.;planner.step_scale=1.
    planner.root_state=torch.tensor([[0.,0.]]);planner.archive=torch.tensor([[0.,0.],[1.,0.]])
    def distribution(state,actions):
        means=torch.tensor([[0.,0.],[1.,0.],[2.,0.]])[None,:,None,:].expand(3,-1,-1,-1)
        return (torch.zeros(3,3,1),means,torch.ones_like(means)),torch.ones(3)
    planner.distribution_reward=distribution
    node=SearchNode(torch.tensor([[1.,0.]]),3);node.history=torch.tensor([[0.,0.],[1.,0.]])
    return planner,node


def test_archive_and_path_rewards_distinguish_known_repeats_and_new_points():
    for mode in ['path_distance','archive_path']:
        planner,node=stub(mode);planner.prepare(node)
        np.testing.assert_allclose(node.rewards,[0,0,1/np.sqrt(2)])
    planner,node=stub('archive_path');node.history=torch.tensor([[0.,0.]])
    planner.prepare(node);np.testing.assert_allclose(node.rewards,[0,0,1/np.sqrt(2)])


def test_root_distance_can_reward_a_repeat_but_soft_gate_suppresses_it():
    planner,node=stub('root_distance');planner.prepare(node)
    np.testing.assert_allclose(node.rewards,[0,1/np.sqrt(2),2/np.sqrt(2)])
    planner,node=stub('disagreement_path');planner.prepare(node)
    np.testing.assert_allclose(node.rewards,[0,0,.5])


def test_archive_refresh_reencodes_existing_observations():
    from map_learning.replay import Replay
    replay=Replay();replay.add(np.array([0.,0.],dtype=np.float32),0,np.array([1.,0.],dtype=np.float32),0,[0,0],[1,0])
    planner=object.__new__(NoveltyPlanner);planner.observations={};planner.mode='archive_path';scale=[1.]
    planner.trainer=SimpleNamespace(tensor=lambda a:torch.as_tensor(a),model=SimpleNamespace(encode=lambda x:x*scale[0]))
    planner.refresh(replay);assert len(planner.archive)==2
    scale[0]=2.;planner.refresh(replay);assert planner.archive.max()==2
    planner.observe(np.array([2.,0.],dtype=np.float32));assert len(planner.archive)==3
    planner.observe(np.array([2.,0.],dtype=np.float32));assert len(planner.archive)==3


def test_persistent_shooting_selects_progress_in_exact_dynamics():
    for mode in NoveltyPlanner.MODES:
        p=object.__new__(NoveltyPlanner);p.mode=mode;p.rng=np.random.default_rng(3)
        p.config=SimpleNamespace(candidates=64,horizon=3,action_repeat_probability=.8,uncertainty_discount=.99)
        p.trainer=SimpleNamespace(dynamics=SimpleNamespace(actions=2));p.archive=torch.zeros(1,1)
        p.reward_variance=1.;p.step_scale=1.;p.recent=None
        def distribution(state,actions):
            nxt=state+actions[:,None]
            means=nxt[None,:,None,:].expand(3,-1,-1,-1)
            return (torch.zeros(3,len(state),1),means,torch.ones_like(means)),torch.ones(len(state))
        p.distribution_reward=distribution
        assert p.shooting(torch.zeros(1,1))==1
        assert p.last_metrics['search_return']>0


def test_recent_memory_reencodes_and_clears_at_reset():
    from map_learning.replay import Replay
    p=object.__new__(NoveltyPlanner);p.mode='recent_archive';p.observations={};p.archive=torch.zeros(0,2)
    p.recent_observations=[];p.recent=None;scale=[1.]
    p.trainer=SimpleNamespace(tensor=lambda a:torch.as_tensor(a),model=SimpleNamespace(encode=lambda x:x*scale[0]))
    for i in range(20):p.observe(np.array([float(i),0.],dtype=np.float32))
    assert len(p.recent_observations)==16 and p.recent[0,0]==4
    replay=Replay();replay.add(np.array([0.,0.],dtype=np.float32),0,np.array([1.,0.],dtype=np.float32),0,[0,0],[1,0])
    scale[0]=2.;p.refresh(replay);assert p.recent[0,0]==8
    p.reset();assert p.recent is None and not p.recent_observations
