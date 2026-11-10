from types import SimpleNamespace
import numpy as np
import torch
from map_learning.config import load_config, PlannerConfig
from map_learning.training import Trainer, seed_all
from map_learning.uncertainty_planning import UncertaintyPlanner
from test_learning import line_replay


def test_mean_search_uses_disagreement_even_when_reward_is_tiny_and_ignores_variances():
    class Identity:
        def encode(self,x):return x
    class Dynamics:
        actions=5
        def __init__(self,variance):self.variance=variance
        def distribution(self,state,actions):
            mu=state[None,:,None,:].repeat(3,1,1,1)
            mu[1,actions==2]+=1e-5
            return torch.zeros(3,len(state),1),mu,torch.ones_like(mu)*self.variance
    for variance in (1e-10,1e8):
        trainer=SimpleNamespace(config=SimpleNamespace(train=SimpleNamespace(prediction_objective='mse')),
            device='cpu',model=Identity(),dynamics=Dynamics(variance),tensor=lambda x:torch.as_tensor(x,dtype=torch.float32))
        planner=UncertaintyPlanner(trainer,PlannerConfig(behavior='disagreement',search='mcts',horizon=1,simulations=24,epsilon_random=0),seed=7)
        assert planner.act(np.array([0.,0.]),latent_scale=.5)==2
        params=trainer.dynamics.distribution(torch.zeros(1,2),torch.tensor([2]))
        a=planner.successor(params);b=planner.successor(params)
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        assert a.abs().max()<1e-4


def test_joint_mse_optimizer_rates_and_exact_checkpoint_continuation(tmp_path):
    seed_all(5)
    cfg=load_config('configs/probabilistic_maze.json',['train.prediction_objective=mse',
        'train.encoder_lr_multiplier=0.3','model.mixture_components=1','model.latent_activation=identity',
        'train.repulsion_weight=0.01','model.hidden_dim=32'])
    trainer=Trainer(cfg,(2,));replay=line_replay()
    before=[p.detach().clone() for p in trainer.model.parameters()]
    assert trainer.map_optimizer.param_groups[0]['lr']==.3*cfg.train.lr
    for _ in range(10):r=trainer.update(replay)
    assert r['nll_per_dim'] is None and r['predictive_variance'] is None
    assert any(not torch.equal(a,b) for a,b in zip(before,trainer.model.parameters()))
    trainer.save(tmp_path/'checkpoint.pt')
    expected=trainer.update(replay)
    restored=Trainer.load(tmp_path/'checkpoint.pt')
    assert restored.update(replay)==expected


def test_novelty_auc_handles_ties_and_local_state_mask():
    import importlib.util
    from pathlib import Path
    spec=importlib.util.spec_from_file_location('calibration',Path(__file__).parents[1]/'scripts/analyze_exploration_uncertainty.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    positive=np.array([False,False,True,True])
    assert mod.auc([0,1,2,3],positive)==1
    assert mod.auc([3,2,1,0],positive)==0
    assert mod.auc([1,1,1,1],positive)==.5
    assert mod.auc([0,2,1,3],positive,np.array([False,True,True,False]))==0
    assert mod.auc([0,1,2,3],positive,positive) is None
