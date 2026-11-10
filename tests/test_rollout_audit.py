import importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from map_learning.config import load_config
from map_learning.maze import Maze,reachable_graph


def test_perfect_dynamics_has_no_compounding_error():
    spec=importlib.util.spec_from_file_location('audit',Path(__file__).parents[1]/'scripts/audit_latent_rollouts.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    cfg=load_config(overrides=['maze.size=3'])
    env=Maze(cfg.maze);points,adj,index=reachable_graph(env)
    z=torch.tensor(np.stack([env.observe(p) for p in points]))
    class Dynamics:
        heads=[None]*3
        def distribution(self,state,actions):
            rows=torch.cdist(state,z).argmin(-1);target=z[adj[rows.numpy(),actions.numpy()]]
            means=target[None,:,None,:].expand(3,-1,1,-1)
            return torch.zeros(3,len(state),1),means,torch.ones_like(means)
    trainer=SimpleNamespace(config=cfg,model=SimpleNamespace(encode=lambda x:x),tensor=lambda x:torch.tensor(x),dynamics=Dynamics())
    rng=np.random.default_rng(3);source=[];actions=rng.integers(5,size=40);position=env.start
    for action in actions:
        source.append(position);position=env.transition(position,action)
    replay={'positions':np.array(source),'actions':actions,'episodes':np.zeros(40)}
    result=module.audit(trainer,replay,40,horizon=8,batch=8)
    for rows in result['curves'].values():
        for row in rows:
            assert row['normalized_mse']<1e-12
            assert row['nearest_state_accuracy']==1
            assert row['position_error_steps_mean']==0
