import importlib.util
from pathlib import Path
import numpy as np
spec=importlib.util.spec_from_file_location('novelty',Path(__file__).parents[1]/'scripts/analyze_novelty.py')
import sys
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_frontier_distances_and_tied_choices():
    adj=np.array([[1,0,0,0,0],[2,0,1,1,1],[2,1,2,2,2]])
    visited=np.array([True,True,False])
    np.testing.assert_array_equal(m.frontier_distances(adj,visited),[2,1,0])
    np.testing.assert_allclose(m.top_choice(np.ones((1,5)),np.array([[1,0,0,0,0]])),[.2])
    seen=np.zeros((3,5),bool);seen[0,0]=True
    score=np.array([[1,0,0,0,0],[1,0,0,0,0],[0,0,0,0,0]],float).ravel()
    out=m.edge_metrics(adj,visited,seen,score,score,score)
    assert out['direction']['frontier']['disagreement']==1
    assert out['direction']['near']['disagreement']==1
    assert out['groups']['visited_source']['new_successor_rate']==.1
    assert out['groups']['tried']['new_successor_auc']['disagreement'] is None


def test_no_frontier_when_all_states_seen():
    np.testing.assert_array_equal(m.frontier_distances(np.array([[0,1],[1,0]]),np.ones(2,bool)),[-1,-1])


def test_path_novelty_counts_unique_states_and_handles_score_ties():
    import torch
    from types import SimpleNamespace
    adj=np.array([[1,0,0,0,0],[2,0,1,1,1],[2,1,2,2,2]])
    z=torch.eye(3)
    class Perfect:
        def distribution(self,state,actions):
            source=state.argmax(-1).numpy()
            target=z[adj[source,actions.numpy()]]
            means=target[None,:,None,:].expand(3,-1,1,-1)
            return torch.zeros(3,len(state),1),means,torch.ones_like(means)
    result=m.paths(SimpleNamespace(dynamics=Perfect()),z,adj,np.array([True,True,False]),
                   np.zeros((3,5)),np.zeros((3,5)),1.,starts=2,candidates=16,horizon=4)
    for family in result['families'].values():
        row=family[0]
        assert 0 < row['best_unique_new_states'] <= 1
        for score in row['scores'].values():
            assert score['selected_unique_new_states']==row['random_unique_new_states']
