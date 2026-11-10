"""Post-hoc calibration and behavior diagnostics; never used by the agent."""
import argparse
import json
from pathlib import Path
import numpy as np


def auc(scores, positive, eligible=None):
    mask=np.ones(len(scores),dtype=bool) if eligible is None else eligible
    scores=np.asarray(scores)[mask];positive=np.asarray(positive)[mask]
    n=int(positive.sum());m=len(scores)-n
    if not n or not m:return None
    _,inverse,counts=np.unique(scores,return_inverse=True,return_counts=True)
    ranks=(np.cumsum(counts)-(counts-1)/2)[inverse]
    return float((ranks[positive].sum()-n*(n+1)/2)/(n*m))


def analyze(row):
    path=Path(row['run'])
    with np.load(path/'probe.npz') as probe:
        points=probe['points'];source=probe['source'];seen=probe['seen'];score=probe['disagreement']
    with np.load(path/'replay.npz') as replay:
        pos=replay['positions'];nxt=replay['next_positions'];actions=replay['actions'];episodes=replay['episodes']
    lookup={tuple(np.round(p,5)):i for i,p in enumerate(points)}
    visited={lookup[tuple(np.round(p,5))] for p in np.concatenate([pos,nxt])}
    at_seen_state=np.isin(source,list(visited))
    still=np.all(pos==nxt,axis=1);same_episode=episodes[1:]==episodes[:-1]
    allpoints=np.stack([pos,nxt],axis=1).reshape(-1,2)
    _,first=np.unique(allpoints,axis=0,return_index=True)
    metric={'variant':row['variant'],'seed':row['seed'],'run':row['run'],
        'unseen_edge_auc':auc(score,~seen),
        'untried_action_at_visited_state_auc':auc(score,~seen,at_seen_state),
        'seen_disagreement_mean':float(score[seen].mean()),
        'untried_action_at_visited_state_disagreement_mean':float(score[~seen&at_seen_state].mean()) if (~seen&at_seen_state).any() else None,
        'unvisited_state_disagreement_mean':float(score[~at_seen_state].mean()) if (~at_seen_state).any() else None,
        'stationary_fraction':float(still.mean()),'noop_fraction':float((actions==4).mean()),
        'wall_block_fraction_given_movement_action':float(still[actions!=4].mean()),
        'action_switch_fraction':float((actions[1:]!=actions[:-1])[same_episode].mean()),
        'late_2000_new_positions':int((first//2>=len(pos)-2000).sum()),
        'untried_actions_at_visited_states':int((~seen&at_seen_state).sum())}
    return metric


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--summaries',nargs='+',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    args=p.parse_args()
    rows=[r for path in args.summaries for r in json.loads(path.read_text())]
    results=[analyze(r) for r in rows]
    args.out.write_text(json.dumps(results,indent=2)+'\n')
    for r in results:print(r['variant'],r['seed'],r['untried_action_at_visited_state_auc'],r['stationary_fraction'])


if __name__=='__main__':main()
