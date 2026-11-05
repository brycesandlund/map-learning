"""Evaluation-only nearest-latent retrieval of the next physical maze position.

Compares learned coordinate systems using a common physical target. The complete
maze catalogue is privileged evaluator data and is never used by the agent.
This evaluates a simple nearest-latent readout, not an optimal decoder or policy.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from map_learning.training import Trainer, seed_all
from map_learning.maze import Maze, reachable_graph


def evaluate(path):
    path=Path(path)
    trainer=Trainer.load(path/'checkpoint.pt')
    points,adjacency,_=reachable_graph(Maze(trainer.config.maze))
    with np.load(path/'probe.npz') as probe:
        np.testing.assert_allclose(points,probe['points'])
        z=trainer.tensor(probe['latents'])
        seen=probe['seen'].copy()
    source=np.repeat(np.arange(len(points)),5)
    target=adjacency.reshape(-1)
    actions=np.tile(np.arange(5),len(points))
    predictions=[]
    with torch.no_grad():
        for start in range(0,len(source),256):
            sl=slice(start,start+256)
            means=trainer.dynamics(z[source[sl]],torch.as_tensor(actions[sl],device=trainer.device)).mean(0)
            predictions.extend(torch.cdist(means,z,compute_mode="donot_use_mm_for_euclid_dist").argmin(-1).cpu().tolist())
    predictions=np.asarray(predictions)
    errors=np.linalg.norm(points[predictions]-points[target],axis=-1)
    moving=np.linalg.norm(points[adjacency]-points[:,None],axis=-1)>.01
    values=z.cpu().numpy()
    latent_step=np.linalg.norm(values[adjacency]-values[:,None],axis=-1)
    horizontal=latent_step[:,:2][moving[:,:2]]
    vertical=latent_step[:,2:4][moving[:,2:4]]
    result={'run':str(path.resolve()),
            'mean_horizontal_latent_step':float(horizontal.mean()),
            'mean_vertical_latent_step':float(vertical.mean()),
            'horizontal_vertical_step_ratio':float(horizontal.mean()/max(vertical.mean(),1e-12))}
    for name,mask in [('all',np.ones(len(target),dtype=bool)),('seen',seen),('unseen',~seen)]:
        result[name]={'count':int(mask.sum()),'exact_next_position_accuracy':float((predictions[mask]==target[mask]).mean()) if mask.any() else None,
                      'mean_physical_error':float(errors[mask].mean()) if mask.any() else None,
                      'physical_error_p95':float(np.quantile(errors[mask],.95)) if mask.any() else None,
                      'error_over_one_room_fraction':float((errors[mask]>1).mean()) if mask.any() else None,
                      'persistence_accuracy':float((source[mask]==target[mask]).mean()) if mask.any() else None,
                      'persistence_mean_physical_error':float(np.linalg.norm(points[source[mask]]-points[target[mask]],axis=-1).mean()) if mask.any() else None}
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--summaries',nargs='+',required=True)
    parser.add_argument('--out',required=True)
    args=parser.parse_args()
    seed_all(123,1)
    rows=[r for f in args.summaries for r in json.loads(Path(f).read_text())]
    results=[]
    for row in rows:
        results.append({'variant':row['variant'],'seed':row['seed'],'environment_steps':row['metrics']['environment_steps'],**evaluate(row['run'])})
        Path(args.out).write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()
