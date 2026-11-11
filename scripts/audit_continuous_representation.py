"""Read-only checkpoint probes; oracle positions/noise interventions are evaluation only."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from scipy.stats import spearmanr
from map_learning.continuous import sample_controls
from map_learning.maze import Maze
from map_learning.training import Trainer


@torch.no_grad()
def audit(path, checkpoint, samples=2048):
    trainer=Trainer.load(path/checkpoint);cfg=trainer.config;env=Maze(cfg.maze,84213)
    rng=np.random.default_rng(78241)
    positions=rng.uniform(.001,cfg.maze.size-.001,(samples,2)).astype(np.float32)
    action=np.stack([sample_controls(rng)[i%4] for i in range(samples)])
    next_positions=np.stack([env.transition(x,a) for x,a in zip(positions,action)])
    obs=np.stack([env.observe(x) for x in positions]);nxt=np.stack([env.observe(x) for x in next_positions])
    repeat=np.stack([env.observe(x) for x in positions])
    z=trainer.model.encode(trainer.tensor(obs));target=trainer.model.encode(trainer.tensor(nxt))
    repeat_z=trainer.model.encode(trainer.tensor(repeat))
    pred=trainer.dynamics(z,trainer.tensor(action));mean_pred=pred.mean(0)
    moving=np.linalg.norm(next_positions-positions,axis=1)>1e-7
    noise=(z-repeat_z).square().sum(-1).numpy()
    transition=(z-target).square().sum(-1).numpy()
    error=(mean_pred-target).square().sum(-1).numpy()
    disagreement=pred.var(0,unbiased=False).sum(-1).numpy()
    # Uniform physical probes, fresh observation noise: not a replay accuracy metric.
    metrics={'run':str(path),'checkpoint':checkpoint,'updates':trainer.updates,
        'noise_rms':float(np.sqrt(noise.mean())),
        'open_transition_rms':float(np.sqrt(transition[moving].mean())),
        'noise_to_open_transition_rms':float(np.sqrt(noise.mean()/max(transition[moving].mean(),1e-12))),
        'transition_error_rms':float(np.sqrt(error.mean())),
        'stay_predictor_error_rms':float(np.sqrt(transition.mean())),
        'blocked_error_rms':float(np.sqrt(error[~moving].mean())),
        'open_error_rms':float(np.sqrt(error[moving].mean())),
        'mean_disagreement':float(disagreement.mean()),'blocked_fraction':float((~moving).mean())}
    # Mean over noise to expose shape separately from per-observation sensitivity.
    axis=(np.arange(cfg.maze.size*2)+.5)/2
    grid=np.stack(np.meshgrid(axis,axis),-1).reshape(-1,2)
    observations=np.stack([env.observe(x) for x in grid for _ in range(16)])
    latent=torch.cat([trainer.model.encode(trainer.tensor(chunk)) for chunk in np.array_split(observations,16)]).numpy()
    centroids=latent.reshape(len(grid),16,-1).mean(1)
    pairs=rng.integers(len(grid),size=(10000,2))
    dxy=np.linalg.norm(grid[pairs[:,0]]-grid[pairs[:,1]],axis=1)
    dz=np.linalg.norm(centroids[pairs[:,0]]-centroids[pairs[:,1]],axis=1)
    metrics['centroid_distance_spearman_xy']=float(spearmanr(dxy,dz).statistic)
    return metrics,{'points':grid,'latents':latent,'centroids':centroids}


def main():
    p=argparse.ArgumentParser();p.add_argument('runs',nargs='+',type=Path);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--checkpoints',nargs='+',default=['checkpoint-5.pt','checkpoint-20.pt']);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1);rows=[]
    for path in a.runs:
        for checkpoint in a.checkpoints:
            r,arrays=audit(path,checkpoint);rows.append(r)
            np.savez_compressed(a.out/f'{path.name}-{checkpoint}.npz',**arrays)
            print(json.dumps(r),flush=True)
    (a.out/'results.json').write_text(json.dumps(rows,indent=2)+'\n')

if __name__=='__main__':main()
