"""Pixel-only persistent novelty queue and nonreactive walking controls."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree
from map_learning.config import MazeConfig
from map_learning.pixel_exploration import PixelMaze,PixelLearner
from map_learning.continuous import SpatialCoverage,sample_controls
from map_learning.training import seed_all


def parser():
    p=argparse.ArgumentParser()
    defaults=dict(checkpoint_every=25000,dynamics_switch=0,normalize=0,rgb=0,background=0,center=0,seed=110,layout=7,size=12,image_size=48,steps=20000,dim=8,width=128,
        warm_steps=0,train_every=1000,epoch_every=5000,batch=128,extra_updates=0,refresh_every=1000)
    for k,v in defaults.items():p.add_argument('--'+k.replace('_','-'),type=int,default=v)
    floats=dict(dynamics_after=1.,infonce=0.,augmentation=0.,temperature=10.,inverse=0.,marker_size=.85,variance=0.,covariance=0.,noise=.12,step_size=.25,lr=.0003,dynamics=1.,adjacency=0.,repulsion=.03,l2=.03,margin=.25,persistence=0.,reset_probability=0.)
    for k,v in floats.items():
        p.add_argument('--'+k.replace('_','-'),type=float,default=v,help='Spatial attention logit multiplier (inverse temperature).' if k=='temperature' else None)
    p.add_argument('--device',choices=['cpu','mps'],default='cpu')
    p.add_argument('--noise-mode',choices=['full','panel'],default='full')
    p.add_argument('--family',choices=['maze','snake','comb','open','rooms'],default='maze')
    p.add_argument('--architecture',choices=['cnn','mlp','spatial','keypoint'],default='cnn')
    p.add_argument('--policy',choices=['learned','frozen','random_queue','walk'],default='learned')
    p.add_argument('--controls',choices=['continuous','cardinal'],default='continuous')
    p.add_argument('--out',type=Path,required=True)
    return p


def main():
    a=parser().parse_args();a.out.mkdir(parents=True,exist_ok=False)
    assert a.architecture!='keypoint' or a.dim==2
    assert a.epoch_every%a.train_every==0
    assert a.steps%a.train_every==0 and a.train_every%5==0 and a.refresh_every%5==0
    assert 0<=a.persistence<1 and 0<=a.reset_probability<=1
    (a.out/'spec.json').write_text(json.dumps(vars(a),default=str,indent=2))
    (a.out/'runner.py').write_text(Path(__file__).read_text())
    import map_learning.pixel_exploration as pixel_module
    (a.out/'pixel_exploration.py').write_text(Path(pixel_module.__file__).read_text())
    seed_all(a.seed,1);start=time.monotonic()
    env=PixelMaze(MazeConfig(size=a.size,seed=a.layout,step_size=a.step_size,image_size=a.image_size),a.seed+71000,a.family,a.noise,a.rgb,a.marker_size,a.noise_mode)
    learner=PixelLearner(a)
    rng=np.random.default_rng(a.seed+101);crng=np.random.default_rng(a.seed+91000);qrng=np.random.default_rng(a.seed+72000)
    wrng=np.random.default_rng(a.seed+93000);rrng=np.random.default_rng(a.seed+94000)
    images=np.empty((a.steps+1,*env.observation_shape),np.uint8)
    positions=np.empty((a.steps+1,2),np.float32);parents=np.full(a.steps+1,-1,np.int32);actions=np.zeros((a.steps+1,2),np.float32)
    z=np.zeros((a.steps+1,a.dim),np.float32);active=np.zeros(a.steps+1,bool);visited=[]
    priority=np.full(a.steps+1,-np.inf);images[0]=env.reset();positions[0]=env.position
    z[0]=learner.encode(images[:1])[0];current=0;heading=None;resets=[]
    coverage=SpatialCoverage(a.size);coverage.add(positions[0],0);rows=[];frames=[];stats={};samples=0
    old_order=np.empty(0,int);old_count=0;tree=None;tree_count=0
    def refresh(n):
        nonlocal tree,tree_count
        ids=np.concatenate([np.array(visited,int),np.flatnonzero(active[:n+1])])
        for j in range(0,len(ids),512):z[ids[j:j+512]]=learner.encode(images[ids[j:j+512]])
        tree=cKDTree(z[visited]);tree_count=len(visited)
        pending=np.flatnonzero(active[:n+1]);priority[pending]=tree.query(z[pending])[0]
    with (a.out/'progress.jsonl').open('w') as log:
        for base in range(0,a.steps,5):
            walking=a.policy=='walk' or base<a.warm_steps
            if not walking:
                if base:
                    ids=np.flatnonzero(active[:base+1])
                    current=int(qrng.choice(ids if a.policy=='random_queue' else ids[priority[ids]==priority[ids].max()]))
                    active[current]=False
                visited.append(current)
                ids=np.flatnonzero(active[:base+1]);priority[ids]=np.minimum(priority[ids],np.linalg.norm(z[ids]-z[current],axis=-1))
                controls=sample_controls(crng)
                if a.controls=='cardinal':
                    controls[:4]=env.ACTIONS[:4]*crng.uniform(.5,1,(4,1)).astype(np.float32)
            for j in range(5):
                n=base+j+1
                if walking:
                    if n>1 and rrng.random()<a.reset_probability:
                        current=int(rrng.integers(n));resets.append((n-1,current))
                    retain,angle,length=wrng.random(3)
                    if heading is None or retain>=a.persistence:heading=angle
                    if a.controls=='cardinal':control=env.ACTIONS[min(3,int(heading*4))]*np.float32(.5+.5*length)
                    else:control=np.asarray([np.cos(heading*2*np.pi),np.sin(heading*2*np.pi)],np.float32)*np.float32(.5+.5*length)
                else:control=controls[j]
                env.position=positions[current].copy();images[n],_=env.step(control)
                positions[n]=env.position;parents[n]=current;actions[n]=control
                coverage.add(positions[n],n)
                if walking:current=n
            if a.policy!='walk':
                if not visited:visited.append(0)
                ids=np.arange(base+1,base+6);z[ids]=learner.encode(images[ids]);active[ids]=True
                d=tree.query(z[ids])[0] if tree is not None else np.full(5,np.inf)
                recent=visited[tree_count:]
                if recent:d=np.minimum(d,np.linalg.norm(z[ids,None]-z[recent],axis=-1).min(-1))
                priority[ids]=d
            if n%a.train_every==0:
                if a.policy=='learned':
                    if a.dynamics_switch and n>=a.dynamics_switch:a.dynamics=a.dynamics_after
                    # Background is an observation-only statistic frozen at the first event.
                    if a.background and n==a.train_every:
                        learner.encoder.background.copy_(torch.from_numpy(images[:n+1].astype(np.float32).mean(0,keepdims=True)/255.-.5))
                    # All new edges once, plus a causal fraction of an old-data epoch.
                    if (n-a.train_every)%a.epoch_every==0:
                        old_count=n-a.train_every;old_order=rng.permutation(np.arange(1,old_count+1))
                    interval=((n-a.train_every)%a.epoch_every)//a.train_every
                    intervals=a.epoch_every//a.train_every
                    ids=np.concatenate([old_order[old_count*interval//intervals:old_count*(interval+1)//intervals],np.arange(n-a.train_every+1,n+1)])
                    ids=rng.permutation(ids)
                    for k in range(0,len(ids),a.batch):
                        batch=ids[k:k+a.batch];stats=learner.update(images[:n+1],parents,actions,batch,rng,metrics=False);samples+=len(batch)
                    for _ in range(a.extra_updates):
                        batch=rng.integers(1,n+1,size=a.batch);stats=learner.update(images[:n+1],parents,actions,batch,rng,metrics=False);samples+=len(batch)
                if stats:stats={k:float(v) for k,v in stats.items()}
                # Updating weights ALWAYS invalidates all cached queue embeddings.
                if a.policy!='walk':refresh(n)
            elif n%a.refresh_every==0 and a.policy!='walk':refresh(n)
            if n%500==0:
                row=dict(interactions=n,updates=learner.updates,samples=samples,seconds=time.monotonic()-start,**coverage.metrics(),**stats)
                rows.append(row);log.write(json.dumps(row)+'\n');log.flush()
                frames.append(coverage.masks[4].copy())
            if n%5000==0:print(json.dumps(row),flush=True)
            if a.checkpoint_every and n%a.checkpoint_every==0 and a.policy=='learned':
                torch.save(dict(interactions=n,updates=learner.updates,encoder={k:v.cpu() for k,v in learner.encoder.state_dict().items()},heads={k:v.cpu() for k,v in learner.heads.state_dict().items()}),a.out/f'checkpoint-{n}.pt')
    np.savez_compressed(a.out/'trajectory.npz',positions=positions,parents=parents,actions=actions,resets=np.array(resets).reshape(-1,2),frames=frames,walls=env.walls)
    # Keep original noisy observations: no re-sampled/clean targets in training.
    np.save(a.out/'images.npy',images)
    torch.save(dict(encoder={k:v.cpu() for k,v in learner.encoder.state_dict().items()},heads={k:v.cpu() for k,v in learner.heads.state_dict().items()},inverse={k:v.cpu() for k,v in learner.inverse.state_dict().items()} if learner.inverse else None),a.out/'checkpoint.pt')
    result=dict(**row,milestones=coverage.milestones,resets=len(resets),parameters=sum(p.numel() for p in learner.encoder.parameters()),
        auc=float(np.trapezoid([1/(a.size*4)**2]+[r['coverage_k4'] for r in rows],[0]+[r['interactions'] for r in rows])/a.steps))
    (a.out/'result.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__':main()
