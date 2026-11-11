"""Opt-in L1, fixed-noise channels, and ancestry rollout training.

No observation equivalence labels or physical positions are used here.
"""
from dataclasses import dataclass, asdict
import math
import numpy as np
import torch
from .training import Trainer
from .distance_learning import Ancestry
from .probabilistic import repulsion_loss


@dataclass
class CompressionSpec:
    objective: str = 'dynamics'
    adjacency_weight: float = .3
    latent_l1: float = 0.
    l1_after: int = 0
    l1_ramp: int = 0
    noise_std: float = 0.
    noise_after: int = 0
    noise_ramp: int = 0
    noise_repulsion: bool = True
    bound: float = 0.
    multistep_weight: float = 0.
    horizon: int = 1
    trajectory_batch: int = 32

    def __post_init__(self):
        assert self.objective in {'dynamics','adjacency'}
        for name in ['adjacency_weight','latent_l1','noise_std','bound','multistep_weight']:
            value=getattr(self,name)
            if not math.isfinite(value) or value<0:raise ValueError(name)
        if self.noise_after<0 or self.noise_ramp<0:raise ValueError('Invalid noise schedule')
        if self.l1_after<0 or self.l1_ramp<0:raise ValueError('Invalid L1 schedule')
        if self.horizon<1 or self.trajectory_batch<1:raise ValueError('Invalid trajectory dimensions')
        if self.noise_std and not self.bound:raise ValueError('Noise channel requires an explicit encoder bound')


def ancestry_sequences(ancestry, indices, horizon, rng):
    """Return actual source IDs and chronological child IDs, with valid masks.

    Replay row i is edge parent[i+1] -> i+1. Array order alone never implies adjacency.
    """
    end=np.asarray(indices,dtype=np.int64)+1
    if np.any(end>=ancestry.size):raise IndexError('Missing ancestry')
    lengths=rng.integers(1,np.minimum(ancestry.depth[end],horizon)+1)
    back=np.zeros((len(end),int(lengths.max())),dtype=np.int64)
    cursor=end.copy()
    for j in range(back.shape[1]):
        active=lengths>j;back[active,j]=cursor[active];cursor[active]=ancestry.parents[cursor[active]]
    valid=np.arange(back.shape[1])[None,:]<lengths[:,None]
    children=back[np.arange(len(end))[:,None],np.maximum(lengths[:,None]-1-np.arange(back.shape[1]),0)]
    return cursor,children,valid,lengths


class CompressionTrainer(Trainer):
    def __init__(self, config, observation_shape, spec):
        super().__init__(config,observation_shape)
        self.compression_spec=spec if isinstance(spec,CompressionSpec) else CompressionSpec(**spec)
        if self.compression_spec.bound and (config.model.latent_activation!='tanh' or config.model.latent_bound!=self.compression_spec.bound):
            raise ValueError('Configure the encoder bound before constructing the trainer')
        if config.train.ensemble_batches!='shared':raise ValueError('Study uses matched shared batches')
        self.ancestry=Ancestry(config.collect_steps,observation_shape)
        self.channel_rng=np.random.default_rng(config.seed+101003)
        self.path_rng=np.random.default_rng(config.seed+101009)
        self.current_noise_std=self.compression_spec.noise_std
        self.path_presentations=0
        self.path_edges=0

    def corrupt(self,z):
        if not self.current_noise_std:return z
        noise=self.channel_rng.normal(0,self.current_noise_std,z.shape).astype(np.float32)
        return z+self.tensor(noise)

    def update(self,replay):
        indices=[self.rng.integers(len(replay),size=self.config.train.batch_size) for _ in self.dynamics.heads]
        return self.update_batch(replay,indices)

    def rollout_loss(self,replay,indices):
        s=self.compression_spec
        chosen=self.path_rng.choice(indices,size=min(s.trajectory_batch,len(indices)),replace=False)
        sources,children,valid,lengths=ancestry_sequences(self.ancestry,chosen,s.horizon,self.path_rng)
        start=self.corrupt(self.model.encode(self.tensor(self.ancestry.observations[sources])))
        target=self.model.encode(self.tensor(self.ancestry.observations[chosen+1]))
        controls=self.tensor(replay.arrays()['actions'][children-1])
        preds=[start for _ in self.dynamics.heads]
        for j in range(children.shape[1]):
            mask=torch.as_tensor(valid[:,j,None],device=self.device)
            preds=[torch.where(mask,self.dynamics.predict_head(h,z,controls[:,j]),z) for h,z in enumerate(preds)]
        self.path_presentations+=len(chosen);self.path_edges+=int(lengths.sum())
        return (torch.stack(preds)-target).square().mean(),float(lengths.mean())

    def update_batch(self,replay,head_indices):
        c,s=self.config.train,self.compression_spec
        noise_factor=(0. if len(replay)<=s.noise_after else min(1.,(len(replay)-s.noise_after)/max(1,s.noise_ramp)))
        self.current_noise_std=s.noise_std*noise_factor
        indices=head_indices[0]
        obs,actions,nxt=replay.transitions_at(indices)
        if s.objective=='adjacency':
            z,target=self.model.encode(self.tensor(np.concatenate([obs,nxt]))).chunk(2)
        else:
            z=self.model.encode(self.tensor(obs));target=self.model.encode(self.tensor(nxt))
        noisy=self.corrupt(z)
        if s.objective=='dynamics':
            prediction=(self.dynamics(noisy,torch.as_tensor(actions,device=self.device))-target).square().mean()
            task=c.dynamics_weight*prediction
        else:
            task=s.adjacency_weight*torch.linalg.vector_norm(noisy-target,dim=-1).square().mean()
            prediction=(self.dynamics(noisy.detach(),torch.as_tensor(actions,device=self.device))-target.detach()).square().mean()
        states=self.model.encode(self.tensor(replay.sample_states(2*c.batch_size,self.rng)))
        sent=self.corrupt(states) if s.noise_repulsion else states
        repulsion=repulsion_loss(*sent.chunk(2),c.repulsion_margin)
        norm=states.square().sum(-1).mean();l1=states.abs().sum(-1).mean()
        l1_factor=(0. if len(replay)<=s.l1_after else min(1.,(len(replay)-s.l1_after)/max(1,s.l1_ramp)))
        total=task+c.repulsion_weight*repulsion+c.latent_l2*norm+(s.latent_l1*l1_factor)*l1
        multi=total.new_zeros(());lag=0.
        if s.multistep_weight:
            multi,lag=self.rollout_loss(replay,indices);total=total+s.multistep_weight*multi
        optimized=total+(prediction if s.objective=='adjacency' else 0.)
        if not torch.isfinite(optimized):raise FloatingPointError('Nonfinite compression loss')
        self.map_optimizer.zero_grad();self.dynamics_optimizer.zero_grad();optimized.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(),10.);torch.nn.utils.clip_grad_norm_(self.dynamics.parameters(),10.)
        self.map_optimizer.step();self.dynamics_optimizer.step();self.updates+=1
        left,right=sent.chunk(2)
        return {'update':self.updates,'joint_loss':float(total.detach()),'prediction_loss':float(prediction.detach()),
                'effective_noise_std':self.current_noise_std,'base_task_loss':float(task.detach()),'multistep_loss':float(multi.detach()),'mean_rollout_lag':lag,
                'path_presentations':self.path_presentations,'path_edges':self.path_edges,
                'repulsion':float(repulsion.detach()),'latent_squared_norm':float(norm.detach()),'latent_l1':float(l1.detach()),'effective_l1_weight':s.latent_l1*l1_factor,
                'repulsion_active_fraction':float(((left-right).square().sum(-1)<c.repulsion_margin**2).float().mean().detach()),
                'random_pair_distance_mean':float((left-right).norm(dim=-1).mean().detach())}

    def save(self,path):
        super().save(path)
        data=torch.load(path,map_location=self.device,weights_only=True)
        data.update(compression_spec=asdict(self.compression_spec),ancestry=self.ancestry.state_dict(),
                    channel_rng_state=self.channel_rng.bit_generator.state,path_rng_state=self.path_rng.bit_generator.state,
                    path_presentations=self.path_presentations,path_edges=self.path_edges)
        torch.save(data,path)

    @classmethod
    def from_checkpoint(cls,data,device):
        from .config import config_from_dict
        cfg=config_from_dict(data['config']);cfg.device=device
        t=cls(cfg,tuple(data['observation_shape']),data['compression_spec'])
        for name in ['model','dynamics','map_optimizer','dynamics_optimizer']:getattr(t,name).load_state_dict(data[name])
        t.updates=data['updates'];t.multiplier.copy_(data['multiplier']);t.rng.bit_generator.state=data['rng_state']
        t.channel_rng.bit_generator.state=data['channel_rng_state'];t.path_rng.bit_generator.state=data['path_rng_state']
        t.path_presentations=data['path_presentations'];t.path_edges=data['path_edges'];t.ancestry.restore(data['ancestry'])
        torch.set_rng_state(data['torch_rng_state'].cpu())
        return t
