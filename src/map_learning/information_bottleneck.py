"""Experimental stochastic encoder and learned-prior information bottleneck.

encode() always returns the deterministic mean. Posterior samples are used only
by this trainer. KL is in nats per observation (sum over latent dimensions).
"""
import math
import time
from dataclasses import asdict
import numpy as np
import torch
from torch import nn
from .training import Trainer
from .probabilistic import mixture_mean, repulsion_loss


class LearnedPrior(nn.Module):
    def __init__(self, dim, components):
        super().__init__()
        self.components=components
        self.means=nn.Parameter(torch.zeros(components,dim))
        if components>1:
            with torch.no_grad():self.means.normal_(0,.03)
        self.log_std=nn.Parameter(torch.full((components,dim),math.log(.05)))
        self.logits=nn.Parameter(torch.zeros(components))
        self.bounded_scales=True

    def log_scales(self):
        return self.log_std.clamp(-12,2) if self.bounded_scales else self.log_std

    def log_prob(self,z):
        logstd=self.log_scales()
        lp=-.5*((z.unsqueeze(-2)-self.means)*(-logstd).exp()).square().sum(-1)-logstd.sum(-1)-.5*z.shape[-1]*math.log(2*math.pi)
        return torch.logsumexp(lp+self.logits.log_softmax(-1),-1)

    def kl(self,mu,logstd,generator,samples=4):
        if self.components==1:
            plog=self.log_scales()[0]
            return (plog-logstd + .5*((2*(logstd-plog)).exp()+(mu-self.means[0]).square()*(-2*plog).exp()-1)).sum(-1).mean()
        eps=torch.randn((samples,*mu.shape),generator=generator,device=mu.device)
        z=mu.unsqueeze(0)+logstd.exp().unsqueeze(0)*eps
        expected_logq=-.5*mu.shape[-1]*(1+math.log(2*math.pi))-logstd.sum(-1)
        return (expected_logq-self.log_prob(z).mean(0)).mean()


class BottleneckTrainer(Trainer):
    def __init__(self,config,observation_shape,beta=0.,prior_components=1,initial_std=.01,prior_lr_multiplier=1.,kl_warmup_updates=0,prior_mode='learned',dynamics_input='posterior',noise_schedule=None,prior_steps=1,latent_l2=0.):
        if config.train.loss != 'probabilistic':raise ValueError('Bottleneck experiments require the legacy probabilistic configuration')
        if len(observation_shape)!=1 or config.model.latent_activation!='identity':raise ValueError('Bottleneck experiment requires vector observations and identity mean encoder')
        if config.train.prediction_objective!='mse' or config.train.ensemble_batches!='shared':raise ValueError('Use mean-MSE and shared ensemble batches')
        if beta<0 or prior_components<1 or not 0<initial_std<1 or prior_lr_multiplier<=0:raise ValueError('Invalid bottleneck settings')
        if kl_warmup_updates<0 or (kl_warmup_updates and prior_components!=1):raise ValueError('Warm-start intervention supports one Gaussian prior')
        if prior_mode not in {'learned','replay_moments','batch_moments'}:raise ValueError('Unknown prior mode')
        if prior_mode!='learned' and prior_components!=1:raise ValueError('Moment priors require one Gaussian')
        if dynamics_input not in {'posterior','mean','scheduled'}:raise ValueError('Unknown dynamics input')
        if prior_steps<1 or int(prior_steps)!=prior_steps or (prior_steps!=1 and prior_mode!='learned'):raise ValueError('Prior steps require a learned prior and positive integer count')
        if prior_steps!=1 and prior_components!=1:raise ValueError('Extra prior steps currently support one Gaussian')
        if not np.isfinite(latent_l2) or latent_l2<0:raise ValueError('Latent L2 must be finite and nonnegative')
        if dynamics_input=='scheduled':
            if not noise_schedule or len(noise_schedule['updates'])!=len(noise_schedule['std']) or len(noise_schedule['std'])<2:raise ValueError('Scheduled noise needs update/std arrays')
            if not np.all(np.diff(noise_schedule['updates'])>0) or not np.all(np.isfinite(noise_schedule['std'])) or min(noise_schedule['std'])<0:raise ValueError('Invalid noise schedule')
        super().__init__(config,observation_shape)
        self.spec={'beta':beta,'prior_components':prior_components,'initial_std':initial_std,'prior_lr_multiplier':prior_lr_multiplier,'kl_warmup_updates':kl_warmup_updates,'prior_mode':prior_mode,'dynamics_input':dynamics_input,'noise_schedule':noise_schedule,'prior_steps':prior_steps,'latent_l2':latent_l2}
        # Preserve original mean encoder/dynamics initialization and global RNG.
        with torch.random.fork_rng():
            torch.manual_seed(config.seed+81000)
            self.model.posterior_log_std=nn.Linear(config.model.hidden_dim,config.model.latent_dim).to(self.device)
            nn.init.zeros_(self.model.posterior_log_std.weight)
            nn.init.constant_(self.model.posterior_log_std.bias,math.log(initial_std))
            self.prior=LearnedPrior(config.model.latent_dim,prior_components).to(self.device)
        if prior_mode!='learned':
            self.prior.requires_grad_(False)
            self.prior.bounded_scales=False  # exact moments, not a clipped fit
        self.prior_fit_stats={'calls':0,'observations':0,'seconds':0.}
        self.prior_updates=0
        self.map_optimizer=torch.optim.Adam(self.model.parameters(),lr=config.train.lr*config.train.encoder_lr_multiplier)
        self.prior_optimizer=torch.optim.Adam(self.prior.parameters(),lr=config.train.lr*prior_lr_multiplier)
        self.sample_rng=torch.Generator(device=self.device).manual_seed(config.seed+82000)
        self.kl_rng=torch.Generator(device=self.device).manual_seed(config.seed+83000)

    def posterior(self,obs):
        hidden=self.model.encoder[:-1](obs)
        mu=self.model.encoder[-1](hidden)
        logstd=self.model.posterior_log_std(hidden).clamp(-12,2)
        return mu,logstd

    def input_std(self,logstd):
        mode=self.spec['dynamics_input']
        if mode=='posterior':return logstd.exp()
        if mode=='mean':return torch.zeros_like(logstd)
        s=self.spec['noise_schedule']
        return torch.full_like(logstd,float(np.interp(self.updates,s['updates'],s['std'])))

    @torch.no_grad()
    def replay_posterior_moments(self,replay,chunk_size=4096):
        """Current encoder over ALL source/next entries, with replay multiplicity.

        Float64 streaming central moments include E posterior variance. No RNG,
        sampling, gradients, privileged state labels, or persistent data changes.
        """
        if not len(replay):raise ValueError('Cannot fit moments to empty replay')
        dim=self.config.model.latent_dim
        mean=torch.zeros(dim,dtype=torch.float64,device=self.device)
        m2=torch.zeros_like(mean);qsum=torch.zeros_like(mean);lssum=torch.zeros_like(mean);count=0
        data=replay.arrays()
        for name in ['observations','next_observations']:
            for start in range(0,len(replay),chunk_size):
                mu,ls=self.posterior(self.tensor(data[name][start:start+chunk_size]))
                mu=mu.double();ls=ls.double();n=len(mu);bm=mu.mean(0);delta=bm-mean
                m2+=(mu-bm).square().sum(0)+delta.square()*(count*n/(count+n))
                mean+=delta*(n/(count+n));count+=n
                qsum+=(2*ls).exp().sum(0);lssum+=ls.sum(0)
        variance=(m2+qsum)/count
        logstd=.5*variance.log()
        return {'mean':mean,'logstd':logstd,'mean_variance':m2/count,'posterior_variance':qsum/count,
                'mean_logstd':lssum/count,'matched_kl':(logstd-lssum/count).sum(),'count':count}

    @torch.no_grad()
    def fit_prior(self,replay):
        if self.spec['prior_mode']!='replay_moments':return None
        start=time.monotonic();m=self.replay_posterior_moments(replay)
        self.prior.means[0].copy_(m['mean']);self.prior.log_std[0].copy_(m['logstd'])
        self.prior_fit_stats['calls']+=1;self.prior_fit_stats['observations']+=m['count']
        self.prior_fit_stats['seconds']+=time.monotonic()-start
        return float(m['matched_kl'])

    @torch.no_grad()
    def fit_batch_prior(self,mu,ls):
        self.prior.means[0].copy_(mu.mean(0))
        self.prior.log_std[0].copy_(.5*(mu.var(0,unbiased=False)+(2*ls).exp().mean(0)).log())

    def update_probabilistic(self,replay,head_indices=None):
        c=self.config.train
        if head_indices is None:
            head_indices=[self.rng.integers(len(replay),size=c.batch_size) for _ in self.dynamics.heads]
        obs,actions,nxt=replay.transitions_at(head_indices[0])
        mu,logstd=self.posterior(self.tensor(obs))
        target=self.model.encode(self.tensor(nxt))  # joint target gradients, no target sampling
        eps=torch.randn(mu.shape,generator=self.sample_rng,device=self.device)
        input_std=self.input_std(logstd)
        z=mu+input_std*eps  # same input samples across ensemble heads
        pred=self.dynamics(z,torch.as_tensor(actions,device=self.device))
        prediction=(pred-target).square().mean()
        states=self.tensor(replay.sample_states(2*c.batch_size,self.rng))
        means,ls=self.posterior(states)
        left,right=means.chunk(2)
        repulsion=repulsion_loss(left,right,c.repulsion_margin)
        if self.spec['prior_mode']=='replay_moments' and not self.prior_fit_stats['calls']:
            raise RuntimeError('Call fit_prior(replay) before a replay-moment training event')
        if self.spec['prior_mode']=='batch_moments':self.fit_batch_prior(means,ls)
        warmup=self.spec['kl_warmup_updates']
        active_beta=self.spec['beta'] if self.updates>=warmup else 0.
        if warmup and self.updates==warmup and self.spec['prior_mode']=='learned':
            # Causal moment fit to current posterior aggregate, no privileged labels.
            with torch.no_grad():
                self.prior.means[0].copy_(means.mean(0))
                variance=means.var(0,unbiased=False)+(2*ls).exp().mean(0)
                self.prior.log_std[0].copy_(.5*variance.clamp_min(1e-12).log())
        kl=self.prior.kl(means,ls,self.kl_rng)
        total=prediction+c.repulsion_weight*repulsion+active_beta*kl
        latent_norm=means.square().sum(-1).mean()
        if self.spec['latent_l2']:total=total+self.spec['latent_l2']*latent_norm
        if not torch.isfinite(total):raise FloatingPointError('Nonfinite bottleneck loss')
        self.map_optimizer.zero_grad();self.dynamics_optimizer.zero_grad();self.prior_optimizer.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(),10.)
        torch.nn.utils.clip_grad_norm_(self.dynamics.parameters(),10.)
        torch.nn.utils.clip_grad_norm_(self.prior.parameters(),10.)
        self.map_optimizer.step();self.dynamics_optimizer.step()
        if active_beta>0 and self.spec['prior_mode']=='learned':
            self.prior_optimizer.step();self.prior_updates+=1
            # Refit the same detached, pre-update posterior batch. No encoder or
            # dynamics gradients, additional replay samples, or RNG consumption.
            for _ in range(self.spec['prior_steps']-1):
                self.prior_optimizer.zero_grad()
                prior_loss=active_beta*self.prior.kl(means.detach(),ls.detach(),self.kl_rng)
                prior_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.prior.parameters(),10.)
                self.prior_optimizer.step();self.prior_updates+=1
        self.updates+=1
        metrics={'update':self.updates,'joint_loss':float(total.detach()),'prediction_loss':float(prediction.detach()),
            'repulsion':float(repulsion.detach()),'kl_nats':float(kl.detach()),'weighted_kl':float((active_beta*kl).detach()),'active_beta':active_beta,
            'posterior_std':float(ls.exp().mean().detach()),'mean_latent_std':float(means.std(0).mean().detach()),
            'dynamics_input_std':float(input_std.mean().detach()),'prior_updates':self.prior_updates,
            'latent_squared_norm':float(latent_norm.detach()),'weighted_latent_l2':float(self.spec['latent_l2']*latent_norm.detach())}
        if self.spec['prior_components']==1:
            with torch.no_grad():
                matched=(.5*(means.var(0,unbiased=False)+(2*ls).exp().mean(0)).log()-ls.mean(0)).sum()
                metrics.update(batch_matched_kl=float(matched),prior_mismatch_batch=float(kl-matched))
        return metrics

    def bottleneck_diagnostics(self,replay):
        rng=np.random.default_rng(self.config.seed+84000)
        with torch.no_grad():
            states=self.tensor(replay.sample_states(2048,rng))
            mu,logstd=self.posterior(states)
            gen=torch.Generator(device=self.device).manual_seed(self.config.seed+85000)
            kl=self.prior.kl(mu,logstd,gen,samples=32)
            mean_var=mu.var(0,unbiased=False)
            std=logstd.exp()
            # A second noisy dynamics probe estimates training-vs-mean mismatch.
            obs,a,nxt=replay.transitions_at(rng.integers(len(replay),size=1024))
            m,ls=self.posterior(self.tensor(obs));target=self.model.encode(self.tensor(nxt))
            noisy=m+self.input_std(ls)*torch.randn(m.shape,generator=gen,device=self.device)
            pred=self.dynamics(noisy,torch.as_tensor(a,device=self.device)).mean(0)
            active_beta=self.spec['beta'] if self.updates>self.spec['kl_warmup_updates'] else 0.
            return {'kl_nats':float(kl),'active_beta':active_beta,'weighted_kl':float(active_beta*kl),'posterior_std_mean':float(std.mean()),
                'posterior_std_min':float(std.min()),'posterior_std_max':float(std.max()),
                'posterior_std_per_dim':std.mean(0).tolist(),'mean_variance_per_dim':mean_var.tolist(),
                'mean_latent_variance':float(mean_var.mean()),'sample_noise_to_mean_variance':float(std.square().mean()/mean_var.mean().clamp_min(1e-12)),
                'sampled_input_normalized_mse':float((pred-target).square().mean()/mean_var.mean().clamp_min(1e-12)),
                'prior_component_weights':self.prior.logits.softmax(-1).tolist(),
                'prior_std_mean':float(self.prior.log_scales().exp().mean()),'prior_fit_stats':dict(self.prior_fit_stats),
                'dynamics_input_std_mean':float(self.input_std(logstd).mean()),'prior_updates':self.prior_updates}

    def save(self,path):
        super().save(path)
        data=torch.load(path,weights_only=True,map_location=self.device)
        data.update(bottleneck_spec=self.spec,prior=self.prior.state_dict(),prior_optimizer=self.prior_optimizer.state_dict(),
                    sample_rng=self.sample_rng.get_state(),kl_rng=self.kl_rng.get_state(),prior_fit_stats=self.prior_fit_stats,prior_updates=self.prior_updates)
        torch.save(data,path)

    @classmethod
    def from_checkpoint(cls,data,device='cpu'):
        from .config import config_from_dict
        cfg=config_from_dict(data['config']);cfg.device=device
        t=cls(cfg,tuple(data['observation_shape']),**data['bottleneck_spec'])
        t.model.load_state_dict(data['model']);t.dynamics.load_state_dict(data['dynamics'])
        t.map_optimizer.load_state_dict(data['map_optimizer']);t.dynamics_optimizer.load_state_dict(data['dynamics_optimizer'])
        t.prior.load_state_dict(data['prior']);t.prior_optimizer.load_state_dict(data['prior_optimizer'])
        t.prior_fit_stats=data.get('prior_fit_stats',{'calls':0,'observations':0,'seconds':0.})
        old_prior_updates=max(0,data['updates']-t.spec['kl_warmup_updates']) if t.spec['beta']>0 and t.spec['prior_mode']=='learned' else 0
        t.prior_updates=data.get('prior_updates',old_prior_updates)
        t.multiplier.copy_(data['multiplier']);t.updates=data['updates'];t.rng.bit_generator.state=data['rng_state']
        t.sample_rng.set_state(data['sample_rng'].cpu());t.kl_rng.set_state(data['kl_rng'].cpu())
        torch.set_rng_state(data['torch_rng_state'].cpu())
        return t
