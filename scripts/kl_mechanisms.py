"""Fixed interventions and shared-archive diagnostics for KL mechanism study."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree
from map_learning.maze import Maze,reachable_graph
from audit_frontier_efficiency import reconstruct,encode,priority_stats
from audit_queue_density import counts,live_sequence

ARMS={
    'mean_zero':dict(beta=0.,dynamics_input='mean'),
    'noise_zero':dict(beta=0.,dynamics_input='scheduled'),
    'mean_kl':dict(beta=1e-5,dynamics_input='mean'),
    'noise_kl':dict(beta=1e-5,dynamics_input='scheduled'),
    'adaptive_zero':dict(beta=0.,dynamics_input='posterior'),
    'adaptive_kl':dict(beta=1e-5,dynamics_input='posterior'),
    'prior_lr3':dict(beta=1e-5,dynamics_input='posterior',prior_lr_multiplier=3.),
    'prior_lr10':dict(beta=1e-5,dynamics_input='posterior',prior_lr_multiplier=10.),
    'prior_steps5':dict(beta=1e-5,dynamics_input='posterior',prior_steps=5),
}
LABELS={'mean_zero':'Mean / no KL','noise_zero':'Fixed noise / no KL','mean_kl':'Mean / KL',
        'noise_kl':'Fixed noise / KL','adaptive_zero':'Learned noise / no KL','adaptive_kl':'Current winner',
        'prior_lr3':'Prior LR ×3','prior_lr10':'Prior LR ×10','prior_steps5':'5 prior steps'}
COLORS=dict(zip(ARMS,['#777777','#e69f00','#009e73','#cc79a7','#999933','#0072b2','#56b4e9','#d55e00','#882255']))


def spec(arm,schedule):
    d=dict(ARMS[arm])
    if d['dynamics_input']=='scheduled':d['noise_schedule']=schedule
    return d


def data_source(family,source_seed):
    if family=='winner':return Path(f'runs/moment-prior-confirm-controls-20261110/learned-rep0.03-beta1e-05-seed{source_seed}')
    if family=='zero':return Path(f'runs/moment-prior-confirm-controls-20261110/learned-rep0.01-beta0-seed{source_seed}')
    raise ValueError(family)


def digest_model(t):
    return hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in list(t.model.parameters())+list(t.dynamics.parameters()))).hexdigest()


def make_noise_schedule():
    # An exogenous scalar RMS schedule, fixed from PREVIOUS screening seeds.
    # No observations, outcomes, or parameters from the new runs are used.
    logs=[json.loads(Path(f'runs/kl-repulsion-intermediate-20261110/rep0.03-beta1e-05-seed{s}/result.json').read_text())['blocks'] for s in range(3)]
    std=[.01]+[float(np.sqrt(np.mean([b[i]['bottleneck']['sample_noise_to_mean_variance']*b[i]['bottleneck']['mean_latent_variance'] for b in logs]))) for i in range(20)]
    return {'updates':[0]+[b['updates'] for b in logs[0]],'std':std}


def queue_audit(t,data,prefix,source_seed,live=False):
    obs,pos,vo,vp=reconstruct(data,prefix)
    env=Maze(t.config.maze);points,adj,index=reachable_graph(env)
    ids=lambda ps:np.array([index[tuple(np.round(p,5))] for p in ps])
    fid=ids(pos);seen=np.zeros(len(points),bool);seen[ids(data['positions'][:1])]=True;seen[ids(data['next_positions'][:prefix])]=True
    expanded=np.zeros(len(points),bool);expanded[ids(vp)]=True
    before=t.sample_rng.get_state().clone();z=encode(t,obs);v=encode(t,vo)
    assert torch.equal(before,t.sample_rng.get_state())
    scores,near=cKDTree(v).query(z,eps=0)
    order=np.argsort(-scores,kind='stable');productive=(~seen[adj]).any(1)
    out={'prefix':prefix,'seen_states':int(seen.sum()),'queue_entries':len(obs),'stats':priority_stats(scores,fid,productive,expanded),
         'top_k':[counts(order[:k],fid,expanded,seen,adj) for k in [10,100,128,1000]],
         'whole_queue':counts(order,fid,expanded,seen,adj),
         'nearest_same_position_fraction':float((ids(vp)[near]==fid).mean()),'live':[]}
    if live and (~seen).any():
        for replicate in range(3):
            rows=live_sequence(t,obs,pos,vo,scores,z,v,seen.copy(),expanded,points,index,128,source_seed+991000+replicate*100)
            out['live'].append({'replicate':replicate,'rows':rows})
    return out
