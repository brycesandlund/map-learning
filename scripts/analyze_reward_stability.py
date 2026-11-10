"""Temporal reward-map changes, separating global scale from spatial shape."""
import numpy as np


def rank(x):
    _,inverse,counts=np.unique(x,return_inverse=True,return_counts=True)
    return (np.cumsum(counts)-(counts+1)/2)[inverse]


def analyze(path):
    with np.load(path/'reward_snapshots.npz') as s:
        mask=s['step']>=5000
        reward=s['reward'][mask].reshape(mask.sum(),-1).astype(float)
        scale=s['scale'][mask]
    mean=reward.mean(1).clip(1e-20);shape=reward/mean[:,None]
    correlations=[];jaccard=[]
    for a,b in zip(reward[:-1],reward[1:]):
        correlations.append(float(np.corrcoef(rank(a),rank(b))[0,1]))
        aa=a>=np.quantile(a,.9);bb=b>=np.quantile(b,.9)
        jaccard.append(float((aa&bb).sum()/(aa|bb).sum()))
    return {'frames':len(mean),'normalized_scale_log_change':float(np.abs(np.diff(np.log(mean))).mean()),
        'raw_scale_log_change':float(np.abs(np.diff(np.log(mean*scale))).mean()),
        'spatial_relative_l1_change':float(np.abs(np.diff(shape,axis=0)).mean()),
        'spatial_rank_correlation':float(np.mean(correlations)),
        'top_decile_jaccard':float(np.mean(jaccard))}
