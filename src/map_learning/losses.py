import torch
from torch.nn import functional as F


def qrl_loss(random_distances, edge_distances, multiplier, epsilon, push_scale=20.0, costs=1.0):
    """QRL primal objective, with a separate projected-ascent multiplier update.

    softplus(-d/s)*s has decreasing push magnitude for already-large distances.
    With s=0 use the paper's original linear objective -E[d]. A positive epsilon
    is an average squared-violation tolerance, not an exact per-edge bound.
    """
    push = (push_scale * F.softplus(-random_distances/push_scale)).mean() if push_scale else -random_distances.mean()
    violation = F.relu(edge_distances-costs).square().mean()
    constraint = violation - epsilon**2
    return push + multiplier.detach()*constraint, {"push": push, "violation": violation, "constraint": constraint}


def time_myopic_loss(predicted_normalized, gaps, horizon):
    targets = (gaps/horizon).clamp(0, 1)
    return F.mse_loss(predicted_normalized, targets)


@torch.no_grad()
def dual_ascent(multiplier, constraint, lr):
    multiplier.add_(lr*constraint).clamp_(min=0)
