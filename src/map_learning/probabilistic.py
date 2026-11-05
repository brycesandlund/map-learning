"""Conditional mixture-density ensemble and distribution-level uncertainty.

Shapes: logits [..., K], means/variances [..., K, D]. Ensemble parameters
have leading axis M. Component labels need not agree between ensemble heads.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .models import mlp


def mixture_log_prob(parameters, value):
    logits, means, variances = parameters
    component = -.5 * (((value.unsqueeze(-2)-means).square()/variances)
                        + variances.log() + math.log(2*math.pi)).sum(-1)
    return torch.logsumexp(logits.log_softmax(-1) + component, dim=-1)


def mixture_mean(parameters):
    logits, means, _ = parameters
    return (logits.softmax(-1).unsqueeze(-1) * means).sum(-2)


def mixture_variance(parameters):
    logits, means, variances = parameters
    weights = logits.softmax(-1).unsqueeze(-1)
    mean = mixture_mean(parameters)
    return (weights * (variances + (means-mean.unsqueeze(-2)).square())).sum(-2)


def mixture_sample(parameters, generator=None):
    logits, means, variances = parameters
    shape, k, d = means.shape[:-2], means.shape[-2], means.shape[-1]
    chosen = torch.multinomial(logits.softmax(-1).reshape(-1, k), 1,
                               generator=generator).reshape(*shape, 1, 1).expand(*shape, 1, d)
    mu = means.gather(-2, chosen).squeeze(-2)
    var = variances.gather(-2, chosen).squeeze(-2)
    noise = torch.randn(mu.shape, dtype=mu.dtype, device=mu.device, generator=generator)
    return mu + var.sqrt()*noise


def ensemble_sample(parameters, generator=None):
    """One draw per batch item from the uniform ensemble predictive mixture."""
    logits, means, variances = parameters
    members, batch = logits.shape[:2]
    indices = torch.randint(members, (batch,), device=logits.device, generator=generator)
    rows = torch.arange(batch, device=logits.device)
    return mixture_sample((logits[indices, rows], means[indices, rows], variances[indices, rows]), generator)


def ensemble_js(parameters, samples=8, generator=None):
    """Stratified MC Jensen-Shannon divergence, in nats, one score per batch item.

    E_m E_{x~p_m}[log p_m(x) - log mean_j p_j(x)]. Broad but identical
    distributions have zero disagreement. This is approximate epistemic uncertainty,
    not a calibrated Bayesian posterior. Small MC estimates can be negative; clamp
    only the final estimate to [0, log M], never individual sample contributions.
    """
    logits, means, variances = parameters  # M, B, K[, D]
    m, b, k = logits.shape
    d = means.shape[-1]
    expanded = (logits.unsqueeze(0).expand(samples, m, b, k),
                means.unsqueeze(0).expand(samples, m, b, k, d),
                variances.unsqueeze(0).expand(samples, m, b, k, d))
    draws = mixture_sample(expanded, generator)  # S, source M, B, D
    # Evaluate each draw under every member; axis order: target M, S, source M, B.
    all_logp = mixture_log_prob((logits[:, None, None], means[:, None, None],
                                variances[:, None, None]), draws.unsqueeze(0))
    own = torch.stack([all_logp[i, :, i] for i in range(m)])  # M, S, B
    pooled = torch.logsumexp(all_logp, 0)-math.log(m)  # S, M, B
    return (own.permute(1, 0, 2)-pooled).mean((0, 1)).clamp(0, math.log(m))


def repulsion_loss(left, right, margin=1.0):
    return (margin**2-(left-right).square().sum(-1)).clamp_min(0).mean()


class MixtureDynamicsEnsemble(nn.Module):
    def __init__(self, config, actions=5):
        super().__init__()
        self.actions, self.components, self.dim = actions, config.mixture_components, config.latent_dim
        self.variance_floor = config.variance_floor
        self.heads = nn.ModuleList([mlp(self.dim+actions, config.hidden_dim,
                                      self.components*(1+2*self.dim))
                                    for _ in range(config.ensemble_size)])
        for head in self.heads:
            nn.init.normal_(head[-1].weight, std=.01)
            with torch.no_grad():
                bias = head[-1].bias.view(self.components, 1+2*self.dim)
                bias.zero_()
                # Break component symmetry without asserting which outcome is likely.
                bias[:, 1:1+self.dim].normal_(0, .03)
                bias[:, 1+self.dim:].fill_(math.log(math.expm1(config.initial_variance-config.variance_floor)))

    def parameters_head(self, index, z, action):
        a = F.one_hot(action.long(), self.actions).to(z.dtype)
        raw = self.heads[index](torch.cat([z, a], -1)).reshape(*z.shape[:-1], self.components, 1+2*self.dim)
        logits = raw[..., 0]
        means = z.unsqueeze(-2) + raw[..., 1:1+self.dim]
        variances = self.variance_floor + F.softplus(raw[..., 1+self.dim:])
        return logits, means, variances

    def distribution(self, z, action):
        heads = [self.parameters_head(i, z, action) for i in range(len(self.heads))]
        return tuple(torch.stack([head[j] for head in heads]) for j in range(3))

    def forward(self, z, action):
        return mixture_mean(self.distribution(z, action))
