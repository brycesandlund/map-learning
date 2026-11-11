import torch
from torch import nn
from torch.nn import functional as F


def mlp(input_dim, hidden_dim, output_dim, hidden_layers=2):
    if hidden_layers < 1:
        raise ValueError("At least one hidden layer is required")
    layers = []
    for index in range(hidden_layers):
        layers.extend([nn.Linear(input_dim if index == 0 else hidden_dim, hidden_dim), nn.SiLU()])
    return nn.Sequential(*layers, nn.Linear(hidden_dim, output_dim))


class IntervalQuasimetric(nn.Module):
    """IQE-maxmean: length of a union of directed intervals per component.

    Each coordinate contributes [x_i, y_i] if x_i < y_i, else an empty interval.
    Union lengths satisfy triangle inequality; max/mean and their convex mixture
    preserve it. This implementation uses sorted starts and cumulative end maxima.
    Reference: https://arxiv.org/abs/2211.15120 (independent implementation).
    """
    def __init__(self, latent_dim, component_dim=8):
        super().__init__()
        if latent_dim % component_dim:
            raise ValueError("component_dim must divide latent_dim")
        self.components = latent_dim // component_dim
        self.component_dim = component_dim
        self.mix_logit = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, y):
        x, y = torch.broadcast_tensors(x, y)
        shape = (*x.shape[:-1], self.components, self.component_dim)
        starts = x.reshape(shape)
        ends = torch.maximum(x, y).reshape(shape)
        starts, order = starts.sort(dim=-1)
        ends = ends.gather(-1, order)
        rightmost = ends.cummax(dim=-1).values
        previous = torch.cat([starts[..., :1], rightmost[..., :-1]], dim=-1)
        lengths = (ends - torch.maximum(starts, previous)).clamp_min(0).sum(dim=-1)
        alpha = self.mix_logit.sigmoid()
        return alpha * lengths.amax(dim=-1) + (1-alpha) * lengths.mean(dim=-1)


class TemporalDistance(nn.Module):
    """Time-Myopic Eq. 2: 1-exp(-relu(MLP(z_next-z))). No triangle guarantee."""
    def __init__(self, dim, hidden):
        super().__init__()
        self.net = mlp(dim, hidden, 1)
        # A positive initial output avoids the completely dead ReLU solution.
        nn.init.constant_(self.net[-1].bias, 0.5)

    def forward(self, x, y):
        return -torch.expm1(-F.relu(self.net(y-x).squeeze(-1)))


class PrototypeBottleneck(nn.Module):
    """Optional finite state alphabet, with a differentiable usage-entropy surrogate.

    Hard assignments are used in the forward pass. Straight-through gradients train
    the encoder; a codebook loss trains prototypes. This is NOT a reconstruction VAE.
    """
    def __init__(self, count, dim, temperature):
        super().__init__()
        self.codes = nn.Parameter(torch.randn(count, dim))
        self.temperature = temperature
        self.register_buffer("initialized", torch.tensor(False))

    @torch.no_grad()
    def initialize(self, z):
        # Farthest-point seeds spread prototypes over the actually encountered data.
        chosen = [z[0]]
        distance = (z-z[0]).square().sum(-1)
        for _ in range(1, len(self.codes)):
            p = z[distance.argmax()]
            chosen.append(p)
            distance = torch.minimum(distance, (z-p).square().sum(-1))
        self.codes.copy_(torch.stack(chosen))
        self.initialized.fill_(True)

    def forward(self, z):
        if not self.initialized:
            raise RuntimeError("Initialize codebook from a representative replay sample before training")
        squared = (z[:, None] - self.codes[None]).square().mean(-1)
        ids = squared.argmin(-1)
        quantized = self.codes[ids]
        probs = (-squared/self.temperature).softmax(-1)
        usage = probs.mean(0)
        entropy = -(usage * usage.clamp_min(1e-12).log()).sum()
        losses = {
            "codebook": F.mse_loss(quantized, z.detach()),
            "commitment": F.mse_loss(z, quantized.detach()),
            "usage_entropy": entropy,
        }
        return z + (quantized-z).detach(), losses, ids

    def project(self, z):
        ids = (z[..., None, :] - self.codes).square().mean(-1).argmin(-1)
        return self.codes[ids]


class MapModel(nn.Module):
    def __init__(self, observation_shape, model_config, loss_kind, horizon):
        super().__init__()
        c = model_config
        if len(observation_shape) == 3:
            self.encoder = nn.Sequential(
                nn.Conv2d(observation_shape[0], 16, 3, stride=2, padding=1), nn.SiLU(),
                nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.SiLU(),
                nn.AdaptiveAvgPool2d((6, 6)), nn.Flatten(),
                nn.Linear(32*6*6, c.hidden_dim), nn.SiLU(), nn.Linear(c.hidden_dim, c.latent_dim))
        else:
            self.encoder = mlp(observation_shape[0], c.hidden_dim, c.latent_dim, c.hidden_layers)
        self.bottleneck = (PrototypeBottleneck(c.codebook_size, c.latent_dim, c.codebook_temperature)
                           if c.codebook_size else None)
        self.distance_head = (None if loss_kind in {"probabilistic", "dynamics"} else
                              IntervalQuasimetric(c.latent_dim, c.iqe_component_dim)
                              if loss_kind == "qrl" else TemporalDistance(c.latent_dim, c.hidden_dim))
        self.loss_kind = loss_kind
        self.horizon = horizon
        self.latent_bound = c.latent_bound
        self.latent_activation = c.latent_activation

    def encode(self, obs, with_aux=False):
        z = self.encoder(obs)
        if self.loss_kind in {"probabilistic", "dynamics"} and self.latent_activation == "tanh":
            z = self.latent_bound * z.tanh()
        aux, ids = {}, None
        if self.bottleneck:
            z, aux, ids = self.bottleneck(z)
        return (z, aux, ids) if with_aux else z

    def distance(self, x, y):
        if self.distance_head is None:
            raise ValueError("This dynamics model has no learned navigation-distance head")
        value = self.distance_head(x, y)
        return value if self.loss_kind == "qrl" else value * self.horizon


class DynamicsEnsemble(nn.Module):
    """Independent residual transition heads sharing the learned map coordinates."""
    def __init__(self, config, actions=5):
        super().__init__()
        self.actions = actions
        self.heads = nn.ModuleList([mlp(config.latent_dim+actions, config.hidden_dim, config.latent_dim, config.hidden_layers)
                                    for _ in range(config.ensemble_size)])
        for head in self.heads:
            nn.init.normal_(head[-1].weight, std=.01)
            nn.init.zeros_(head[-1].bias)

    def predict_head(self, index, z, action):
        one_hot = F.one_hot(action.long(), self.actions).to(z.dtype)
        return z + self.heads[index](torch.cat([z, one_hot], -1))

    def forward(self, z, action):
        return torch.stack([self.predict_head(i, z, action) for i in range(len(self.heads))])
