from dataclasses import asdict, dataclass, field
import json
from pathlib import Path


@dataclass
class MazeConfig:
    size: int = 12
    seed: int = 7
    loops: float = 0.18
    step_size: float = 0.25
    episode_steps: int = 400
    observation: str = "coordinates"  # coordinates, features, pixels
    nuisance_dims: int = 0
    image_size: int = 48


@dataclass
class ModelConfig:
    latent_dim: int = 32
    hidden_dim: int = 128
    iqe_component_dim: int = 8
    ensemble_size: int = 5
    codebook_size: int = 0  # 0 disables quantization and compression
    codebook_temperature: float = 0.1


@dataclass
class TrainConfig:
    loss: str = "qrl"  # qrl or time_myopic
    updates: int = 3000
    batch_size: int = 256
    lr: float = 0.001
    dual_lr: float = 0.01
    epsilon: float = 0.1
    initial_multiplier: float = 1.0
    push_scale: float = 20.0  # 0: linear push; otherwise softplus(-d / scale)
    horizon: int = 20
    max_pair_steps: int = 100
    exact_revisit_min: bool = False
    compression_weight: float = 0.01
    commitment_weight: float = 0.25
    log_every: int = 100


@dataclass
class PlannerConfig:
    behavior: str = "random"  # random, distance, disagreement, hybrid
    horizon: int = 6
    beam_width: int = 32
    distance_reference: str = "episode_start"  # episode_start or current
    disagreement_weight: float = 1.0
    uncertainty_discount: float = 0.9
    epsilon_random: float = 0.1
    action_repeat_probability: float = 0.8


@dataclass
class ExperimentConfig:
    seed: int = 0
    device: str = "cpu"
    threads: int = 1
    collect_steps: int = 20000
    warmup_steps: int = 1000
    train_every: int = 10
    updates_per_train: int = 5
    maze: MazeConfig = field(default_factory=MazeConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)

    def validate(self):
        assert self.maze.size >= 2 and 0 <= self.maze.loops <= 1
        assert 0 < self.maze.step_size < 1 and self.maze.episode_steps > 0
        assert self.maze.observation in {"coordinates", "features", "pixels"}
        assert self.maze.nuisance_dims >= 0 and self.maze.image_size >= 16
        assert not (self.maze.observation == "pixels" and self.maze.nuisance_dims)
        assert self.model.latent_dim >= 2 and self.model.hidden_dim > 0
        assert self.model.iqe_component_dim > 0
        assert self.model.latent_dim % self.model.iqe_component_dim == 0
        assert self.model.ensemble_size >= 2 and self.model.codebook_size >= 0
        assert self.model.codebook_temperature > 0
        assert self.train.loss in {"qrl", "time_myopic"}
        assert self.train.updates > 0 and self.train.batch_size > 1
        assert self.train.lr > 0 and self.train.dual_lr > 0
        assert self.train.epsilon >= 0 and self.train.initial_multiplier >= 0
        assert self.train.push_scale >= 0 and self.train.horizon > 0
        assert self.train.max_pair_steps >= self.train.horizon
        assert self.train.compression_weight >= 0 and self.train.commitment_weight >= 0
        assert self.train.log_every > 0 and self.threads > 0
        assert self.collect_steps > 0 and self.warmup_steps >= 0
        assert self.train_every > 0 and self.updates_per_train > 0
        assert self.planner.behavior in {"random", "distance", "disagreement", "hybrid"}
        assert self.planner.distance_reference in {"episode_start", "current"}
        assert self.planner.horizon > 0 and self.planner.beam_width > 0
        assert self.planner.disagreement_weight >= 0
        assert 0 <= self.planner.uncertainty_discount <= 1
        assert 0 <= self.planner.epsilon_random <= 1
        assert 0 <= self.planner.action_repeat_probability <= 1
        return self

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n")


def config_from_dict(data):
    data = dict(data)
    for key, cls in [("maze", MazeConfig), ("model", ModelConfig),
                     ("train", TrainConfig), ("planner", PlannerConfig)]:
        if key in data:
            data[key] = cls(**data[key])
    return ExperimentConfig(**data).validate()


def load_config(path=None, overrides=()):
    data = asdict(ExperimentConfig())
    if path:
        supplied = json.loads(Path(path).read_text())
        for key, value in supplied.items():
            if isinstance(value, dict) and key in data:
                data[key].update(value)
            else:
                data[key] = value
    for override in overrides:
        key, value = override.split("=", 1)
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            pass
        target = data
        parts = key.split(".")
        for part in parts[:-1]:
            target = target[part]
        if parts[-1] not in target:
            raise ValueError(f"Unknown setting: {key}")
        target[parts[-1]] = value
    return config_from_dict(data)
