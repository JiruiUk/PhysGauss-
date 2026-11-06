from dataclasses import asdict, dataclass, field
from pathlib import Path
import json
import math


@dataclass
class LossConfig:
    rgb: float = 1.0
    ssim: float = 0.1
    uncertainty: float = 0.01
    depth: float = 0.1
    flow: float = 0.02
    trajectory: float = 0.1
    momentum: float = 0.01
    uncertainty_prior: float = 0.001


@dataclass
class TrainConfig:
    seed: int = 7
    device: str = "cpu"
    steps: int = 30
    lr: float = 0.003
    hidden_dim: int = 32
    frequencies: int = 2
    canonical_time: float = 0.0
    density: float = 1.0
    body_acceleration: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    physics_warmup: int = 5
    physics_ramp: int = 10
    collocation_count: int = 8
    adaptive_sampling: bool = True
    adaptive_uniform_fraction: float = 0.3
    log_every: int = 5
    checkpoint_every: int = 15
    cpu_threads: int = 2
    freeze_canonical_geometry: bool = False
    loss: LossConfig = field(default_factory=LossConfig)

    def validate(self):
        numeric = [self.lr, self.canonical_time, self.density, *self.body_acceleration,
                   *asdict(self.loss).values()]
        if not all(math.isfinite(v) for v in numeric):
            raise ValueError("config contains nonfinite numeric values")
        if self.steps < 1 or self.lr <= 0 or self.hidden_dim < 4:
            raise ValueError("steps, lr and hidden_dim must be positive (hidden_dim >= 4)")
        if self.density <= 0 or self.frequencies < 0 or self.collocation_count < 1:
            raise ValueError("invalid density, frequencies or collocation_count")
        if self.physics_warmup < 0 or self.physics_ramp < 1:
            raise ValueError("invalid physics schedule")
        if self.log_every < 1 or self.checkpoint_every < 1 or self.cpu_threads < 1:
            raise ValueError("logging intervals and cpu_threads must be positive")
        if len(self.body_acceleration) != 3:
            raise ValueError("body_acceleration must contain three values")
        if not 0 <= self.adaptive_uniform_fraction <= 1:
            raise ValueError("adaptive_uniform_fraction must lie in [0, 1]")
        if any(v < 0 for v in asdict(self.loss).values()):
            raise ValueError("loss weights must be nonnegative")
        return self

    @classmethod
    def load(cls, path):
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        raw["loss"] = LossConfig(**raw.get("loss", {}))
        return cls(**raw).validate()

    def to_dict(self):
        return asdict(self)
