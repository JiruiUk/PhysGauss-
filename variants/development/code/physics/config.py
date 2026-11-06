from dataclasses import dataclass, asdict
from functools import lru_cache
import json
import os


@dataclass
class PhysicsConfig:
    phase: int = 1
    time_mode: str = "continuous"
    use_kinematic_closure: bool = False
    velocity_source: str = "network"
    use_acceleration: bool = False
    use_deformation_gradient: bool = False
    anchor_deformation: bool = False
    spatial_bandlimit: float = 0.
    use_mass_conservation: bool = False
    use_volume_preservation: bool = False
    use_constitutive_model: bool = False
    constitutive_model: str = "linear"
    constitutive_transition: int = 25000
    use_material_field: bool = False
    momentum_mode: str = "baseline_cmr"
    use_weak_form: bool = False
    use_moe: bool = False
    use_neural_residual: bool = False
    use_contact: bool = False
    gravity_enabled: bool = False
    gravity_vector: tuple = (0.,-9.81,0.)
    contact_stiffness: float = 100.
    contact_offset: float = -1.3
    use_uncertainty_flow: bool = False
    motion_source: str = "hash"
    use_rollout: bool = False
    rho0: float = 1.
    physics_samples: int = 64
    weak_form_k: int = 16
    weak_form_bandwidth_scale: float = 1.5
    lambda_kin: float = .001
    lambda_mass: float = .01
    lambda_volume: float = .01
    lambda_constitutive: float = .001
    lambda_momentum: float = .01
    lambda_weak_momentum: float = .01
    lambda_material_smooth: float = .001
    lambda_residual: float = .001
    lambda_contact: float = .01
    lambda_rollout: float = .001
    lambda_jacobian: float = .01
    clip_grad_norm: float = 100.

    def validate(self):
        if self.time_mode not in {"continuous","legacy"}: raise ValueError("Invalid time mode")
        if self.velocity_source not in {"network","deformation","gated_deformation"}: raise ValueError("Invalid velocity source")
        if self.momentum_mode not in {"baseline_cmr","closed_strong","weak"}: raise ValueError("Invalid momentum mode")
        if self.constitutive_model not in {"linear","neo_hookean","linear_to_neo"}: raise ValueError("Invalid constitutive model")
        if self.motion_source not in {"hash","bspline"}: raise ValueError("Invalid motion source")
        if self.rho0<=0 or self.physics_samples<4: raise ValueError("Invalid quadrature settings")
        if self.use_constitutive_model and not self.use_deformation_gradient: raise ValueError("Constitutive stress requires F")
        if self.use_mass_conservation and not self.use_deformation_gradient: raise ValueError("Mass closure requires J")
        if self.momentum_mode!="baseline_cmr" and not (self.use_acceleration and self.use_constitutive_model):
            raise ValueError("Closed momentum needs trajectory acceleration and constitutive stress")
        if self.use_weak_form and self.momentum_mode!="weak": raise ValueError("Inconsistent weak mode")
        if (self.use_moe or self.use_neural_residual or self.use_material_field) and not self.use_constitutive_model:
            raise ValueError("Material enhancements require constitutive closure")
        if any(value<0 for key,value in asdict(self).items() if key.startswith("lambda_")):
            raise ValueError("Negative loss weight")
        return self


@lru_cache(maxsize=32)
def _load_config(path):
    if path is None: return PhysicsConfig().validate()
    with open(path,encoding="utf-8") as stream: raw=json.load(stream)
    return PhysicsConfig(**raw).validate()


def load_config():
    return _load_config(os.environ.get("PIDG_UPGRADE_CONFIG"))
