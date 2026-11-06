"""One new enabled concept per numbered stage; later curriculum remains staged."""
from copy import deepcopy


def stage_configs():
    config={"phase":1,"time_mode":"continuous"}
    result={1:deepcopy(config)}
    changes={2:{"use_kinematic_closure":True},3:{"use_acceleration":True},
             4:{"use_deformation_gradient":True,"anchor_deformation":True,"spatial_bandlimit":2.},5:{"use_mass_conservation":True},
             6:{"use_constitutive_model":True,"constitutive_model":"linear_to_neo"},
             7:{"use_material_field":True},8:{"momentum_mode":"closed_strong","velocity_source":"gated_deformation"},
             9:{"momentum_mode":"weak","use_weak_form":True},10:{"use_moe":True},
             11:{"use_neural_residual":True},12:{"use_contact":True},
             13:{"use_uncertainty_flow":True},14:{"motion_source":"bspline"},15:{"use_rollout":True}}
    for phase,change in changes.items():
        config.update(change,phase=phase)
        result[phase]=deepcopy(config)
    return result
