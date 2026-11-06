grid_args = dict(
    canonical_num_levels=16,
    canonical_level_dim=2,
    canonical_base_resolution=16,
    canonical_desired_resolution=2048,
    canonical_log2_hashmap_size=19,

    deform_num_levels=32,
    deform_level_dim=2,
    deform_base_resolution=[8, 8, 8],
    deform_desired_resolution=[32, 32, 16],
    deform_log2_hashmap_size=19,

    bound=1.6,
)

network_args = dict(
    depth=0,
    width=256,
    directional=True,
)

grid_lr_scale = 50.0
network_lr_scale = 5.0

lambda_spatial_tv = 0.5
spatial_downsample_ratio = 0.1
spatial_perturb_range = 1e-2

lambda_temporal_tv = 0.5
temporal_downsample_ratio = 0.1
temporal_perturb_range = [1e-2, 1e-2, 1e-2, 1e-3]

use_flow = False
dataset_type = "PIDG"
network_args = dict(network_args, width=256, depth=0, metric_residual_width=0)
warm_up = 0
densify_until_iter = 0
position_lr_init = 2e-05
position_lr_final = 1e-07
position_lr_max_steps = 100
deform_lr_max_steps = 100
feature_lr = 0.0005
opacity_lr = 0.005
scaling_lr = 5e-05
rotation_lr = 0.0001
grid_lr_scale = 25.0
network_lr_scale = 5.0
velocity_net_lr = 0.0001
force_sigma_lr = 0.0001
lambda_dssim = 0.2
lambda_spatial_tv = 0.5
lambda_temporal_tv = 0.5
