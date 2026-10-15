# M4 fixed vs learnable material ablation

Both runs: Rubber Duck, 300 iterations, width400, seed0, identical initialization protocol and loss weights; original full official train/test split. Engineering smoke, not a 50k research comparison.

| Method | PSNR | SSIM | LPIPS | Legacy CMR sampled loss | Constitutive raw | mu | kappa |
|---|---:|---:|---:|---:|---:|---:|---:|
| A2a | 20.80272 | 0.91000 | 0.14297 | 0.0001911 | 5.09439 | 1.00000000 | 10.00000000 |
| A2b | 21.00809 | 0.91041 | 0.14297 | 0.0001918 | 6.82177 | 0.99998795 | 9.99998031 |

A2a fixed the new mu/kappa bank. The original PIDG neural/stress parameters, including force_sigma, remain learnable in both configurations. A2b adds only two scene-level material parameters. Their finite updates and optimizer checkpoint were verified.

No material recovery, future prediction or reconstruction improvement is claimed from these short runs. Parameter trajectories are in M4_parameter_trajectory.csv.
