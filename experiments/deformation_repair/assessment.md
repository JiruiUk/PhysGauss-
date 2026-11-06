# H3 worst-frame repair: final50k comparison

Model chosen on validation: H3_reused. No improvement is assumed from adding a module.

| Model | PSNR | SSIM | LPIPS | MS-SSIM | Worst5 PSNR | Core low90 % | BG spill % |
|---|---:|---:|---:|---:|---:|---:|---:|
| C1_reused | 40.372128 | 0.9946761 | 0.0104375 | 0.9965463 | 33.85023 | 0.090762 | 1.004121 |
| H3_reused | 40.878231 | 0.9950647 | 0.0094411 | 0.9969448 | 34.17046 | 0.060594 | 0.803588 |
| D1_motion_affine | 40.915855 | 0.9950998 | 0.0095761 | 0.9969874 | 33.85937 | 0.075367 | 0.944347 |
| D2_covariance_continuity | 39.925003 | 0.9944530 | 0.0106770 | 0.9964454 | 33.61484 | 0.118725 | 1.202941 |
| D3_combined | 40.618591 | 0.9948962 | 0.0098604 | 0.9968740 | 34.28660 | 0.064816 | 0.952013 |

GT/render gallery: F:\A_Edinburgh\PhysGauss\runs\pidg_deformation_repair\render_exports\final50000\gt_h3_repairs\index.html

Every source panel preserves1600x900 pixels. Each variant also has renders/ and gt_render/ directories.
