# Original PIDG + foreground centroid ablation

Base: original | Status: completed
Active: G1_centroid / metrics_test

120 train /30 validation /50 fixed test; two50k total arms from one shared random start and full3k state.
G0: center weight0. G1: center weight1, start after3k, ramp2k. No mask L1, affine, iso, kinematic or late density additions.
Original PIDG objective/physics-time convention on the clean temporal split; V3 is a separately named optional base.
This is a 2D whole-scene alpha centroid, not a measured physical3D center of mass. Training targets never use validation/test alpha.

| Stage | Current iter | Latest validation PSNR | Test PSNR |
|---|---:|---:|---:|
| shared_prefix | 3000/3000 | 17.770377540588377 | - |
| G0_control | 50000/50000 | 36.727505620320635 | 37.139381408691406 |
| G1_centroid | 50000/50000 | 37.118106079101565 | 37.527896881103516 |

Final50k G1 minus G0: {"PSNR": 0.3885154724121094, "SSIM": -0.00026494264602661133, "LPIPS": 0.0006608851253986359, "MS-SSIM": 9.834766387939453e-05}

[5-second graphical monitor](monitor.html) | [Plan](plan.json) | [Preflight](preflight.json)

GT/render galleries: F:\A_Edinburgh\PhysGauss\runs\pidg_centroid_tuning\render_comparisons

Updated UTC: 2026-10-27T14:29:54.808756+00:00

## Verified final assessment

[Full assessment and plots](assessment.md)

Mixed: better mean PSNR/MS-SSIM and 2D alpha/centroid metrics; worse SSIM/LPIPS. PSNR improves on23/50 frames with a negative median; large early-frame gains drive the mean.
