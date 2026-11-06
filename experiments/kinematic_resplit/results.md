# V3 / V4 time-disjoint retraining

120 train / 30 validation / unchanged 50 test; global frame clock

Train and validation loader views contain no final-test RGB. Validation selects snapshots; test is evaluated only after both models finish and selections are locked.
The prior official test set has historical tuning exposure. This is a prospectively separated re-evaluation, not a new untouched benchmark.

Status: completed
Active: V4_frame_time_kinematic / metrics_test

| Stage | Iteration | Latest validation PSNR |
|---|---:|---:|
| P0_train_only | 50000/50000 | 39.09530824025472 |
| V3_frame_time | 50000/50000 | 39.65109151204427 |
| V4_frame_time_kinematic | 50000/50000 | 39.69553311665853 |

## Final test comparison (same 50 RGB, poses, and frame-index times)

| Model | Metric | Previous final50k | Resplit final50k | Change |
|---|---|---:|---:|---:|
| V3_frame_time | PSNR | 37.3459435 | 40.0917091 | +2.7457657 |
| V3_frame_time | SSIM | 0.9937512 | 0.9946094 | +0.0008581 |
| V3_frame_time | LPIPS | 0.0124080 | 0.0103923 | -0.0020157 |
| V3_frame_time | MS-SSIM | 0.9951608 | 0.9962949 | +0.0011341 |
| V4_frame_time_kinematic | PSNR | 38.7692337 | 40.1055183 | +1.3362846 |
| V4_frame_time_kinematic | SSIM | 0.9939139 | 0.9946030 | +0.0006891 |
| V4_frame_time_kinematic | LPIPS | 0.0119559 | 0.0104603 | -0.0014955 |
| V4_frame_time_kinematic | MS-SSIM | 0.9960272 | 0.9962841 | +0.0002570 |

## Controlled new V4 versus new V3 at final50k

| Metric | V4 minus V3 | Improved frames / 50 |
|---|---:|---:|
| PSNR | +0.0138092 | 30 |
| SSIM | -0.0000063 | 29 |
| LPIPS | +0.0000680 | 26 |
| MS-SSIM | -0.0000108 | 28 |

## Validation-selected snapshots: final test scores

| Model | Validation objective | Selected iter | PSNR | SSIM | LPIPS | MS-SSIM |
|---|---|---:|---:|---:|---:|---:|
| V3_frame_time | PSNR | 50000 | 40.0917091 | 0.9946094 | 0.0103923 | 0.9962949 |
| V3_frame_time | SSIM | 50000 | 40.0917091 | 0.9946094 | 0.0103923 | 0.9962949 |
| V3_frame_time | LPIPS | 50000 | 40.0917091 | 0.9946094 | 0.0103923 | 0.9962949 |
| V3_frame_time | MS-SSIM | 50000 | 40.0917091 | 0.9946094 | 0.0103923 | 0.9962949 |
| V4_frame_time_kinematic | PSNR | 50000 | 40.1055183 | 0.9946030 | 0.0104603 | 0.9962841 |
| V4_frame_time_kinematic | SSIM | 50000 | 40.1055183 | 0.9946030 | 0.0104603 | 0.9962841 |
| V4_frame_time_kinematic | LPIPS | 50000 | 40.1055183 | 0.9946030 | 0.0104603 | 0.9962841 |
| V4_frame_time_kinematic | MS-SSIM | 50000 | 40.1055183 | 0.9946030 | 0.0104603 | 0.9962841 |

Raw before/after changes include fewer training images and a newly trained parent. They do not isolate the effect of validation separation.
The controlled method comparison is new V4 versus new V3, including per-frame metrics.

[5-second graphical monitor](monitor.html) | [Split audit](split_audit.json) | [Plan](plan.json)

GT/render pairs: F:\A_Edinburgh\PhysGauss\runs\pidg_kinematic_resplit\render_comparisons

Updated UTC: 2026-10-26T03:24:10.097850+00:00
