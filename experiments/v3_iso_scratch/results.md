# V3 / V3+iso: from-scratch 50k

FROM RANDOM INITIALIZATION; 50k total per arm; 120 train/30 validation/50 test; iso=0 vs .001 from step1

Status: completed
Active: V3_scratch_control / metrics_test

No trained model is loaded. Both arms copy the same full step-zero state with empty optimizer histories.
V3 image losses/shared time; original PIDG initial-training optimization, warm-up3000 and densification through15000.
Iso weight0.001 is computed from step1. Validation alone selects checkpoints; test stays locked until both arms finish.

| Model | Total training iter | Latest validation PSNR |
|---|---:|---:|
| V3_scratch_control | 50000/50000 | 39.16217403411865 |
| V3_scratch_iso_1e3 | 50000/50000 | 39.0024611155192 |

## Fixed final50k test results

| Model | PSNR | SSIM | LPIPS | MS-SSIM |
|---|---:|---:|---:|---:|
| V3_scratch_control | 39.7239647 | 0.9943330 | 0.0106815 | 0.9961962 |
| V3_scratch_iso_1e3 | 39.5581360 | 0.9943981 | 0.0111843 | 0.9963505 |

| Metric | Iso minus control | Improved frames / 50 |
|---|---:|---:|
| PSNR | -0.1658287 | 18 |
| SSIM | +0.0000650 | 20 |
| LPIPS | +0.0005028 | 10 |
| MS-SSIM | +0.0001543 | 22 |

## Validation-selected snapshots: final test scores

| Model | Validation objective | Selected iter | PSNR | SSIM | LPIPS | MS-SSIM |
|---|---|---:|---:|---:|---:|---:|
| V3_scratch_control | PSNR | 50000 | 39.7239647 | 0.9943330 | 0.0106815 | 0.9961962 |
| V3_scratch_control | SSIM | 50000 | 39.7239647 | 0.9943330 | 0.0106815 | 0.9961962 |
| V3_scratch_control | LPIPS | 50000 | 39.7239647 | 0.9943330 | 0.0106815 | 0.9961962 |
| V3_scratch_control | MS-SSIM | 50000 | 39.7239647 | 0.9943330 | 0.0106815 | 0.9961962 |
| V3_scratch_iso_1e3 | PSNR | 50000 | 39.5581360 | 0.9943981 | 0.0111843 | 0.9963505 |
| V3_scratch_iso_1e3 | SSIM | 50000 | 39.5581360 | 0.9943981 | 0.0111843 | 0.9963505 |
| V3_scratch_iso_1e3 | LPIPS | 50000 | 39.5581360 | 0.9943981 | 0.0111843 | 0.9963505 |
| V3_scratch_iso_1e3 | MS-SSIM | 47000 | 39.5092545 | 0.9943807 | 0.0112619 | 0.9963357 |

This replaces the paused continuation experiment; no continuation checkpoint is used.
The same official test set has historical exposure; prospective separation does not undo that history.
[5-second graphical monitor](monitor.html) | [Plan](plan.json) | [Initial state](initialization.json) | [GPU preflight](preflight.json)

GT/render pairs absolute directory: F:\A_Edinburgh\PhysGauss\runs\pidg_v3_iso_scratch\render_comparisons

Updated UTC: 2026-10-26T19:54:07.559962+00:00
