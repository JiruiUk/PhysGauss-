# C1 / C2 / C3 color-repair retraining

Status: completed | Active: C3_repaired / metrics_test

Each arm: random initialization,50k total. Shared3k; C2/C3 share full15k pre-density state.
V3 image objective and shared time retained. Positive SH radiance; inverse initial DC; delayed/ramped mask0.03; foreground RGB0.1.
C2/C3 affine5e-5 within radius0.05; no claimed object-ID isolation. C3 uses training RGB-consistent clones, at most256/window.
Only120 training RGB/alpha targets enter losses;30 validation selects four snapshots;50 final test stays locked until all complete.

| Stage | Iteration | Validation PSNR | Test PSNR | SSIM | LPIPS | MS-SSIM |
|---|---:|---:|---:|---:|---:|---:|
| shared_prefix | 3000/3000 | 17.570457712809244 | - | - | - | - |
| affine_prefix | 15000/15000 | 33.77849369049072 | - | - | - | - |
| C1_repaired | 50000/50000 | 39.664894612630206 | 40.3721275 | 0.9946761 | 0.0104375 | 0.9965463 |
| C2_repaired | 50000/50000 | 38.58421732584635 | 39.2913704 | 0.9942592 | 0.0111913 | 0.9962929 |
| C3_repaired | 50000/50000 | 38.55284156799316 | 39.2737961 | 0.9942705 | 0.0112402 | 0.9963008 |

C1_repaired minus historical C1_mask: {"PSNR": 4.626182556152344, "SSIM": 0.0019525885581970215, "LPIPS": -0.0028910692781209946, "MS-SSIM": 0.0037648677825927734}

C2_repaired minus historical C2_mask_affine: {"PSNR": 2.1402435302734375, "SSIM": 0.0011827945709228516, "LPIPS": -0.0016204547137022018, "MS-SSIM": 0.0018527507781982422}

C3_repaired minus historical C3_mask_affine_density: {"PSNR": 3.4103775024414062, "SSIM": 0.0015777349472045898, "LPIPS": -0.0020626364275813103, "MS-SSIM": 0.0035639405250549316}

These are repair-package comparisons, not attribution to one modified component.
[5-second charts](monitor.html) | [Plan](plan.json) | [Original color diagnostic](original_color_diagnostic.json) | [Preflight](preflight.json)

GT/render galleries: F:\A_Edinburgh\PhysGauss\runs\pidg_color_repair\render_comparisons

Updated UTC: 2026-10-27T19:02:29.679935+00:00
