# Ordered foreground coverage ablation

Ordered additions; same random zero state; 50k total; iso=0; train120/val30/test50

Status: completed
Active: C3_mask_affine_density / metrics_test

| Model | Iteration | Mask weight | Local affine weight | Coverage densify | Latest validation PSNR |
|---|---:|---:|---:|---|---:|
| C0_V3_reference | 50000/50000 | 0.0 | 0.0 | False | 39.16217403411865 |
| C1_mask | 50000/50000 | 0.1 | 0.0 | False | 35.552971903483076 |
| C2_mask_affine | 50000/50000 | 0.1 | 0.0001 | False | 36.886919085184736 |
| C3_mask_affine_density | 50000/50000 | 0.1 | 0.0001 | True | 35.4038808186849 |

## Locked final50k test comparison

| Model | PSNR | SSIM | LPIPS | MS-SSIM | Foreground PSNR | Mask IoU | Interior hole rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| C0_V3_reference | 39.7239647 | 0.9943330 | 0.0106815 | 0.9961962 | 30.676503944396973 | 0.20640030351111308 | 1.8548222877658985e-05 |
| C1_mask | 35.7459450 | 0.9927235 | 0.0133286 | 0.9927815 | 26.44152442932129 | 0.9763858430889545 | 0.007715729195381101 |
| C2_mask_affine | 37.1511269 | 0.9930764 | 0.0128118 | 0.9944401 | 28.191415271759034 | 0.984178119164153 | 0.0003711879335953236 |
| C3_mask_affine_density | 35.8634186 | 0.9926928 | 0.0133028 | 0.9927368 | 26.913816852569582 | 0.9684211047803589 | 0.009352784513816693 |

Training masks: original RGBA alpha, L1 mean over all pixels, weight0.1 from step1.
Local affine: weight1e-4, every10 steps with interval compensation, starts3000/ramp2000; permits affine stretch/shear.
Coverage seeds: contributors to eroded foreground alpha<0.7, target0.95, sampled every20 steps after16000; >=3 distinct training times.
At18000..36000 every2000: add <=512 points, cap200000; keep parents; no extra pruning/reset. Original densification ends15000.
Coverage-only metrics use threshold0.5 and 3-pixel eroded GT interior; foreground PSNR uses alpha>0.5.
The reference reuses completed V3 scratch control; all candidates start from identical random iteration0, not from the reference weights.
Final test remains historically exposed. Single seed; paired arm comparisons do not establish statistical significance.

Render / GT pairs absolute directory: F:\A_Edinburgh\PhysGauss\runs\pidg_coverage_tuning\render_comparisons

[5-second charts](monitor.html) | [Plan](plan.json) | [Training alpha audit](alpha_audit.json)

Updated UTC: 2026-10-27T01:01:52.805885+00:00
