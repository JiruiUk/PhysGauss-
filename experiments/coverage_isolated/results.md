# Independent component effects: C4 / C5

unsegmented affine-only vs train-alpha-guided density-only; zero mask loss in both

Status: completed
Dependency: completed
Active: C5_density_only / metrics_test

Both new arms start from the same random iteration0 as C0-C3;50k total; no pretrained parent.
C4 loads no separate alpha target; baseline RGBA-to-RGB compositing is unchanged. C5 uses training alpha for point-addition selection, with L_mask=0.
Affine formulation and density thresholds/schedule match the cumulative runs. No object-ID isolation is claimed.

| Model | Components | Iter | Validation PSNR | Test PSNR | SSIM | LPIPS | MS-SSIM | Foreground PSNR | Mask IoU | Interior hole rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C0_V3_reference | V3 reference | 50000/50000 | 39.16217 | 39.7239647 | 0.9943330 | 0.0106815 | 0.9961962 | 30.6765039 | 0.2064003 | 0.0000185 |
| C1_mask | mask | 50000/50000 | 35.55297 | 35.7459450 | 0.9927235 | 0.0133286 | 0.9927815 | 26.4415244 | 0.9763858 | 0.0077157 |
| C2_mask_affine | mask + affine | 50000/50000 | 36.88692 | 37.1511269 | 0.9930764 | 0.0128118 | 0.9944401 | 28.1914153 | 0.9841781 | 0.0003712 |
| C3_mask_affine_density | mask + affine + density | 50000/50000 | 35.40388 | 35.8634186 | 0.9926928 | 0.0133028 | 0.9927368 | 26.9138169 | 0.9684211 | 0.0093528 |
| C4_affine_only | affine ONLY | 50000/50000 | 38.23327 | 38.6373291 | 0.9938812 | 0.0112210 | 0.9953121 | 29.5213014 | 0.2168333 | 0.0000301 |
| C5_density_only | density ONLY | 50000/50000 | 38.86942 | 39.3918228 | 0.9943611 | 0.0105702 | 0.9961648 | 30.2986353 | 0.2360323 | 0.0000054 |

Interpret independent effects as C4-C0 and C5-C0; mask-only remains C1-C0.
C2-C1 and C3-C2 instead describe conditional effects when earlier components are enabled.
Checkpoint choices use validation only; lock C4/C5 selections before their final test exports.
Prior and early test exposure remains a limitation; new runs do not create a fresh untouched test set.

GT/render pairs absolute directory: F:\A_Edinburgh\PhysGauss\runs\pidg_coverage_isolated\render_comparisons

[5-second monitor](monitor.html) | [Plan](plan.json) | [GPU preflight](preflight.json)

Updated UTC: 2026-10-27T03:24:32.298567+00:00
