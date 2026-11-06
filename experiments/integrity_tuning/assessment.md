# Completeness-first experiment assessment

C1 is reused; all promoted models trained to50k total from the identical full3k state.
Promotion and checkpoint selection used validation only. Test data did not select variants.
Success requires a lower validation integrity score, four-metric quality guard, and background spill within+0.2 percentage point.

| Model | Validation-supported improvement | Test PSNR | SSIM | LPIPS | MS-SSIM | Test core low50 | Core low90 | BG spill |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| H3_combined | True | 40.8782310 | 0.9950647 | 0.0094411 | 0.9969448 | 0.0000367 | 0.0006059 | 0.0080359 |
| H2_temporal_cohesion | True | 39.9131584 | 0.9945875 | 0.0103162 | 0.9966189 | 0.0000966 | 0.0009248 | 0.0117359 |

These are2D opacity/appearance and projected-connectivity diagnostics, not proof of complete3D surfaces.
Opaque appearance errors are reported separately from holes. No learned semantic/material segmentation was added.

GT + baseline + promoted models: F:\A_Edinburgh\PhysGauss\runs\pidg_integrity_tuning\render_exports\final50000\gt_baseline_variants\index.html
