# Continued variants: full50k comparison

H1/H4/F1/F2 resumed original20k full checkpoints to50k total. C1/H2/H3 reused without training.
Original recipes, validation selection and optimizer/RNG state were retained. Earlier pilot rejection remains recorded.

| Model | Test PSNR | Delta vs C1 | SSIM | LPIPS | MS-SSIM | Interior low50 (%) | Interior low90 (%) | BG spill (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C1_reused | 40.3721275 | +0.0000000 | 0.9946761 | 0.0104375 | 0.9965463 | 0.0108231 | 0.0907618 | 1.0041206 |
| H2_temporal_cohesion | 39.9131584 | -0.4589691 | 0.9945875 | 0.0103162 | 0.9966189 | 0.0096601 | 0.0924796 | 1.1735855 |
| H3_combined | 40.8782310 | +0.5061035 | 0.9950647 | 0.0094411 | 0.9969448 | 0.0036730 | 0.0605940 | 0.8035876 |
| H1_balanced_coverage | 40.0026665 | -0.3694611 | 0.9945776 | 0.0105493 | 0.9965711 | 0.0121225 | 0.0838700 | 1.2532356 |
| H4_color_density | 39.3668823 | -1.0052452 | 0.9943725 | 0.0106298 | 0.9964535 | 0.0118166 | 0.0789795 | 1.3173882 |
| F1_feedback | 39.7815933 | -0.5905342 | 0.9943615 | 0.0109807 | 0.9964935 | 0.0035652 | 0.0654289 | 1.2819685 |
| F2_conservative_retry | 39.9095612 | -0.4625664 | 0.9944645 | 0.0106980 | 0.9965023 | 0.0098064 | 0.1153286 | 1.2607099 |

These ratios describe2D rendered alpha/appearance, not3D holes. No improvement is assumed merely because a run reached50k.

GT/render gallery: F:\A_Edinburgh\PhysGauss\runs\pidg_integrity_continuation\render_exports\final50000\gt_all_models\index.html
