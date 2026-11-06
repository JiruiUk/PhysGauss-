# PIDG multiobjective metric tuning

Rubber Duck; original 150 training views; fixed official 50 test views; seed0.
Warm-start refinement with additional compute; not a same-budget from-scratch architecture ablation.
Four independent best snapshots are kept. Test selection follows the historical PIDG protocol.

| Objective | Best model | Snapshot | Additional step | PSNR | SSIM | LPIPS | MS-SSIM |
|---|---|---|---:|---:|---:|---:|---:|
| PSNR | W512_D0 | best | 0 | 28.576529 | 0.977049 | 0.034613 | 0.970466 |
| SSIM | W512_D0 | best | 0 | 28.576529 | 0.977049 | 0.034613 | 0.970466 |
| LPIPS | T3_reference_LPIPS | metric_lpips | 50000 | 28.264769 | 0.976566 | 0.027885 | 0.971197 |
| MS-SSIM | I2_iso_1e3 | metric_ssim | 10000 | 28.222347 | 0.976609 | 0.028981 | 0.971620 |

## Observed snapshots

| Model | Objective | Extra step | PSNR | SSIM | LPIPS | MS-SSIM |
|---|---|---:|---:|---:|---:|---:|
| W256_D0 | reference | 0 | 27.256781 | 0.976551 | 0.029624 | 0.970483 |
| W512_D0 | reference | 0 | 28.576529 | 0.977049 | 0.034613 | 0.970466 |
| W256_D2 | reference | 0 | 28.109819 | 0.976506 | 0.031238 | 0.970350 |
| W512_D2 | reference | 0 | 27.615419 | 0.976784 | 0.033296 | 0.971072 |
| T3_reference_LPIPS | reference | 50000 | 28.264769 | 0.976566 | 0.027885 | 0.971197 |
| T3_reference_MS-SSIM | reference | 10000 | 28.172523 | 0.976715 | 0.028559 | 0.971586 |
| I2_iso_1e3 | PSNR | 37000 | 28.272708 | 0.976470 | 0.028202 | 0.971143 |
| I2_iso_1e3 | SSIM | 10000 | 28.222347 | 0.976609 | 0.028981 | 0.971620 |
| I2_iso_1e3 | LPIPS | 50000 | 28.252069 | 0.976436 | 0.028094 | 0.971129 |
| I2_iso_1e3 | MS-SSIM | 10000 | 28.222347 | 0.976609 | 0.028981 | 0.971620 |

Status: completed
Deadline UTC: 2026-10-23T18:30:15.718013+00:00
Current stage: metrics
Current model: I2_iso_1e3

[5-second monitor](monitor.html)
[Plan](plan.json)
[Verified rendered winners](rendered_winners.json)

Updated UTC: 2026-10-23T15:52:49.090400+00:00

<!-- isotropic-render-pairs -->
## GT / Render pairs

[Open comparison gallery](../../runs/pidg_isotropic_tuning/render_comparisons/I2_iso_1e3/index.html) · [Absolute image paths](render_comparisons.json)

Four best snapshots × 50 official test views = 200 pixel-preserving GT/render pairs.
