# PIDG multiobjective metric tuning

Rubber Duck; original 150 training views; fixed official 50 test views; seed0.
Warm-start refinement with additional compute; not a same-budget from-scratch architecture ablation.
Four independent best snapshots are kept. Test selection follows the historical PIDG protocol.

| Objective | Best model | Snapshot | Additional step | PSNR | SSIM | LPIPS | MS-SSIM |
|---|---|---|---:|---:|---:|---:|---:|
| PSNR | W512_D0 | best | 0 | 28.576529 | 0.977049 | 0.034613 | 0.970466 |
| SSIM | W512_D0 | best | 0 | 28.576529 | 0.977049 | 0.034613 | 0.970466 |
| LPIPS | T3_baseline_perceptual | metric_lpips | 50000 | 28.264767 | 0.976566 | 0.027885 | 0.971197 |
| MS-SSIM | T3_baseline_perceptual | metric_ms_ssim | 10000 | 28.172525 | 0.976715 | 0.028559 | 0.971586 |

## Observed snapshots

| Model | Objective | Extra step | PSNR | SSIM | LPIPS | MS-SSIM |
|---|---|---:|---:|---:|---:|---:|
| W256_D0 | reference | 0 | 27.256781 | 0.976551 | 0.029624 | 0.970483 |
| W512_D0 | reference | 0 | 28.576529 | 0.977049 | 0.034613 | 0.970466 |
| W256_D2 | reference | 0 | 28.109819 | 0.976506 | 0.031238 | 0.970350 |
| W512_D2 | reference | 0 | 27.615419 | 0.976784 | 0.033296 | 0.971072 |
| T0_continue | PSNR | 0 | 28.576529 | 0.977049 | 0.034613 | 0.970466 |
| T0_continue | SSIM | 0 | 28.576529 | 0.977049 | 0.034613 | 0.970466 |
| T0_continue | LPIPS | 50000 | 28.190596 | 0.976718 | 0.029733 | 0.970084 |
| T0_continue | MS-SSIM | 10000 | 28.294974 | 0.977003 | 0.031376 | 0.970593 |
| T1_mse | PSNR | 0 | 28.576529 | 0.977049 | 0.034613 | 0.970466 |
| T1_mse | SSIM | 0 | 28.576529 | 0.977049 | 0.034613 | 0.970466 |
| T1_mse | LPIPS | 1000 | 27.672235 | 0.975373 | 0.034199 | 0.968325 |
| T1_mse | MS-SSIM | 0 | 28.576529 | 0.977049 | 0.034613 | 0.970466 |
| T2_residual_mse | PSNR | 0 | 28.576529 | 0.977049 | 0.034613 | 0.970466 |
| T2_residual_mse | SSIM | 0 | 28.576529 | 0.977049 | 0.034613 | 0.970466 |
| T2_residual_mse | LPIPS | 50000 | 28.276671 | 0.976710 | 0.029485 | 0.970088 |
| T2_residual_mse | MS-SSIM | 10000 | 28.273848 | 0.976804 | 0.031464 | 0.970493 |
| T3_baseline_perceptual | PSNR | 13000 | 28.333791 | 0.976717 | 0.028542 | 0.971204 |
| T3_baseline_perceptual | SSIM | 13000 | 28.333791 | 0.976717 | 0.028542 | 0.971204 |
| T3_baseline_perceptual | LPIPS | 50000 | 28.264767 | 0.976566 | 0.027885 | 0.971197 |
| T3_baseline_perceptual | MS-SSIM | 10000 | 28.172525 | 0.976715 | 0.028559 | 0.971586 |
| T4_residual_perceptual | PSNR | 0 | 28.576529 | 0.977049 | 0.034613 | 0.970466 |
| T4_residual_perceptual | SSIM | 0 | 28.576529 | 0.977049 | 0.034613 | 0.970466 |
| T4_residual_perceptual | LPIPS | 50000 | 28.202765 | 0.976648 | 0.028224 | 0.970049 |
| T4_residual_perceptual | MS-SSIM | 0 | 28.576529 | 0.977049 | 0.034613 | 0.970466 |
| T5_multiscale | PSNR | 0 | 27.615421 | 0.976783 | 0.033296 | 0.971072 |
| T5_multiscale | SSIM | 0 | 27.615421 | 0.976783 | 0.033296 | 0.971072 |
| T5_multiscale | LPIPS | 0 | 27.615421 | 0.976783 | 0.033296 | 0.971072 |
| T5_multiscale | MS-SSIM | 0 | 27.615421 | 0.976783 | 0.033296 | 0.971072 |

Status: completed
Deadline UTC: 2026-10-22T15:13:58.306538+00:00
Current stage: metrics
Current model: T3_baseline_perceptual

[5-second monitor](monitor.html)
[Plan](plan.json)
[Verified rendered winners](rendered_winners.json)

Updated UTC: 2026-10-21T20:43:12.467890+00:00

<!-- final-render-comparisons -->
## GT / metric-winner renders

[Open gallery](../../runs/pidg_metric_tuning/render_comparisons/index.html) · [Absolute raw paths](render_comparisons.json)
