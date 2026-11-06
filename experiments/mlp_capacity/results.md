# Baseline PIDG: deformation MLP capacity ablation

Rubber Duck; 50k; width1600; seed0. F/NH/material/force extensions are OFF.
W256_D0 reuses the completed original baseline. Smoke results are not long-run results.

| Variant | MLP params | Status | Iteration | Best PSNR | Delta | Final-50k logged PSNR |
|---|---:|---|---:|---:|---:|---:|
| W256_D0 | 61,454 | completed (reused) | 50000 | 27.256781 | 0.000000 | 27.456247 |
| W512_D0 | 122,894 | completed | 50000 | 28.576529 | 1.319748 | 28.416312 |
| W256_D2 | 193,038 | completed | 50000 | 28.109819 | 0.853039 | 28.157720 |
| W512_D2 | 648,206 | completed | 50000 | 27.615419 | 0.358639 | 27.591804 |

No conclusion about a PSNR gain is made before the corresponding long run finishes.
Single-seed differences are descriptive, not statistically established. Best snapshots use test-based selection.

[Five-second monitor](monitor.html)

Runs: `runs/pidg_mlp_capacity/`. Configurations: `variants/mlp_capacity/configs/`.

Updated UTC: 2026-10-20T17:14:38.560826+00:00
