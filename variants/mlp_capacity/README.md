# Baseline PIDG MLP capacity ablation

This experiment uses `third_party/PIDG` and the original Rubber Duck trainer.
F, Neo-Hookean, learnable material and structured-force extensions are disabled.
No baseline runtime source is edited. Per-run `model_config.py` snapshots change
only the existing deformation MLP's width/depth; hash grids, velocity/stress
network, losses, learning-rate rules and densification settings remain fixed.

| Configuration | Width | Extra hidden layers | Deformation MLP parameters |
|---|---:|---:|---:|
| W256_D0 | 256 | 0 | 61,454 |
| W512_D0 | 512 | 0 | 122,894 |
| W256_D2 | 256 | 2 | 193,038 |
| W512_D2 | 512 | 2 | 648,206 |

The completed original 50k W256_D0 baseline is reused. The other three models
are trained from scratch for 50k iterations, width1600, seed0, on Rubber Duck.
All four architectures first undergo gradient/configuration checks and separate
300-step width400 smoke train/render/metrics runs. Smoke scores are engineering
checks and must not be compared against 50k results.

Start: `powershell -NoProfile -ExecutionPolicy Bypass -File variants/mlp_capacity/start.ps1`

Results and the five-second auto-refresh monitor:
`experiments/mlp_capacity/results.md` and `experiments/mlp_capacity/monitor.html`.
Runs, checkpoints and evaluation images: `runs/pidg_mlp_capacity/`.

The primary evaluation retains original best-test-PSNR selection for compatibility.
The report also includes final-50k logged test PSNR where present. The historical
baseline lacks complete source hashes and this is a single-seed exploratory
comparison, so observed differences are not statistical or causal proof.

Pause safely: create `runs/pidg_mlp_capacity/stop_queue.request`. The worker
forwards a full-state save/exit request, releases the trainer and stops the queue.
For a later user-authorized resume, archive the request and use `start.ps1 -Resume`.
Completed runs are preserved; incomplete runs require verified full checkpoints.
