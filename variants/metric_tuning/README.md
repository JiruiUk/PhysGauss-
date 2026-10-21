# PIDG: bounded search for PSNR / SSIM / LPIPS / MS-SSIM

User authorization: approximately **24 hours**, compare all four metrics and keep
different optimal models. Rubber Duck remains the only scene. Training uses the
original 150 training views. Every metric uses the unchanged 50 official test
views at 1600x900. Test RGB never enters a gradient loss. As in the original
experiments, test metrics do select checkpoints and configurations, so these
results are tuned to this test set rather than an untouched test evaluation.

Original PIDG code in `third_party/PIDG`, historical models, existing experiments
and PDF reports are preserved. `code/` is an independent, source-frozen copy with
small opt-in hooks. With `metric_residual_width=0` and no enabled metric config,
the original model and loss path remain active. No new F / NH / material / force
module is enabled. Physics-driven PIDG losses and original velocity/stress fields
are retained during refinement.

## Search space

| Candidate | Frozen parent best | Refinement |
|---|---|---|
| T0_continue | W512_D0 | Original L1 + SSIM objective; conservative learning rates |
| T1_mse | W512_D0 | MSE + SSIM; weaker grid regularization |
| T2_residual_mse | W512_D0 | Same MSE objective; 512→128→512 residual adapter |
| T3_baseline_perceptual | W256_D0 | L1 + SSIM + VGG LPIPS |
| T4_residual_perceptual | W512_D0 | Charbonnier + SSIM + VGG LPIPS; residual adapter |
| T5_multiscale | W512_D2 | L1 + MS-SSIM; weaker grid regularization |

The residual adapter is added after feature fusion and existing hidden layers,
before the three output heads. Its last affine layer starts at zero, preserving
the parent function initially. Hash grids and output dimensions are unchanged.
LPIPS training supervision is evaluated on training images resized to 800x450
every four updates; its weight is multiplied by four on those updates. Final
metrics always use full 1600x900 images and the original VGG LPIPS convention.

This is **warm-start refinement with extra compute**, not a matched-budget
from-scratch architecture ablation. Every new run loads its parent's best
Gaussian/deformation/material inference weights with a new optimizer. Thereafter
screening and longer refinement use full optimizer/RNG/camera-order checkpoints,
without resetting or restarting. Original source runs are read-only.

Each candidate is screened for 1,000 extra iterations with a 50k target schedule.
Up to four candidates are selected across the four objectives for continuation
to 50k extra iterations. PSNR, SSIM, LPIPS and MS-SSIM each keep a distinct best
snapshot, including the parent-equivalent zero-step snapshot. The coordinator
retains the observed Pareto set and the old reference models, so failure to improve
an objective is reported rather than relabelled as improvement.

The 24-hour wall-clock budget includes preflight, screening, long training and
exports. Training stops safely 40 minutes before the deadline to reserve time
for winner rendering and original-metric verification. A deadline stop saves a
full checkpoint; a saved partial run is not described as a completed 50k run.
Budget bounds the local search, not proof of a global optimum.

## Validation and outputs

CPU tests cover exact flags-off forward/backward/RNG parity, zero-initialized
residual identity and learning, finite gradients for all pixel losses, PNG
quantization, checksum-bound checkpoint metadata, multiobjective selection and
the no-grad-only official evaluator.

Before screening, the coordinator runs an original flags-off smoke train/render/
metrics test, then a residual/perceptual smoke test with verified full-state
continuation from step20 to step40. All four saved snapshots are rendered and
checked against the streamed evaluator. Step0 must reproduce each parent's
existing PNG metrics within a small numeric tolerance.

- Plan: `experiments/metric_tuning/plan.json`
- Results: `experiments/metric_tuning/results.md` and `results.json`
- Monitor (refreshes every5 seconds): `experiments/metric_tuning/monitor.html`
- Preflight: `experiments/metric_tuning/preflight.json`
- Export manifest: `experiments/metric_tuning/rendered_winners.json`
- Final GT / four-winner gallery: `runs/pidg_metric_tuning/render_comparisons/index.html`
- New models/logs/checkpoints: `runs/pidg_metric_tuning/<candidate>/rubber_duck/`
- Snapshot selection: `metric_best.json`; history: `metric_history.jsonl`
- Rendered images: `test/ours_<snapshot>/{renders,gt}/`

Start: `./variants/metric_tuning/start.ps1 -Hours 24`

Pause safely by creating `runs/pidg_metric_tuning/stop_queue.request`. The worker
requests a complete saved state and waits for process exit, freeing GPU memory.
Resume with `start.ps1 -Resume`; the original deadline is retained. Resuming never
reinitializes an existing full checkpoint.

Design references: [original 3DGS](https://arxiv.org/abs/2308.04079) and
[identity residual mappings](https://arxiv.org/abs/1603.05027). The specific
adapter/loss combinations above are local engineering experiments; improvement
is to be measured, not assumed from those papers.
