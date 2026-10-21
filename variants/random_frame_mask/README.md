# Random missing training frames: original PIDG baseline

This is a separate dependent queue. It waits for the MLP capacity controller to
finish, for its processes to exit, and for all 50-view GT/render sets and metrics
to be verified. A failed or paused MLP queue does not trigger this experiment.
The currently running MLP source, configurations and runs are not modified.

Only the original baseline network is used: deformation width256/depth0, original
hash grids and velocity/stress network, with F/NH/material/force extensions off.

| Training frames removed | Retained / 150 | Dropped-frame GT views |
|---|---:|---:|
| 0% (completed baseline reused) | 150 | N/A |
| 10% | 135 | 15 |
| 30% | 105 | 45 |
| 50% | 75 | 75 |
| 70% | 45 | 105 |
| 90% | 15 | 135 |

Mask seed0 creates one random permutation; each higher dropout is a superset of
the earlier dropped frames. Original frame order, timestamps, poses, files and
canonical point cloud are preserved. New manifest overlays use directory links;
no original data are deleted, overwritten or blacked out.

After the dependency, a 300-step width400 smoke test uses the most severe (90%)
case, including both evaluation paths. Each of the five ratios then trains from
scratch for 50k at width1600, training seed0, with original baseline losses,
learning rates, warmup and densification. The fixed iteration budget means more
reuses of retained images at higher dropout.

Each model is rendered/evaluated on (1) the unchanged official 50 test views and
(2) the dropped training images, which are never used for fitting or checkpoint
selection. Selection retains the historical best-official-test-PSNR protocol.
Use the fixed official test set to compare ratios; dropped-frame evaluation sets
are nested but have different membership. This is not leakage-free validation.

PIDG's original momentum query is `fid / train_count`. This native denominator
changes when training frames are removed; no physics normalization fix is silently
introduced. Observed effects include this native response to reduced training
data. The study is single-scene/single-seed, not statistical evidence of robustness.

Start queued watcher: `powershell -NoProfile -ExecutionPolicy Bypass -File variants/random_frame_mask/start.ps1`

Results: `experiments/random_frame_mask/results.md`, `.json`, `.csv`.
Five-second monitor: `experiments/random_frame_mask/monitor.html`.
Raw split overlays: `data/pidg_random_frame_mask/rubber_duck/`.
Training outputs: `runs/pidg_random_frame_mask/long_dropXX_seed0/rubber_duck/`.
Official GT/render PNGs: `test/ours_best/{gt,renders}/`.
Dropped-frame GT/render PNGs: `masked_holdout/test/ours_best/{gt,renders}/`.

Create `runs/pidg_random_frame_mask/stop_queue.request` for a safe save/stop. A
future user-authorized resume must acknowledge this request and use `-Resume`;
existing full checkpoints and completed jobs are preserved.

## Post-training retained / missing / full-data evaluation

The original primary PSNR covers **only the 50 official test images**. The
dropped-frame column covers the removed training images; neither includes
retained training images. To evaluate both parts at the same frozen best model:

```powershell
./variants/random_frame_mask/evaluate_splits.ps1
./envs/pidg-cu128/python.exe variants/random_frame_mask/validate_splits.py
```

This renders the retained frames for all six completed models, including the
0% control. It reuses the verified official and dropped-frame per-image metrics,
without training or choosing a new checkpoint. New renders and GT are saved at
each run's `retained_evaluation/test/ours_best/{renders,gt}/`. Individual source
IDs, groups, image paths and metrics for all 200 views are recorded in
`retained_evaluation/all_frame_metrics.json`.

All four metrics use the original PIDG functions and VGG LPIPS settings, streamed
one image pair at a time. Scores are arithmetic means of per-image values.
In particular PSNR averages **per-image dB**, rather than converting a pooled MSE
to dB. Union scores weight subset means by their actual frame counts. The
original training set contains retained + removed = 150 frames; all data adds
the fixed 50 official test frames = 200 frames, matching `transforms_all.json`.
The full-data score includes seen training images and is a reconstruction score.
Keep official-test scores separate when discussing generalization.

An evaluation-only directory alias gives the retained test images distinct
loader IDs, because PIDG otherwise overwrites camera metadata when train/test
file paths overlap. The alias points to the same original RGB directory; poses,
times, source IDs used for reporting and frozen model files are preserved.

Summaries: `experiments/random_frame_mask/split_metrics.{md,json,csv}`. Absolute
render and GT directories are included. The independent validator checks all
1,200 model/view pairs against the original GT after official compositing and
resizing, recomputes PSNR on CPU, checks every union, and verifies model/input
hashes. It saves `split_metrics_validation.json`. Split metrics remain separate
artifacts. The Chinese and English PDF reports compare only the fixed 50-view
official test set, including GT/render comparisons and absolute image paths.
Rebuild both with `scripts/build_pidg_baseline_ablation_report.py`.
