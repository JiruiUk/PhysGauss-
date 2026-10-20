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
