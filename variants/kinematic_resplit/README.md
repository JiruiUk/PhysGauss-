# V3 / V4: time-disjoint retraining

The historical experiment has no independent validation metadata: the same 50
official test images select all four best snapshots. Original training/test RGB
membership is disjoint. Equal original metadata times at 0 and 1 are a separate
clock issue, not evidence of duplicate RGB.

This independent experiment preserves those 50 test images and poses. It removes
30 temporally distributed frames from the original 150 training images for
validation: select index 2 in every block of five original training frames.
The resulting 120/30/50 image sets and their globally assigned times are pairwise
disjoint. All times use `(original_frame_id - 1)/199`, under the explicitly
unverified uniform-cadence assumption. This is interpolation, not extrapolation.

## Training and selection

1. Train P0 for 50k using the original PIDG optimization recipe, newly generated
   independent random geometry, the 120 training images and 30 validation images.
   The renderer uses the shared metadata times; P0 retains legacy physics time
   division. P0's best validation-PSNR snapshot initializes both refinements.
2. Start V3 with the historical T3/V3 optimization and image objectives. Save the
   complete state at step 1000, before kinematic consistency starts at 1010.
3. Copy that full state byte-for-byte into V4, including optimizer/RNG/camera
   state. Only the declared kinematic loss weight changes from 0 to 0.001 in the
   metric-sidecar config; record this transition in `branch.json`.
4. Complete each refinement to 50k. Use only the 30 validation frames for the
   four metric-specific best checkpoints. Keep final50k independently.
5. After both refinements complete, lock all selections in `selection_lock.json`.
   Evaluate only the five predeclared snapshots on the 50 test images. Export
   full-resolution GT/render pairs and comparisons against prior final50k scores.

The training loader's legacy `test` slot contains **validation**, not test images.
The separate `data/derived/kinematic_resplit/training` view has no final-test RGB.
No flow/depth files or released point cloud are imported. Training masks are
copied only for training frames. The original dataset and all old study files
remain unchanged. Final evaluation uses the `final_test` loader view.

New V3/V4 each represent 100k lineage iterations (50k P0 + 50k refinement).
GPU work is 149k unique iterations because the first 1000 refinement steps are
shared. CUDA kernels may still diverge after the branch; this is a single-seed
comparison, not proof of a statistically robust causal effect.

## Limitations

The official test set was exposed in previous studies. This protocol prevents
further test-based selection in the new run; it cannot create a historically
untouched test set. Before/after scores also include changed training count,
new random initialization and a newly trained common parent. Only the new
V4-minus-V3 comparison controls those changes. Physical velocity accuracy still
requires independent motion/velocity truth.

## Run

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File variants/kinematic_resplit/launch.ps1 -Smoke
powershell -NoProfile -ExecutionPolicy Bypass -File variants/kinematic_resplit/launch.ps1 -Hours 24
```

The second command requires a successful GPU preflight. It resumes existing full
checkpoints. Stop safely by creating `runs/pidg_kinematic_resplit/stop_queue.request`.
The current training state is saved before the queue exits.

Progress and validation charts refresh every 5 seconds in
`experiments/kinematic_resplit/monitor.html`; counters are actual training
iterations, separate from image-evaluation progress. Results and the eventual
before/after test table are in `experiments/kinematic_resplit/results.md`.
Pairs will be written to `runs/pidg_kinematic_resplit/render_comparisons`.
