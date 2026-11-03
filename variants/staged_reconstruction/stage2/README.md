# S2: independent per-object canonical reconstruction

This optional branch fixes the accepted S1 static bank and initializes four
immutable dynamic Gaussian groups from the clear first training image. Each
group uses an independent original PIDG `DeformModel` (unmodified class and
forward architecture). Canonical coordinates are local to an initial world
center. Gaussian groups are combined before the unchanged renderer performs
depth sorting/compositing. S1 and historical training code remain frozen.

The initialization uses a three-layer camera-depth slab with a depth prior
derived from the static cloud. This is uncertain monocular initialization,
not surface/depth GT. The point count is 15,507 dynamic plus 4,126 static;
four bounded hash grids are an explicit new architecture, not a matched C1
comparison. S2 uses only original RGB L1/SSIM, with all physics, flow, tracks,
contact, material/force changes and densification disabled.

Training-only raw bidirectional image flow supplies a separate conservative
identity ledger. All four seed identities connect through original frame42.
Seeds1/2 lose support at43, seed0 at52, seed3 at166. Loss of an early identity
is sticky. Late component numbers cannot reinstate it. These are adjacent
visible image links with conditional error, not physical/material identity
GT or an occlusion bridge. Structural model IDs never certify physical re-ID.

Run in the repository's CUDA environment:

```powershell
. .\reproduction\scripts\enable_cuda.ps1
envs/pidg-cu128/python.exe -m variants.staged_reconstruction.stage2.prepare --enable-object-stage
envs/pidg-cu128/python.exe -m variants.staged_reconstruction.stage2.train --enable-object-stage --preflight-only
envs/pidg-cu128/python.exe -m variants.staged_reconstruction.stage2.train --enable-object-stage --smoke --stop-after 40
envs/pidg-cu128/python.exe -m variants.staged_reconstruction.stage2.train --enable-object-stage --smoke --stop-after 80
envs/pidg-cu128/python.exe -m variants.staged_reconstruction.stage2.freeze
envs/pidg-cu128/python.exe -m variants.staged_reconstruction.stage2.train --enable-object-stage
```

Formal state: `runs/pidg_staged_reconstruction/S2_objects`. The trainer saves
all model tensors, Adam state, RNG states and camera stack. A native process
lock prevents duplicate trainers. `stop.request` at the shared runs directory
requests a complete-state stop. Resume never changes protocol or resets LR.
Native val/test and per-object/fixed-camera review are still required after
formal training; finite loss or opaque static structures behind missing ducks
cannot satisfy dynamic reconstruction admission. Dynamic alpha is measured
using a separate dynamic-only render, not full-composite opacity.
