# Staged static-anchored PIDG reconstruction

Explicit opt-in S1–S6 branch. Original PIDG and historical variants are unchanged.
S1 uses the existing C1 Gaussian rasterizer without a deformation network, velocity
network or dynamic optimizer. A separate static Gaussian bank starts from a
training-only visual hull; it is not an existing dynamic checkpoint with fixed warps.

S1 camera contract is **metadata CV**, including the metadata horizontal FoV and
standard Blender-to-CV transformation. This is behind the new stage command;
the original PIDG loader remains untouched. Historical legacy-camera weights
cannot initialize this branch or provide a directly matched metric comparison.
The shared track audit is conditional evidence, not unbiased calibration GT.

Only eroded visible blue candidate pixels and known transparent background are
supervised. Yellow dynamic regions, their margins and other uncertain foreground
are unknown. Multitime hull support filters the initial candidate points. Static
geometry remains a reconstruction hypothesis, not material labels or dense GT.
The final initialization has4126 points, below the14624 static-point cap.

```powershell
. .\reproduction\scripts\enable_cuda.ps1
envs/pidg-cu128/python.exe -m pytest tests/test_staged_static.py tests/test_static_pool.py -q
envs/pidg-cu128/python.exe -m variants.staged_reconstruction.camera_audit
envs/pidg-cu128/python.exe -m variants.staged_reconstruction.prepare_data
envs/pidg-cu128/python.exe -m variants.staged_reconstruction.train_static --enable-static-stage --preflight-only
envs/pidg-cu128/python.exe -m variants.staged_reconstruction.freeze
envs/pidg-cu128/python.exe -u -m variants.staged_reconstruction.train_static --enable-static-stage
```

Complete optimizer/RNG/camera-stack checkpoints are saved every1000 updates.
Create `runs/pidg_staged_reconstruction/stop.request` to save and stop at the next
iteration. Do not kill formal training; remove a stop request only on user resume.
A Windows file lock prevents duplicate training. Final evaluation uses native
1600×900 validation/test images and candidate static region metrics, not full
scene PSNR against an image containing unrendered ducks. The final12000 state
is preselected; validation does not choose a checkpoint.

S1 completion requires native visual review and evidence gates; opacity collapse
or structural time invariance alone is not success. S2–S6 remain the user task
after S1. See `experiments/staged_reconstruction_plan.md` and
`experiments/staged_reconstruction/S1_protocol.json`.
