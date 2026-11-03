# R0: partial static Gaussian pool

Independent, default-off C1 fork. The original PIDG and all earlier experiments
are retained. A fixed per-row rendering transform replaces the query-time warp
only for14,624 previously accepted static rows. Canonical geometry, opacity and
SH remain learnable; no points, free forces or material parameters are added.
Unknown rows remain dynamic. Known-static rows are omitted from dynamic CMR.

This is a logical static pool within the existing Gaussian storage, not learned
segmentation. Its source row map, particle embedding map and reference arrays
are SHA-bound; topology changes fail explicitly. Inference uses the same renderer
hook as training and requires the bound per-run config. The captured geometry
is a hypothesis, not physical ground truth.

Controls: two identical complete15k states (including optimizer/RNG/camera
stack), each trained35000 updates to final50k. Only `hard_static_pool` differs.
No previous final model initializes either arm. All new soft transfer losses
remain off. Original image/physics coefficients and LR schedule are retained.

```powershell
. .\reproduction\scripts\enable_cuda.ps1
envs/pidg-cu128/python.exe -m pytest tests/test_static_pool.py -q -p no:cacheprovider
envs/pidg-cu128/python.exe -m variants.static_pool.run --preflight-only
envs/pidg-cu128/python.exe -m variants.static_pool.pipeline
```

The pipeline completes matched training, final-state metrics, static support,
native fixed-view scans, full video decoding and a numeric assessment. It then
waits for manual visual review and an evidence-based plan revision; it does not
start another model. Resume preserves saved full states, with no LR reset.

Plan: `experiments/static_pool/plan.json`. Live state:
`runs/pidg_static_pool/controller.json`, `pipeline.json`, `active.json`.
Pause by creating `runs/pidg_static_pool/stop_queue.request`; the controller
requests an end-of-iteration full checkpoint and stops the queue. Do not suspend
a process retaining GPU memory. Remove a pause request only on user resume.

Fixed-view morphology and blue/yellow regions are descriptive proxies.
Image-track observations were held from triangulation only; they are training
images for the model and cannot be advertised as held-out model geometry GT.
