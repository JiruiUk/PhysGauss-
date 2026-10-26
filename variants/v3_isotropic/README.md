# V3 + isotropic regularization

This independent fork adds the previously tested Gaussian axis-isotropy loss to
the completed time-disjoint V3 model. It does not modify earlier source trees,
weights, data splits, reports or renders.

Both arms warm-start from the same validation-PSNR-selected V3 checkpoint in
`runs/pidg_kinematic_resplit/V3_frame_time/rubber_duck`. They each receive 50k
additional refinement steps with fresh optimizers and the same V3 learning-rate
schedule. The first 1000 additional steps are shared. At that boundary the full
checkpoint, optimizer, RNG and camera-sampling state are copied identically;
only `isotropic_weight` changes in the receipted metric sidecar.

| Arm | Iso weight | First active iso step | Kinematic weight |
|---|---:|---:|---:|
| V3_continue_control | 0 | never | 0 |
| V3_iso_1e3 | 0.001 | 1001 | 0 |

The unchanged normalization in `variants/isotropic_tuning/model.py` operates on
the magnitudes of rendered axes `abs(exp(canonical_log_scale) + d_scaling)`:

`L_iso = mean_i mean_axis abs(s_i / mean_axis(s_i) - 1)`.

It is a dimensionless, soft shape prior, not a material law. Each Gaussian is
normalized by its own mean axis length: different Gaussians can retain different
overall sizes. Signed offsets are handled through magnitudes, matching squared
covariance axes. The existing Python-covariance fallback uses canonical scales,
as in the prior implementation. The evaluated configuration uses the normal
rendered-covariance path. Weight zero or disabled tuning returns before touching
Gaussian state or consuming RNG. No extra network parameters are introduced.

The data stay at 120 train / 30 validation / the same 50 official test images,
with `(frame_id-1)/199` time coordinates. The uniform-cadence assumption remains
unverified in physical seconds. Training loads only train/validation RGB. All
four checkpoint objectives use validation; both arms' selections are locked
before final test rendering. Final50k is also exported for equal-step comparison.

Compare iso against the **continued control** to assess its effect. A gain versus
the previous V3 alone also includes 50k extra training. CUDA nondeterminism,
single-scene/single-seed evidence, and historical official-test tuning exposure
remain limitations.

Run GPU preflight, then the production queue:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File variants/v3_isotropic/launch.ps1 -Smoke -Hours 24
powershell -NoProfile -ExecutionPolicy Bypass -File variants/v3_isotropic/launch.ps1 -Hours 24
```

Outputs:

- Frozen plan, preflight and results: `experiments/v3_isotropic`
- 5-second validation charts/current iteration: `experiments/v3_isotropic/monitor.html`
- Separate checkpoint/optimizer state: `runs/pidg_v3_isotropic`
- Raw final renders/GT: each run's `rubber_duck/test/ours_<tag>/{renders,gt}`
- Pixel-verified GT/render pairs: `runs/pidg_v3_isotropic/render_comparisons`

The queue supports full-state resume. To pause safely, create
`runs/pidg_v3_isotropic/stop_queue.request`; training saves state before exiting.
