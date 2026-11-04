# J4: Rubber Duck matched joint reconstruction pilot

The user explicitly requested detection and joint improvement on Rubber Duck.
This is a bounded real-data pilot, not automatic promotion of J3: J3 passed
the original numerical synthetic gates but failed manual support cleanup.
Its RGB spill fell about 40%, while outside-GT high-alpha support fell only
2.4–2.6%. All original synthetic conclusions and thresholds remain unchanged.

Use the pinned complete S3-U final6k parent and its own static bank. Its static
parameters exactly match S1 final12k, so inherit that complete static Adam,
including step12000, without a reset. Inherit the complete S3-U dynamic Adam,
RNG, remaining camera stack and physics graph cache. The legacy parent used
conditional color masks; this experiment cannot erase that lineage.

Two arms receive 2000 new updates each with identical sampling and full-image
RGB supervision: frozen static control, and static opacity/SH plus the J3
fixed-weight 0.1 parent expected-depth self-reference. Freeze static geometry
and velocity/stress parameters. The existing velocity/stress Adam and graphs
are preserved exactly, but no physics residual is evaluated or optimized:
this is a joint reconstruction pilot, not original physics continuation.
The unified dynamic population remains unknown-count with unknown identities.

The RGB target uses the dataset's original RGBA composited onto black, following
the native image contract. Alpha is not a training target or selector. No
static/dynamic color masks, new depth/normal estimates, GT, object counts or
semantic labels enter the new objective. Teacher depth comes only from the
frozen parent static render and can preserve false support. No topology change,
deletion, weight/threshold sweep, new initialization or static-bank graft.

Qualification precedes formal updates: tests, native initial RGB/alpha/depth
parity, isolated opacity gradient, 20-step smoke and two independent processes
restoring complete model, both Adams/bindings, RNG, sampler and physics graphs
exactly. Use one saved native gradient for exact next-Adam replay; preserve
historical CUDA full40 failures. A fresh native gradient is checked at the
previously authorized tolerance, not required to yield deterministic trajectories.

Evaluate all 30 validation images at native1600 with full-RGB PSNR/SSIM and
loss; do not read test RGB for candidate selection. Inspect parent/control/
candidate static, dynamic and joint RGB, alpha and normalized expected depth
at fixed cameras86/188, times2/100/200, plus observed training-trajectory views.
Render fixed full200-frame RGB comparisons, H264/yuv420p/faststart and decode
every frame. Export static/canonical/time-deformed PLY matched to checkpoints.
Fixed camera cross-time images have no independent GT. No mask-derived
geometry or physical identity conclusion is available from this dataset.

Report full-image validation and self-reference proxies separately from visual
cleanup/completeness. Validation PSNR may drop at most0.3dB vs matched control;
manual review must find fewer detached/halo supports in alpha/depth AND no
visible duck/structure loss across both fixed sequences. No automatic promotion
by PSNR or RGB darkening. If not met, finish as not promoted; retain artifacts
and use the real evidence to choose a later single mechanism, not a sweep.

All code opt-in; no original PIDG/frozen sources/history or paused jobs change.
Stop requests atomically save both optimizers/RNG/sampler/graphs and release
GPU. No automatic resume of a user-stopped pilot.
