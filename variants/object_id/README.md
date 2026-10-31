# C1 self-supervised object-motion grouping experiment

This is a **MovingParts/DynaVol-inspired custom ablation**, not a reproduction
of either paper, AdaSlot, or QASA. It preserves the frozen repaired C1 runtime
and adds an independent opt-in object-motion branch. Existing studies remain
unchanged. No material segmentation, per-kernel stiffness or free force is added.

## Arms and matched protocol

- O0_C1_control: repaired C1, object branch disabled.
- O1_frozen_ID: same added architecture, geometrically initialized identity
  field frozen; shared motion is trainable. This is not a GT/pseudo-ID target.
- O2_selfsup_ID: same initialization and motion branch, identity field trainable.

All three branch the identical full C1 random-initialization **3000-step**
checkpoint (`6da523ea247b4068ec9bd6a7c8f83dacb7506c698daaeecc9eb0ecd93ea1ebce`).
Each finishes at **50000 total**, not an additional50000 after a converged C1.
The full Gaussian/network/optimizer/RNG/camera-stack state is copied. New module
initialization uses an isolated RNG context and a fixed seed1729.

Same120 train /30 validation /50 final-test frames, global `(frame_id-1)/199`
clock, C1 image/physics/density recipe and seed0. Validation selects snapshots;
all three selections are locked before final-test RGB is loaded for exports.
The historical test has earlier tuning exposure: this is not a pristine benchmark.

**Instance-label-free does not mean a fully RGB-only baseline.** C1's existing
total-foreground alpha and foreground RGB losses remain in all three arms.
No instance IDs/masks, SAM, tracker, offline cluster pseudo-targets or material
labels enter the new module. C1 alpha may indirectly influence the learned
motion/assignment through rendering, and must be disclosed as retained supervision.

## Model and actual coupling

The identity MLP maps encoded canonical XYZ to soft assignments p_ik, initialized
with geometric RBF affinities to deterministic farthest-point anchors. The time
MLP predicts one quaternion/translation per candidate. Candidate k transforms
a canonical center X as `R_k(t)*(X-anchor_k)+anchor_k+translation_k(t)`.
Weighted candidate transforms produce `x_group`. No per-frame RGB or identity
teacher is input to the grouping field.

The render center is `(1-blend)*x_C1 + blend*x_group`. The original per-kernel
quaternion is retained and its translation adjusted to realize this center
through the existing renderer. Gaussian covariance rotation/scale and the PIDG
velocity/predicted-stress path remain C1's. The branch is a reconstruction prior,
not a hard physical collision or rigid-solid model.

Blend ramps from0 after3k to0.25 over5k steps; the retained C1 path supplies
nonrigid residual capacity. RGB reconstruction therefore reaches identity and
shared motion parameters. Auxiliary terms are trajectory-residual Huber1e-3,
assignment entropy5e-5, and sqrt-occupancy complexity1e-6, ramped on the same
schedule. No equal-size/balanced-slot objective forces a chosen object count.
All new parameters use Adam2e-4 and gradient norm clipping1.

**16 is a candidate capacity ceiling, not an actual object count.** Candidate
mass below0.001 can be pruned after10k every2k, never deleting the final active
candidate. Occupancy >1% and max assignment confidence are diagnostics, not
instance accuracy. This is not capacity-free discovery; mergers, part splits,
all-one collapse or unused candidates must be reported if observed.

## Baseline and state guarantees

`prepare.py` copies C1 code into `variants/object_id/code`, changes only the
Tuning import and opt-in deformation/state hooks. With object_id_enabled=false,
no object model/optimizer is created and the warp object is returned unchanged.
Original C1 sources are checksum-verified and not modified.

Membership is a coordinate field, so densification/pruning does not require
resizing a per-point ID parameter bank. Children at parent coordinates receive
the same assignment. Selected snapshots contain `deform/.../object_id.pt`.
Full training checkpoints have SHA-bound `.object.pt` model/optimizer sidecars,
and missing state is allowed only at the explicitly receipted3k branch.

## Checks and outputs

```powershell
. ./reproduction/scripts/enable_cuda.ps1 -ProjectRoot (Get-Location).Path
& ./envs/pidg-cu128/python.exe -m pytest tests/test_object_id.py tests/test_object_id_runtime.py -q
& ./variants/object_id/launch.ps1 -Smoke -Hours 2
& ./variants/object_id/launch.ps1 -Hours 24
```

CPU tests cover disabled identity, zero initial grouped transform, reconstruction
gradients, frozen identity, checkpoint parity and point-count changes. Explicit
CUDA tests compare the actual C1 prefix deformation tensor-for-tensor and verify
rendered RGB gradients reach identity. Preflight trains3x100-step arms, exercises
resume from70, and checks30-view validation PNG metrics against in-training
metrics. Test images remain locked during these checks.

The queue refreshes `experiments/object_id/monitor.html` every5 seconds with
current iterations and validation curves. Final exports include individual RGB,
GT pairs, GT/O0/O1/O2 panels, and O1/O2 identity maps for best/final snapshots.
Shuffling assignment columns while holding motion slots fixed tests branch use;
it does not establish semantic identity correctness. An automatic paired final50k
assessment reports image changes without claiming GT object-ID accuracy.

Interpret O2 minus O0 as the effect of the whole added package. O2 minus O1
tests learnable membership against the same initialized frozen membership;
it does not by itself prove semantic object discovery or separate that claim
from added trainable capacity. Compare all four image metrics, foreground PSNR,
mask IoU and coverage diagnostics, then inspect grouping maps for mergers,
fragmentation and collapse. A shuffled-ID response establishes use of the
branch, not correctness of its groups. No penetration penalty is included in
this first grouping ablation.

Relevant research:
[MovingParts](https://arxiv.org/pdf/2303.05703),
[DynaVol](https://arxiv.org/pdf/2305.00393).
