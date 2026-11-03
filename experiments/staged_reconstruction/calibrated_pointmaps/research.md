# Wider temporal MonST3R pairs with known-camera scale

This is a new opt-in, train-only geometry qualification. The original adjacent
pair pilot and I9 remain immutable. No PIDG target, optimizer update or training
admission follows merely from a positive scale estimate.

## Primary sources and motivation

- MonST3R: <https://arxiv.org/html/2410.03825v2>, official implementation
  <https://github.com/Junyi42/monst3r>, frozen commit
  `574cc77ad278bad582f470e5382624e01f8769a7`.
  Both time-specific pointmaps are expressed in the first-camera coordinate
  frame. The paper's training samples temporal strides 1–9, favoring larger
  strides. Our actual available training frames are nonconsecutive; the fixed
  +/-9 and boundary +/-18 frame targets are an adaptation, not its full
  training or global alignment protocol. Actual frame gaps are recorded.
- Depth Anything 3: <https://arxiv.org/html/2511.10647v1>, official repository
  <https://github.com/ByteDance-Seed/Depth-Anything-3>. Its camera-conditioned
  API is relevant, but the paper identifies extending its reasoning to dynamic
  scenes as future work. Different-time observations will not be fused under
  a static-scene assumption here. No DA3 model was installed or evaluated.

The previous MonST3R adjacent-pair pilot failed confidence/static-depth/raw
known-camera projection checks. That result does not reject the whole method.
This experiment changes the temporal pair rule and separately tests whether
the supplied metric camera translations can fix the pair's scale gauge.

## Known-camera scalar estimate

Let Q_j(p) be the predicted second-time point in first-camera coordinates,
indexed by its own second-image pixel p, and b_j(p) its known-K unit ray.
Known relative pose is [R_ji,t_ji] = W2C_j inverse(W2C_i). Solve one positive s:

    b_j(p) cross (R_ji (s Q_j(p)) + t_ji) = 0.

This uses the second-time geometry's own projection and never asserts that a
moving surface occupies the same world position at both times. Cameras stay
fixed. A deterministic median initialization and eight Cauchy IRLS iterations
fit the scalar on one third of eroded observed second-image pixels with raw
confidence >=3; the complementary pixels test projection independently.
Only numerical scalar observability is reported, not material/force
identifiability. Tiny translation and degenerate rays must be rejected.

Two arms share identical frozen official float32 predictions:

1. Wider pairs, original single static-depth scale control.
2. Wider pairs, known-camera scale, with no subsequent fit to S1.

The original 1,000-static-support, held-depth, dynamic confidence/extrapolation,
raw first-view projection and partner-stability gates are retained. Additional
held second-view projection and predicted-parallax checks constrain the camera
arm. No reprojecting or focal warping of pointmaps is allowed to force a gate.
S1 rendered depth is conditioned on the existing reconstruction, not GT.
Neither a held-pixel split nor agreement with S1 certifies unseen surfaces.

## Baseline and reproducibility

The user-selected S3-U final6k SHA256 is
`8c767b7835eee936e49989369d7bc0277d7567d3aedaa7385c06654284b68ad0`.
The existing pinned loader verifies that checkpoint and frozen S1 sources.
RGB is training120 only, black-alpha-composited, LANCZOS resized to512x288 and
normalized to[-1,1], matching the previous pilot. Weight/vendor manifests,
source/protocol hashes and both time-specific pointmaps are retained.

Ten synthetic geometry tests precede real inference: metric scale recovery,
independent second-time shapes, raw scale gauge, robust bad-pixel handling,
pure rotation and baseline-parallel degeneracy, wrong intrinsics, held-out bad
geometry, insufficient/nonfinite inputs and invalid camera transforms.

No original renderer/trainer/config/physical branch is edited. Thus this pilot
does not add a physics milestone or require a full-training smoke before
geometry qualifies. Any later PIDG integration requires opt-in tests, original
warmup admission and actual native cross-view image review.
