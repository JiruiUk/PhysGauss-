# Literature-guided S3-U motion integrity experiments

Date: 2026-11-03. Status: implementation and parent diagnostics; no success claim.

The user requires continued model improvement until objects remain whole in
other views. The comparison parent remains S3-U **final6k**, after 16k warmup,
SHA256 `8c767b7835eee936e49989369d7bc0277d7567d3aedaa7385c06654284b68ad0`.
The original surface-trajectory evidence gate failed and stays failed. This is
an independent reconstruction-prior experiment, not admission of those tracks.

## Primary literature and applicability

1. [Dynamic 3D Gaussians, 3DV 2024](https://dynamic3dgaussians.github.io/paper.pdf):
   persistent Gaussian properties and local rigidity/isometry constrain motion.
   Its experiments use synchronized multi-camera images and depth initialization;
   that supervision is stronger than our one-view-per-time data. We borrow a
   rotation-invariant distance prior, not its benchmark guarantees or full method.
2. [SC-GS, CVPR 2024](https://yihua7.github.io/SC-GS-web/): sparse control points
   interpolate compact SE(3) motion, with ARAP regularization. This motivates a
   later restricted-motion representation if a loss on the present field fails.
   No control representation is implemented in the first arm.
3. [MoSca, CVPR 2025; preprint v2](https://arxiv.org/html/2405.17421v2):
   trajectory proximity uses the maximum separation over time, rather than only
   proximity at a contact instant. Sparse motion scaffolds fuse observations.
   Its foundation-model depth/tracks and dual-quaternion skinning are not present
   here. We borrow only maximum-trajectory neighborhood selection in this arm.

Sources were read online on 2026-11-03. CVF PDF requests for SC-GS/MoSca returned
403; the author project page and arXiv text above were accessible. These are
method adaptations, not paper reproductions.

## First falsifiable intervention

Build an immutable graph from 40 evenly distributed **training** times of the
frozen parent. Candidate edges are unions of spatial 12-nearest neighbors.
Rank by maximum 3D separation over all sampled times, retain mutual 12-nearest
edges within 0.08 scene units. No object count, IDs or learned segmentation.
Neither model positions nor edges are independent geometry/correspondence GT.
Model errors and cross-object links remain possible despite the contact filter.

For each edge, freeze the median parent length as a conditional rest length.
Penalize robust absolute log strain outside a symmetric `log(1.2)` band. Fixed
weights/normalization prevent opacity or learned confidence from reducing the
loss by discarding supervision. This permits moderate local elastic change.
It does not impose a solid constitutive law or infer material stiffness/forces.

I0 and I1 both start from the full same final6k state, retaining Adam moments,
RNG, camera stack and original physics. Both take 2,000 actual optimizer steps
at native width1600. I1 adds weight0.03, linearly ramped over400 updates; I0
disables the added loss. Edge/time samples use deterministic schedules, so the
new loss consumes no RNG needed by inherited camera/physics sampling. Frozen
S1, renderer, dataset, camera contract and topology are unchanged.

## Verification and acceptance

Before formal runs: behavioral unit tests, real intended gradients/Adam update,
100-update training smoke, flags-off parity, and independent40-to80 full-state
recovery. Record deviations/retries, protect source/checkpoint hashes and
preserve all unrelated concurrent work. Commit implementation and results as
separate milestones. This is not a new M1–M10 physics module.

Predeclare six fixed training cameras2/43/86/146/188/200. Screen nine times
2/27/43/62/86/108/146/172/200; eventual acceptance requires all200 times from
all six cameras plus image review. Measure dynamic-alone alpha islands, thin
support, enclosed holes and area, alongside full RGB and original-camera GT.
Do not count legitimate separate objects as fragmentation or connect them to
improve a score. No image dilation, inpainting or postprocessing as a repair.

Require validation dynamic PSNR no worse than the frozen parent by0.3dB,
dynamic alpha at least0.95, and no obvious missing parts/detached fragments in
the declared cross-view evidence. Metrics alone cannot prove a correct duck
shape: smooth blobs, disappearances and merged objects fail visual acceptance.
Use training/validation for iteration; reserve test RGB for the final selected
candidate. Fixed new view/time pairs have no independent GT, so report visible
integrity over this protocol without claiming all viewpoints or true3D recovery.

If I1 fails, preserve the negative result and diagnose graph coverage, strain,
appearance and visibility. Next investigate restricted sparse shared motion
and multi-time geometry fusion. Do not repeatedly increase a weight or lower
acceptance thresholds just to obtain a passing number.
