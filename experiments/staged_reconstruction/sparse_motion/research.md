# Second iteration: shared sparse motion, not another weight increase

Date:2026-11-03. First iteration I1 improved validation dynamic PSNR but visibly
failed cross-view integrity. The parent remains original S3-U final6k; I2 does
not replace the baseline with a selected intermediate or the I1 trained state.

[SC-GS Sec.4.1–4.3](https://arxiv.org/html/2312.14937v2) interpolates dense
Gaussian motion from sparse SE(3) control transforms. Its ARAP prior discourages
independent local deformation. Here256 immutable canonical controls are sampled
by farthest-point coverage; each query uses four canonical RBF neighbors.
An original PIDG grid/MLP predicts only these control transforms in full sparse
mode. Their complete original `R X + translation` mappings are interpolated.
Quaternions are hemisphere aligned; the output API encodes exactly the blended
center. Scale/rotation deltas also share sparse predictions. The same adapter
serves native rendering and all physics trajectory/finite-difference queries.

This is an adaptation: controls/radii and count stay fixed, no adaptive topology
or full paper optimization, no new network or material/force parameter. Spatial
canonical support can still link different surfaces; no physical ownership is
asserted. The prior cannot invent independently verified hidden geometry.

Retain all original model/Adam/RNG/camera state. Progressively mix dense→sparse
centers over400 updates, then eliminate dense motion queries completely. Retain
the same I1 strain term, image objective and original guarded physics. I2 takes
2,000 actual updates at width1600, matched to I0/I1. Mix is determined by saved
local iteration and protocol; controls are checksum bound. Flags-off mix0 returns
the original deformation result without any sparse query. No original/frozen
source is changed.

Required before formal: unit tests for rigid motion, quaternion signs, subset/
chunk query consistency, intended gradients and zero-mix identity; real native
and physics-center agreement; independent40→80 recovery. The initial100-step
smoke reaches only mix0.25, so preflight was expanded to500 actual updates to
exercise the entire400-step transition and at least100 fully sparse updates.
The passing initial100-step receipt is retained in smoke_audit.log and is not
sufficient for formal admission. No formal weights, budget or acceptance changed.
Use the same six fixed cameras and nine screen times, then all200 times before
any final acceptance. Preserve failures, reject holes/sheets/blobs or merged
objects despite lower proxy scores, and retain the validation quality gate.
Do not read test RGB during this iterative stage.
