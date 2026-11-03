# MonST3R geometry evidence pilot

Primary paper: [MonST3R, ICLR2025, sections3.1–3.4 and Appendix C](https://arxiv.org/html/2410.03825v2).
It estimates per-time pointmaps in a common pair-camera frame, then aligns
multiple pairs. Confidence-aware dynamic geometry is potentially complementary
to the current appearance-conditioned reconstruction. A pointmap does not by
itself establish persistent material identity or complete unseen surfaces.

Official implementation: [Junyi42/monst3r](https://github.com/Junyi42/monst3r),
pinned at574cc77ad278bad582f470e5382624e01f8769a7, with CroCo submodule
bd6f4e07d5c4f13ae5388efc052dadf142aff754. The README reports memory beyond our
16GB GPU for its65-frame global example, with reduced-memory/window options.
Our pilot therefore evaluates isolated pairs; it is not the full paper system.

Source inspection found preset camera/intrinsic interfaces. However,
`compute_global_alignment(init='known_poses')` explicitly reenables pose
gradients, and window parameter copies can also enable them. Intrinsic preset
averages fx/fy and only sets the principal point when `optimize_pp` is enabled.
These defaults cannot establish a fixed-camera implementation for this project.

The separate source/camera audit tests metadata GL-to-CV conversion and
pixel-center resize transforms on all120 training cameras, without reading RGB.
Roundtrip error is approximately1e-12 pixels. At512x288 no crop is needed for
1600x900 inputs. This checks numerical conventions, not image calibration truth.
The pilot does not run a camera optimizer: known metadata is used to check the
network's raw pointmaps, not supplied as conditioning to its forward network.

The official Hugging Face revision1ea2a9eade23a10898ec73be3fd44b5aa4ea0255
provides safetensors. Its2,284,790,056-byte weights match the official LFS SHA256
53089f3bf16ef8d546d9c00232474f2b42490ce359986a1795795c6dc07a3e57.
No pickle/eval checkpoint loader is used. Eight tied DPT aliases are restored
from safetensors metadata after checking live parameter storage identity;
strict state loading then succeeds. The isolated vendor removes one unused
RAFT import and makes one trajectory-evaluator import lazy. Forward mathematics
and weights are unchanged; the baseline Python environment is not modified.

Before predictions, `pilot_protocol.json` fixes six train-only anchors, two
nearest available training partners each,512x288, confidence>=3, sufficient
foreground/static support, held-out static errors, partner stability and
known-camera self-projection tolerances. Each pair has an independent scale
gauge, so one scalar is fitted from one-third of its static support. The other
two-thirds are held out. No depth shift is fitted. This differs explicitly from
the earlier VDA metric/global-scale protocol. Frozen S1 depth is conditioned
model evidence, not GT; passing the pilot would still not prove dynamic truth.

This pilot was run while I9 continued unchanged. No prediction is a PIDG target,
no object count/instance/material label is inferred, and no val/test RGB is read.
The separately preserved failed runs are loading/reporting repairs, not tuning
of prediction weights, confidence, pairs or quality thresholds.
