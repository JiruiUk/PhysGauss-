# Matched optical-flow / structural-velocity comparison

Run from the project root after activating `reproduction/scripts/enable_cuda.ps1`:

```powershell
envs/pidg-cu128/python.exe -u -m variants.flow_comparison.run --preflight-only
envs/pidg-cu128/python.exe -u -m variants.flow_comparison.run --hours 12
```

The queue trains **K1_flow_independent first**, then **K2_flow_trajectory**. Both
reuse the identical full 3,000-step checkpoint, including optimizers, camera
stack and RNG, and train to 50,000 total steps. Only their velocity definition
differs. The structural model retains all parameter objects/checkpoint keys but
freezes its unused independent velocity head. Its stress head remains trainable.

`observed_flow`, `material_momentum`, `velocity_source`, and
`velocity_clock_mode` are explicit configuration switches. When disabled, the
trainer dispatches to the inherited original operators. No renderer, dataset
loader, camera sampler, material constitutive law, or force module is replaced.

Flow targets use the 119 actual chronological pairs within the 120 training
images, with their true `(target-source)/199` intervals. UniMatch weights/code,
input images, camera metadata, teacher snapshot, and every generated artifact
have SHA-256 locks. Camera compensation uses **estimated** frozen H3 depth;
neither depth nor optical flow is ground truth. Validation/test images are not
used to generate training targets. RGB is 1600x900, while optical-flow targets
and raster correspondence are 400x225.

The custom LPFM uses the original Gaussian affine-covariance idea and detached
visibility/contribution weights. It fixes H,W,2 component normalization, avoids
clipping pixel motion, uses a stable SPD square root, and evaluates projection
and covariance with Torch. A CUDA-forward parity test verifies these Torch
projections against the installed renderer. This avoids incorrect pixel scaling
in the released metadata projection backward. This is not a verbatim paper
implementation and should not be presented as an official full reproduction.

Both new arms share a local **Lagrangian** momentum residual. The input spatial
coordinate is the material label X; acceleration is evaluated at fixed X, while
stress divergence and the original deviatoric velocity-rate correction use
weighted local fits in the current world positions x(X,t). There is no additional
Eulerian convective term. The common discretization avoids unsupported
hash-encoder coordinate Hessians in the structural arm. Neighborhood condition
and valid fractions are logged; an all-degenerate sample raises an error.
Predicted PIDG stress is retained. The residual uses Huber delta=1, rho=1,
weight=.01 every 100 updates, starting at 8k with a 4k ramp. This common change
means comparisons with historical H3/J1 cannot isolate optical flow alone.

Trajectory velocity differentiates the **full renderer centre**, R(q(X,t))X+b,
using second-order finite differences (h=.001); acceleration uses h=1/199.
Second-order one-sided stencils handle endpoints. Checkpointed, deterministic
trajectory queries avoid consuming regularization RNG and limit GPU memory.
Units are distance per normalized time, not metres per second.

K2's zero internal velocity/trajectory disagreement is true by construction,
not independent evidence of physical accuracy. The inherited console velocity
head mean remains a statistic of the unused legacy head in K2; use the labelled
`kinematic_diagnostics.jsonl`, `structural_consistency.jsonl`, and the flow/momentum
logs for the selected velocity definition.

Preflight includes analytic and CUDA tests, both 100-step training smokes with
70-step full-state resume, direct loss-to-parameter gradient audits, and a
20-step flags-off smoke. Full runs select checkpoints using validation only;
both finish and selection is locked before final-test rendering. The test set
has historical exposure from earlier experiments. PNG concatenation crops are
verified byte-for-byte, PSNR is recomputed independently, MP4 frame counts are
checked, and ZIP CRCs are verified before delivery.
