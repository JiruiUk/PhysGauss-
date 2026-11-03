# I9 repository audit and integration map

The baseline mapping already exists in `PIDG_REPO_MAP.md`; historical sequential
M0-M10 receipts remain in `experiments/milestones.json`. I9 is an independent
reconstruction-prior experiment. It does not replace that engineering work or
claim that historical smokes establish cross-view completeness.

|Concept|Current implementation|
|---|---|
|Training entry/config|`relative_strain/train.py`, immutable `protocol.json`, explicit opt-in CLI|
|Original full step-zero state|`early_depth/train.py:load_initial`, original S3-U warmup initial SHA in protocol|
|Gaussian population|`union/model.py:UnionScene`, original15,507 dynamic centers and4,126 frozen static points|
|Position field|`invertible_motion/core.py:InvertibleMotion`, original initialization seed/capacity|
|Covariance transport|Unchanged `covariance_transport/{core,runtime}.py`|
|Velocity/stress/momentum/flow|Original `scene.physics`, invoked by `early_depth/train.py:step_loss` with original guard/schedule|
|New loss|`relative_strain/core.py:strain_prior`; analytic spatial differential, relative right solve, log singular values|
|Renderer|Original CUDA renderer via inherited precomputed-covariance wrapper; no renderer edits|
|Dataset|Existing train120/val30/test50 kinematic split; train/val only for decisions|
|Validation/rendering|Own `validation.py`, `evaluate.py`; native1600, same fixed cameras/times and original gates|
|Checkpoint/recovery|Unchanged `union/runtime.py:save/restore`, own run directory/protocol hash|
|State/native qualification|Own `preflight.py`, original control100 and independently resumed prior40/80/200|
|Read-only diagnostics|Own `preview.py`, `audit_snapshot.py`; pinned checkpoints, no optimizer updates|

All new prior parameters are fixed scalars in the protocol. No learnable
parameters/groups beyond the already controlled I8 map are added. Original
parameter tensors/groups remain; original grid/deformation heads are inactive
only in the enabled transport arm. All-off follows the original path. Turning
only the strain flag off reproduces I8's objective. Tests cover these two cases.
