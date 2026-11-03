# S1-D2 source-envelope follow-up

Single candidate and all reliability/coverage gates predeclared at3b69cf52,
before any candidate outputs. It uses fixed D1 depths/pixels, original six
source views, empirical returned-depth envelopes, a local quantization floor,
distinct calibrated camera poses and possible occlusion markers. Uncertain
points remain stored; neither probability of staticity nor physical identity
is inferred. There is no Gaussian/physics training in this evidence stage.

Implementation/preflight:21 tests passed (13 new/8 inherited), including flat
plane calibrated geometry, exact disabled object identity, missing/occluded/
out-of-view samples, duplicate/signed-zero camera poses, temporal insufficiency,
and maximum reprojection rather than median-only screening. No PIDG baseline,
renderer, old loss, weight or source file was edited. Source/parent input/protocol
freeze9032241d689c212b412dcfd23e4e02f0d63fe376ed458bea0477f145f23b7afb.

Next: one serialized inference and evaluator-only GT audit for diffuse/glossy,
96 training images each. Check actual processes, logs, stop.request and locks at
runs/pidg_staged_reconstruction/S1_depth_uncertainty/v1 before launching.
Keep per-view records and preserve user stops. Native full-Gaussian/Adam/RNG
recovery is required only for a separately declared subsequent training stage;
do not describe this data-only inference as that training or artifact cleanup.

## Two-condition inference completed

All96 views per case completed with unchanged423747/423100 D1 candidates and
depths. Predeclared tight evidence:8600/8521 candidates. Every unavailable or
wide candidate remains stored. All per-view files and parent/source hashes are
bound in v1/inference.json; no evaluator GT was read. The independent evaluator
is running. These counts alone establish neither accuracy nor usable coverage.

## Independent audit and actual image review completed

Both cases fail only coverage/retention parts of the fixed evidence gate:
tight retention2.030%/2.014%, visible supported retention2.041%/2.026%, thin
supported retention3.171%/3.149%, all below25%. Tight interval GT coverage
95.860%/95.963%, large-error fraction0.686%/0.493%, depthMAE0.00682/0.00673.
AUROC0.7622/0.7631 is useful ranking but not a calibrated static probability.
Widening/relaxing to pass is not permitted; no paired Gaussian or real training.

Actual c00_t1/c08_t0/c24_t0 error/score/width/selection-GT panels reviewed for
both cases: tight evidence concentrated on a few column patches, not broad
base/boundary support. All candidates and source/parent data retained. See
v1/assessment.md, gate.json, final_review.json and ../S1_depth_evidence reports.
The tested RGB-MVS chain has a measured reliability/coverage bottleneck, not a
proof of impossibility for all RGB methods. Next requires choosing an external
learned geometry prior or additional simultaneous views; details next_choice.md.
User explicitly selected fixed Depth Anything V2 Small RGB geometry prior.
Continue that new branch after report/plan sync; no pause or repeated approval.
Do not stop unrelated tasks or invent user stops. Do not repeat S1-C/N/D1/D2
or sweep this gate. A new prior protocol must not pretend D2 itself passed.
