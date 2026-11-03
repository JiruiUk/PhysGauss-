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
