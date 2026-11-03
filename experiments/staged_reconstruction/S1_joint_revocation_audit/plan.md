# S1-J0: full-RGB static-revocation feasibility audit

Preregistered diagnostic only, before implementation/results. The user asks
whether static smears can be removed jointly during dynamic reconstruction.
Current S3-U freezes static opacity/SH/geometry; original S6 only opens SH.
D7 remains rejected for depth/Gaussian training, and is not an input here.

Reuse the existing static_dynamic_coupling audit9a9c8fb6 rather than repeating
its conditional-color gradient or occlusion probes. That audit did not test
gradients into static opacity/SH under a label-free full-RGB objective. J0 tests
this limited missing capability without optimization or changing old sources.
It cannot establish correct ownership, geometric completeness or cleanup.

Load an in-memory copy of exactly user-selected S3-U final6k and its own static
bank. No graft of S1-C18k and no reinitialization. Freeze original protocol,
data receipt, checkpoint and all imported sources before formal measurement.
Historical S3-U training DID use conditional color masks; disclose this lineage.
J0 uses only original train RGB/cameras/times and full-image objective; it does
not import those masks, geometry priors, labels, body counts or evaluator assets.

At six fixed train-list indices0/24/48/72/96/119, render unchanged scene values
through native depth-sorted static+dynamic union. Default mode must leave all
caller flags/state unchanged. Audit mode temporarily enables static opacity,
DC and rest SH gradients only; static geometry remains frozen. Compute full-RGB
0.8 L1+0.2 DSSIM gradients for static opacity/SH, dynamic canonical and inherited
grid/deform parameters; export aggregate finite/norm/sign statistics, not a
learned static/dynamic segmentation. No parameter, Adam or camera update.

Native static-only/dynamic-only/combined images must be labeled as renders of
the OLD checkpoint. Combined improvement or opposing gradients would not prove
static cleanup, and this audit makes no update. Fixed cameras86/188 at times
2/100/200 are visualization only without independent novel-view GT. At observed
frames, dynamic-containing RGB is not a clean-static target.

Before six-frame measurement, test default-off/temporary flag restoration,
failure restoration and no caller/RNG mutation; native smoke checks same scene
RGB/alpha before and after enabling static autograd, finite gradients, state
hash invariance. Two independent processes must reload the pinned checkpoint
for fixed index0: exact underlying parameters/images, aggregate gradients within
predeclared atol1e-6/rtol1e-5. This is no-update reload, not Adam/RNG/camera
training recovery. Preserve any failed logs; do not adjust tolerances.

Audit outcomes only qualify an update path or expose its failure. Full-RGB
gradients may remove correct surfaces, compensate for reflections or hide
incomplete dynamics. Geometry anchors/occlusion attribution still need independent
evidence. Any learned joint candidate requires its own fixed matched Gaussian
protocol, full native recovery and two-condition independent static/structure
gates before Rubber Duck continuation; J0/D7 cannot bypass these. No sweeping,
new physics/force/materials, topology or restored paused training.
