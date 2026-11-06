# J7: user-authorized color-free static library replacement in J6

The user explicitly requests displaying the existing static reconstruction made
without a blue threshold, then integrating it into J6 and testing the result.
The native c86/c188 source images were displayed before any fusion training.
This is an exploratory target-data replacement, not retroactive promotion of
S1-C or a claim that its old geometry gates passed.

Use the previously retained S1-G / S1-C unchanged-row control18k (3273 points).
Its inputs were RGB/cameras and generic geometric support; no blue/static masks,
old static weights or object count. Do not select between several new candidates.
The quarantine arm was not better in the prior matched evaluation and is not used.
Native source images show tall structures and bases, but also yellow-looking
contamination, edge haze and floating support. Preserve unknown identities.

Two arms each make 2000 additional native1600 RGB updates. Both inherit exact
J6 final dynamic model/Adam, physics/Adam/graph, RNG and remaining camera stack.
old_static_control retains J6 static+Adam; colorfree_static replaces only the
static module with the complete3273-point RGB-derived bank and its complete
18k Adam. Preserve candidate point_ids/born_at buffers and source confidence
provenance. Neither source confidence maps nor old color masks select training
pixels. Geometry/topology are frozen; only static opacity/SH and dynamic
canonical/grid/deform update. Both use the unchanged0.8L1+0.2DSSIM full-image
objective plus0.1 static opacity-only depth self-reference to EACH ARM'S OWN
frozen initial static bank. The candidate must never use old color-conditioned
static depth as a teacher. Parent depth is a self-reference, not independent GT,
and can protect erroneous support. No new pseudo-depth/tracking/spring losses.

This is a library-replacement experiment: point counts and static Adam histories
differ, so its outcome cannot be attributed solely to removing a color rule.
Dynamic J6 itself retains historical color-conditioned lineage; replacing static
does not make the entire model a from-scratch label-free reconstruction.

Before formal training: default-off/no-mutation/shape/finite/schema unit tests,
exact source-bank and all non-static/J6 state composition, native flags-off RGB/
alpha/D parity and finite intended gradients, plus20 actual updates per arm.
Then two independent processes per arm must restore full model/twoAdams/bindings/
RNG/camera/physics graphs exactly, reproduce native images and original gradient
tolerances, and give exact next Adam state under one fixed native gradient.
Use the user's previously accepted recovery contract; retain all old CUDA
40-step failures, epsilon1e-15 and original rates. Freeze inputs and code.

New opt-in modules only; no frozen-source/weight changes. Task-local locks prevent
duplicate launches. Global/task stop.request saves full state, releases GPU and
never resumes automatically. Check actual processes first; do not touch others.
Save complete checkpoints every250 updates. Preserve failures and partial logs.

Evaluate unchanged J6, direct fusion before training, matched control and trained
fusion on all30 validation images; no val/test-driven training selection. Inspect
native86/188 at2/100/200, static/dynamic/joint RGB/alpha/D-alpha, and six original
train camera/time observations. Export exact complete Gaussian PLY and two200
frame H264 comparisons, decode all frames and inspect contact sheets. RGB PSNR
drop must be at most0.3dB versus matched control AND retained J6; actual static
structure nonregression and dynamic cleanup/completeness remain required.
No independent fixed-camera cross-time geometry GT exists. Alpha/D are proxies;
black missing pixels, darker RGB or fewer points cannot certify correct geometry.
Do not replace retained J6 automatically. Report the measured result, leave old
models intact, and do not automatically extend, sweep weights or add physics.

Commit preregistration, implementation/qualification, training and reviewed
results separately, staging only this task's files and protecting shared edits.
