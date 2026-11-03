# S1-J1: one RGB-only matched joint Gaussian experiment

Preregister before implementation or new outputs. J0 qualified the static
opacity/SH gradient path only. Test the user's joint cleanup hypothesis with
actual Gaussian training, not another depth micro-diagnostic. The target stays
Rubber Duck, but no real continuation is admitted before independent synthetic
static/structure gates. D0-D7 and S1-C/N remain completed; no repeated old runs.

Use the existing two v2 conditions (diffuse/glossy),96 train RGB images each,
calibrated cameras and normalized timestamps. They provide independent camera
and time variation. Evaluator GT/alpha/static labels/body counts do not enter
the trainer. Existing RGB-only S1-N full_RGB_control6k checkpoints supply an
identical initial static bank and its complete Adam state. These parents were
trained with RGB-inferred geometric support and a position reference, not
semantic static labels; this conditional provenance must be disclosed. They
are not certified geometry. Checkpoint/input/source hashes qualify provenance,
not correctness. No D1-D7 depth/normal/DA-V2 output is used as new supervision.

Two arms, one switch: freeze all static parameters, or open static opacity/DC/
rest SH with original static learning rates/moments. Freeze static XYZ/scale/
rotation in both. Both train the same one canonical dynamic Gaussian population
with the original PIDG grid and deformation architecture, one native depth-
sorted static+dynamic rendering, and only0.8 full-image L1+0.2 DSSIM. No support
mask/alpha-one/residual classification/physics/force/normal/depth loss is added.
This differs from rejected S1-N's independent per-image nuisance planes.

Initialize the unified dynamic bank by copying all4000 parent Gaussian rows,
centering their coordinates, resetting opacity to0.01 and deformation heads to
identity. Copy parent SH/scale/rotation; no GT selection or semantic sampling.
This deliberately uncertain duplicate covers the train-inferred support,
including possible contamination. Dynamic geometry may still be incomplete;
report that limitation. Budget4000 is not object count; no physical identities
are estimated. Use S3-U fixed network/grid, rates and bounds. No growth/pruning,
random hypothesis banks, weight/threshold search or automatic extra rounds.

Each arm gets6000 new updates from identical parent RNG/camera stack and fresh
identically seeded dynamic parameters/Adam; retain all static Adam groups,
including frozen ones, to verify unchanged moments. Inherit parent sampler
position, record every new index, and save complete model/optimizers/RNG/stack/
visits/source+protocol hashes atomically. Keep initial snapshots. A stop file
saves full state and exits without resuming itself. Do not touch parallel I9.

Before formal training: meaningful CPU tests for switch/no mutation, static
geometry freezing, full-RGB loss, independent optimizer states and recovery;
native control/treatment value-equal RGB/alpha, flags-off native PIDG adapter
parity and finite intended gradients. Check dynamic grid/network gradients
after the identity heads have updated, rather than treating initial zero grid
gradients as an absent path. Freeze train-only inputs and imported native
sources before measurement. For each condition and both arms compare an
independent40-update run with two independent processes20+20: all model/buffer,
static+dynamic Adam, RNG, camera stack/visits and sample sequence must match
atol1e-6/rtol1e-5, with exact discrete state. This is not the no-update J0 check.

Then serially train the four arms, preserving all failures. Evaluate static-
only output on all32 cameras and8 genuinely novel cameras using the original
v2 evaluator-only GT. Retain S1-N's exact four strict promotion gates in both
conditions: novel static PSNR gain>=0.3dB, visible center-surface coverage loss
<=1 percentage point, thin coverage loss<=2 points, static RGB spill reduction
>=10%. Empty valid support is failure. Additionally require novel composite
observed-RGB PSNR degradation<=0.3dB across all4 time indices, no degradation
>1 point in full-denominator accurate static ray coverage or thin alpha recall.
Report D/alpha errors with missing/pixel counts, alpha-independent GT surface
denominators, thin structures, active centers and low-opacity rows separately.
0.05 center coverage cannot establish0.04 thin thickness or ellipsoid shape.

Export fixed train indices0/24/72, novel camera24 at all four times, static/
dynamic/composite RGB, alpha and D/alpha, PLY corresponding to checkpoints and
one serial-camera/time video per case; decode and actually review. Synthetic
camera24 is not Rubber Duck camera86/188. GT is independent evaluation only;
composite PSNR, lower opacity or deletion alone cannot establish static cleanup.
Final review must audit all four6000 sample traces and source/checkpoint hashes.

This single mechanism is an adaptation motivated by persistent+transient
decomposition (DeSplat already read) and original PIDG temporal coherence, not
a reproduction of those papers or an identifiable physical segmentation.
Success requires both static cleanup and retained structure; a dynamic bank
absorbing correct static surfaces is failure. Only a later separately fixed
Rubber Duck matched protocol could follow a passed synthetic experiment.
