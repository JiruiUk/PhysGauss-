# J4 Rubber Duck joint reconstruction pilot

Two matched arms completed2000 updates each on the target Rubber Duck data.
Allowing static opacity/SH to update jointly with the unified dynamic pool, with
the fixed0.1 J3 parent-depth self-reference, improves held-out RGB over continuing
only the dynamic pool. It does **not** yet remove geometric trailing support or
complete the fragmented ducks. Retain the appearance result, do not promote as
clean/complete reconstruction.

| All30 validation camera-time images, native1600 | Parent S3-U final6k | Static frozen+2000 | Joint+2000 |
|---|---:|---:|---:|
| Mean full-image PSNR/dB |28.1663|30.1266|31.3365|
| Mean full-image SSIM |0.969260|0.974798|0.978989|
| Mean L1 |0.005410|0.004083|0.003526|
| Mean0.8L1+0.2DSSIM |0.010476|0.008307|0.007023|

The joint bundle gains1.2099dB over the matched static-frozen continuation and
3.1703dB over the parent. This is a validation RGB result, not static geometry
accuracy or an isolated causal share for opacity/SH/depth. Black background
occupies much of the image, so visual/decomposed support evidence remains required.
The same120-view parent sampler/RNG generates exactly the same2000 indices in
both arms. No test RGB is used for selection.

Parent SHA8c767b7835eee936e49989369d7bc0277d7567d3aedaa7385c06654284b68ad0
retains its own4126 static and15507 unified dynamic points; no object count,
identity, reinitialization, growth/pruning or S1-C bank is introduced. This
legacy parent used conditional color masks. New updates use only calibrated
RGB/time with the original black RGBA compositing contract, no color masks or
alpha targets. This is not a newly trained unknown-static model from scratch.
“Real pilot” labels mean actual target Rubber Duck rather than the auxiliary
fixtures; they do not assert physical-camera capture. Rubber Duck belongs to
the PIDG physics benchmark; no independent static geometry or fixed-view
cross-time GT is supplied to this experiment.

Both complete Adams are inherited. Static values exactly match S1 final12k, so
its12000-step Adam is reused, not reset. Static geometry remains exact. In the
candidate, opacity/SH steps reach14000; frozen appearance and all static geometry
remain12000. Dynamic appearance/canonical/grid/deformation Adam states continue
from22000 to24000, not from6000 to8000. Velocity/stress parameters, their600-step
Adam state and graph caches remain exact: this is RGB reconstruction, not a new
physics-loss continuation. No original/frozen PIDG code or historical weights
were changed.

Qualification:10 tests,198 frozen training-input/source files, native1600 initial
RGB/alpha/D exact parity and RGB-gradient tolerance; isolated self-depth gradient
only to static opacity. Two arms each completed20 smoke updates. Independent
two-process complete model/twoAdam/bindings/RNG/sampler/graph restore and fixed
native-gradient next Adam update are exact, using the human-authorized recovery
acceptance. Original native40-step trajectory failures remain failures. A
CPU/CUDA comparator error occurred before freeze or updates and is retained;
the fix is local to the new pilot comparator.

Actual inspection includes both fixed86/188 views at2/100/200, separate static/
dynamic/joint RGB/alpha/D-alpha, candidate native1600 RGB, selected native
alpha/depth and all six original-trajectory observation comparisons. Static
edges and some training-trajectory duck appearances are cleaner. At c86/t200
and c188/t100/200, dynamic alpha still contains large bright trailing/halo
supports with dark RGB; several ducks remain fragmented in off-trajectory
view-time combinations. At c86/t100 a dark RGB upper patch still has broad
dynamic alpha/depth support. Merely allowing static revocation does not resolve
that. No obvious new large static-body disappearance was seen in the sampled
images, but this is not an independent completeness measurement.

On each fixed native image,1440000 pixels:

| Static self-reference proxy | c86 | c188 |
|---|---:|---:|
| Parent alpha>0.5 pixels |135441|148933|
| Joint alpha>0.5 pixels |134111|147546|
| Parent-supported pixels now alpha<=0.5 |1336|1387|
| D/alpha MAE on parent support |0.002774|0.002762|

The support changes are about1%; their correctness is unknown. Parent-render
depth is a self-reference and may preserve false support. D/alpha is a renderer
proxy; black low-alpha depth is missing, never zero error. Neither reduced
support nor small self-depth drift proves accurate geometry.

Six full Gaussian PLYs (static+canonical for parent/control/candidate) and nine
XYZ-only deformed-world diagnostics are checked against their exact states;
the latter are not full Gaussian models. Native fixed-view H264 videos include
parent/control/candidate joint RGB, separate dynamic RGB, joint alpha and static
RGB. Each uses200 original normalized times and is fully decoded; browser
playback is not inferred from decoding. Final video hashes/review are recorded
in videos.json/manual_review.json/final_review.json.

The original PSNR non-regression gate passes. Manual clean-support/completeness
admission fails: final completed_not_promoted. This is a limited RGB improvement,
not evidence that RGB is unsolvable or that geometry is now complete. The next
single audit should stay on Rubber Duck and quantify actual camera-time
coverage and visibility of the dark, high-alpha dynamic supports using supplied
RGB/cameras and frozen models. Reuse completed coupling/motion-integrity work;
do not delete points from darkness/residuals, invent independent cross-time GT,
repeat J3/J4 or raise weights. A later candidate needs its own fixed protocol
and state/gradient/structure checks. No automatic extended training is launched
by this result.
