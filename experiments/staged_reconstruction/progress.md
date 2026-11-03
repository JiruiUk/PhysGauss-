# S1–S6 progress

S1 implementation and preflight passed.21 related tests passed, native1600×900
gradients/Adam updates passed, adapter output exactly matches the unchanged PIDG
Gaussian renderer, and native projection agrees with metadata camera math within
0.000107px. A60→120 short training recovery completed with full optimizer/RNG
and camera stack. No dynamic parameters are present or optimized.

The static bank initializes4126 points from training-only multitime silhouette
support. Early preparation attempts using the legacy/canonical bounds were
retained in preparation_attempt1/2.json; no full training used those attempts.
Metadata-camera static-track held reprojection mean0.536px versus legacy0.757px
on the same legacy-selected tracks; this is conditional evidence with selection
bias, not independent calibration truth. The original PIDG loader is unchanged.

S1 implementation milestone `f92fc7a7`; formal12000 updates completed. Native
validation/test and actual RGB/alpha review passed conditional admission as the
static initialization for S2. Test static-region PSNR35.79119dB, mean alpha
0.997191, known-background mean alpha0.002614. Selected opacity median0.6584;
only0.291% below0.01. Edges remain imperfect; no dense geometry/contact GT claim.

Final test-path failure was repaired in a separate evaluator, with zero repeated
training updates and unchanged source/checkpoint. Final checkpoint SHA:
`2dd8d5e0ea35d55e648aa236ea779eebe2fe3d415d34ad931e732f0140b89031`.
Assessment: `S1_assessment.md`; full evidence: `S1_final_verification.json`.

S2 implementation/preflight are complete. Four independent original PIDG
deformation networks and fixed canonical groups initialize15,507 dynamic
Gaussians; accepted S1 static4,126 points remain frozen and excluded from Adam.
27 related tests passed, native1600x900 gradients are finite, all three dynamic
parameter groups receive gradients, and separate-process40→80 recovery retains
all88 Adam states at step80 plus RNG/camera stack. CMR/contact/tracks are off.

Training-only adjacent reciprocal raw-FB identity audit covers120 observations
and119 image pairs. Four early image identities persist through42. Seeds1/2
lose continuity at43, seed0 at52, seed3 at166; unknown is never filled by nearest
component or late component number. These local image links are conditional
evidence, not physical/material identities or a bridge through occlusion.
The initial depth slab is an explicit monocular prior, not reconstructed depth.

Formal S2 protocol preselects16,000 updates (800px through12k, native1600px
thereafter), final checkpoint, native val/test, dynamic-only alpha and native
per-object/fixed-camera review. Training admission is not yet established.
S2–S6 remain active. Automation `pidg-s1-s6` follows the whole sequence and
does not finish on an implementation, audit or individual stage completion.

S2 RGB-only formal16k completed with native val/test. Native validation dynamic
PSNR9.351dB and dynamic-only alpha0.4945 fail admission20dB/0.85; composite alpha
0.6280 would overstate dynamic support. Static validation33.894dB also exceeds
the allowed1dB loss against S1. This is a failed S2 candidate, not completion.
Final native/contribution/video review is running; all original sources/states
remain unchanged. Independent S2 recovery adjusts the uncertain depth prior,
foreground RGB weighting and conservative one-sided image coverage. Unknown
physical identities remain unknown; they receive only unlabeled union coverage.
31 related tests, native gradients/static freeze and separate40→80 recovery pass.
The single repaired candidate retains the same point/network budget and16k final
state; no sweep or post-hoc threshold relaxation. Reports/state are isolated in
`S2_recovery/`. S3 semantic operator tests are being prepared independently, but
S3 training remains gated on actual S2 reconstruction admission.

Second S2 candidate also completed16k/native val/test. Dynamic PSNR25.086dB and
dynamic-only alpha0.98499 pass, but static31.335dB versus S1's35.644dB fails the
unchanged1dB preservation gate.5k read-only scale audit shows entire groups0/1
clamped to1e-4 at42/86 and other groups inflating to0.15. This supports repairing
the additive scale degeneracy rather than calling high opacity a successful
persistent-object reconstruction. Final contribution/fixed-camera export is
running; all prior training code/states are preserved.

Third sequential correction `S2_geometry_recovery` keeps initialization, point
and network budgets; it adds a flagged positive bounded scale policy and
symmetric static RGB normalization.40 related tests, native gradient/freeze
checks and separate40→80 recovery pass. Formal16k training is being started.
This is a diagnosis-driven correction, not a candidate/weight sweep.

S3 engineering preparation includes six semantic/gating tests, a GPU probe with
finite gradients into fields, deformation, grids and canonical points, and an
actual Adam field update; static remains unchanged. The native6000-step S3 entry
is implemented with a strict reviewed-S2 hash gate and full continuation states;
formal S3 smoke/training remain pending S2 admission. S4 image-only sparse track
extraction and two advection tests are also running/prepared, without Gaussian
bindings or any claim that the later stages are complete.

Third S2 correction completed16000/native val/test and image/video review.
Validation dynamic24.25343dB, dynamic-only alpha0.982654, static35.53011dB pass
the unchanged numeric gates. Conditional continuation is allowed for subsequent
reconstruction corrections; geometry success is NOT established. Fixed86/188
novel views still fragment, and green own-component share at42 is0.479. These
unresolved results remain in S2_geometry_recovery/assessment.md; no physical
identity, contact geometry or complete novel-view claim is made.

Real-parent S3 smoke40→80 failed on insufficient nondegenerate neighborhoods at
step80. The last atomic40 checkpoint and failure are preserved under
runs/pidg_staged_reconstruction/S3/pregraph_smoke_failed. The frozen trainer is
unchanged after this failure. Independent state_recovery entry replaces canonical
kNN with current-world same-bank kNN, expands support64/128/256 while retaining
the same rank and25%valid gate, and preserves graph-cache state explicitly.
The corrected smoke is running; this is not formal S3 training yet.

S4 image fragments131 are prepared without held-out data or Gaussian bindings.
Binding now requires one fixed row to support every observation with complete
visible alpha*T, known early component purity and tight reprojection; conflicts
are rejected. S5's conditional static volume exactly rebuilds S1 eligibility,
and manufactured sphere/plane300-step matched optimization plus independent
150→300 recovery passes. It is not Rubber Duck contact validation. S4/S5/S6
4000-step continuation entries are implemented but remain gated on real-parent
short training and reviewed prior stage. Physical force/friction/restitution stay
off.49 earlier related tests and8 new graph/binding/entry tests passed.

Corrected S3 separate40→80 recovery verified136 Adam states, exact Python RNG/
camera replay, finite updates and static freeze. Formal6000 is now running via
state_recovery.train (PID267804); the full57-test suite passed. The earlier
rank failure and frozen source remain preserved, not overwritten.

S3 completed6000 actual updates and native val/test. Validation dynamic25.39230dB,
dynamic-only alpha0.985505, static35.94167dB; test dynamic25.55725dB. Exact final
Python RNG/camera replay and finite136-state Adam pass; all fixed-frame current-
world physics stencils pass. Nine observed/group panels plus native pixel crops
for fixed86/fixed188/original at2/86/188 were actually viewed; all3 videos decode
200frames each. Fragments/sheets and group leakage remain, so admission is only
for conditional sequential reconstruction correction, not geometry/physical-ID
success. S4 exact-S3 fixed-row binding is running.

S4 binding accepted91 fragments/455observations/79fixed rows across all4 early
banks;40 fragments rejected. The inherited CUDA color backward missed the tiny-
patch conservation gate (relative error0.001116); this failure is preserved.
Exact forward one-hot contributions replace the approximation without relaxing
the5e-4 gate; maximum partition error1.62e-7.40→80 smoke verifies136 optimizer
states and exact sampling; formal4000 S4 is running (PID262500).

An independent new S3 binding-baseline audit (separate output, preserving its
final report) gives0.536841px at400px over455 observations. All currently lie
outside the conditional static interior: signed distance minimum0.113664,
margin0.08,0active violations. This cannot validate collision. S5 must record
actual eligibility/activation at its parent/final states; a zero prior may not
be advertised as a demonstrated nonpenetration improvement.

S4 completed4000 actual updates, native val/test, exact sampler/optimizer audit
and actual observed/group/native-fixed-view review. Validation dynamic25.34022dB,
alpha0.986185, static36.19936dB pass unchanged image gates; test dynamic25.50344dB.
All455 track observations worsen0.536841→0.785017px@400. This is a negative
track-accuracy result, not a successful correspondence improvement; fragments
and bank leakage remain. Preserve the outcome and do not increase weights/sweep.
Conditional continuation permits only the already planned S5/S6 curriculum,
with the predeclared lower S6 learning rates; physical identity stays unknown.
S5 parent's455 qualified observations have0active conditional interior violations
(minimum signed distance0.112658, margin0.08). Thus no actual collision correction
has been demonstrated; S5 real-parent40→80 recovery smoke is now running.

S5 real-parent40→80 recovery passes136 Adam states, exact sampler/RNG and
static freeze. Formal4000 launched PID263796. All prior losses/weights remain
frozen. Whole-run conditional-interior activation will be audited explicitly.

S5 completed4000 actual updates, independent native val/test and full-state
audit, observed/group/native fixed-view review and three200-frame video decode
checks. Validation dynamic25.603743dB, alpha0.984816, static35.983147dB;
test dynamic25.745418dB. Track error0.664671px@400 remains worse than the
S3 baseline0.536841. All800 prior calls and16000 eligible observation instances
had zero active violations: no actual collision correction was demonstrated.
Novel-view sheets, holes and late-bank leakage remain. S5 is finalized only for
the predeclared conditional S6 reconstruction continuation; object-object
nonpenetration/contact forces/friction/restitution remain unvalidated/off.
S6 real-parent40→80 recovery smoke is now running, with0.25 learning rates,
static appearance unfrozen and static geometry exactly fixed to S1.

S6 actual40→80 independent-process recovery passes138 Adam states, exact
Python RNG/camera replay, finite model/optimizer and static geometry freeze.
The two additional appearance tensors update as intended. Formal4000 launched
PID262664. Completion still requires native val/test, independent objective and
source/sampler audits, actual fixed-view review and final measured report.

S6 completed4000 actual updates and native val/test. Independent full-state,
source/parent hashes and exact camera/Python-RNG replay pass;138 Adam states
are finite. Static geometry exactly matches S1; static appearance changes as
planned. Validation dynamic28.327031dB/alpha0.985981/static36.169914dB;
test dynamic28.563049dB. All455 qualified training-track observations now average
0.427107px@400 versus S3's0.536841 (20.44% lower), while the intermediate S4
negative outcome remains preserved. All800 conditional-interior calls are
inactive, so there is still no demonstrated collision correction.

Actual observed/group panel and nine native fixed86/fixed188/original pixel
crops were viewed; all three videos independently decode200frames. Observed
contours improve, but novel-view sheets/holes and long late green-bank streaks
remain. No complete geometry/physical-ID/contact success is claimed. S6 formal
result is finalized, completing this revised conditional training curriculum;
the original full contact scope remains incomplete. Final standard native
quality and measured report/plan are being consolidated; no new training starts.

Final standard native metrics for S2--S6 and the measured report, corrected
scientific trend figure and local video gallery are complete. Final S6 full-frame
val PSNR30.585203dB/SSIM0.975011/VGG-LPIPS0.041803, test LPIPS0.041247.
Actual plot review caught categorical insertion-order error in the first figure;
the original image/issue receipt are retained and explicit S1--S6 numeric ticks
fix the chart without changing measurements. The corrected figure was viewed.

Final integrity verifies9 source/input/protocol freezes covering247 unique
files, all6 final checkpoints, reviewed-image hashes and15 exported videos.
No task trainer/evaluator remains active. The revised training budgets sum46k;
two earlier failed S2 runs add32k preserved updates. Original geometry/physical
ID/contact limitations and the calibrated-RGB validation dependencies remain
explicit in both plans. Results are committed separately from implementation
and smoke milestones; pause pidg-s1-s6 after this final report commit, rather
than starting an unplanned new run or continuing duplicate notifications.

Playback follow-up: the gallery's original mp4v exports decoded successfully
but were not HTML5-browser compatible. Fifteen separate H.264 Main/yuv420p/
faststart display copies now decode200frames each; original exports/checkpoint
hashes and metrics remain intact. The gallery uses the browser copies and direct
video links. See playback_fix.md and browser_video_conversion.json. No training
or automation was resumed. Local-file browser UI observation was blocked by
the browser tool protocol policy; compatibility checks do not claim UI playback.

C1 reference follow-up: the user explicitly identified C1_repaired50k, whose
historical export folder is C1_original_50k. Checkpoint/source-video hashes
verify that identity. Added nine display-only H.264 comparisons covering fixed86,
fixed188 and the original camera trajectory: C1 alone, C1 versus S6 and the six
models C1/S2/S3/S4/S5/S6. All200 source frames are retained and displayed at20fps;
the old C1 label banner is cropped without changing the RGB region. The main
gallery now links this comparison. Actually reviewed the three early/middle/late
pair sheets and the fixed86 middle-frame grid. C1 is sharper on the sampled
original-camera frames but still shows severe novel-view sheets; S6 keeps the
sampled static support more complete while dynamic fragments remain. Legacy
versus metadata CV cameras and different capacity/budgets prevent a matched
causal comparison. C1 leads its C1/C2/C3 image-metric cohort, not a certified
ranking of all historical geometry. No training or automation resumed; original
checkpoint/video hashes and historical results remain preserved. See
C1_baseline_comparison.md and its generation/review receipts.


## S3-U: user-authorized unified dynamic population (2026-11-03)

The user selected one unified dynamic Gaussian population without estimating
body number. This is a separate branch, not unknown-count instance discovery.
No instance labels, original four-group initializations, identity ledgers or
trained S2/S3 dynamic weights are inputs. Frozen S1 static support is shared.
One shared canonical population, deformation network and velocity/stress field
replace separate instance fields. The global15,507 point budget is unchanged;
capacity and initialization differences are disclosed rather than treated as
a matched single-switch ablation.

Frozen protocol: staged_reconstruction/S3_union/protocol.json. Execute16k
image-only warmup, actual native review and unchanged numeric gates, then6k
S3 kinematic/CMR continuation. Existing weights, time units, camera convention
and static freeze remain. Validate each phase's40-to80 full-state recovery
before formal training. Dynamic-only spatial stencils have no physical-boundary
exclusion. No contact/constitutive/unknown-force/instance-count inference is on.
Compare native val/test and fixed86/fixed188/original-camera videos against
original S3, preserve negative results and late identity uncertainty.

Implementation:49 related tests, native renderer parity(max3.5763e-7) and
warmup40-to80 Adam/RNG/sampler/static-freeze verification passed. Formal warmup
is ready; no historical paused trainer or automation is resumed. Test setup
failures involving the shared temp directory and a missing new parent directory
are retained in the test receipt; the isolated fresh test directory passes.


S3-U warmup milestone:16k actual updates and native validation completed.
Dynamic23.990448dB/alpha0.986168/static34.829215dB pass the unchanged gates;
static drop from S1 is0.814601dB. All16k sampler choices/final RNG and22 Adam
counters verify; S1 tensors remain exact. Actual observed/fixed-view review
retains fragmented sheets/streaks and blurred foreground, so admission is only
for the declared6k S3 representation comparison, not geometry/contact success.
The real-parent physics40-to80 recovery passes34 Adam states and all RNG/static
checks. Original sources/weights and loss thresholds are unchanged. See
S3_union/warmup_assessment.md and receipts. Independent preview failure and
source are retained; its CPU/CUDA comparison repair changed no training state.

### S3-U completed and reviewed (2026-11-03)

The predeclared16k reconstruction warmup plus6k S3 continuation completed22k
updates. No physical body number is supplied or estimated;15,507 dynamic points
share one canonical pool, deformation field and original PIDG velocity/stress
field. S1 static tensors remain exact. All22k sampler choices, final Python/
NumPy/Torch/CUDA RNG payloads, Adam counts and source/input/protocol freezes
pass;49 related tests and both independent40-to80 recovery checks pass.

Final native validation dynamic PSNR is25.181221dB versus original S3's
25.392299dB(delta-0.211079); static34.999955 versus35.941669dB. Standard full-frame
validation PSNR28.165765/SSIM0.969167/VGG-LPIPS0.045682; test dynamic25.393077dB.
The4k intermediate dynamic26.156537dB is retained, but the preselected6k endpoint
is reported without cherry-picking or changing weights. Shared architecture
has3,645,217 trainable parameters versus5,831,273; fresh initialization and union
coverage also differ, so this is not a single-switch causal ablation.

Actual review of13 recorded images finds persistent fixed86 late yellow sheets/
streaks and fixed188 middle fragments/stretching, plus incomplete observed RGB
fit near42/43. Neither representation certifies complete novel-view geometry;
physical IDs and boundary-safe neighborhoods remain unknown. Six new H264
videos fully decoded200frames each; codec/faststart, camera fingerprints and73
local HTML links verify. All six original final checkpoints, C1 protection hashes
and15 original stage videos remain intact. No task trainer/evaluator remains;
training lock is free. See S3_union/assessment.md and completion_integrity.json.

The gallery now links original-S3/S3-U comparisons at fixed86/fixed188 and the
original trajectory. Original stage metrics/figures are unchanged. This request
ends at S3-U; S4-U--S6-U, new physics and historical paused automations are not
started. Further evidence work still prioritizes independently observable
geometry; no count/identity inference is introduced into this unified-pool branch.
