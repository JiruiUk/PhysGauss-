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
