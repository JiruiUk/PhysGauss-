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
