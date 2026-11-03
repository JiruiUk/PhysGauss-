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

S2 is now preparing persistent known local image identities and per-object
canonical initialization. S2–S6 remain the active task. Automation `pidg-s1-s6`
follows the whole stage sequence; it does not finish on S1 completion.
