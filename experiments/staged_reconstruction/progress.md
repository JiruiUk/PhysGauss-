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

Formal S1 protocol:12000 updates (800px first8000,1600px last4000), native final
evaluation, no densification, no dynamic branch or physics. Formal S1 starts
after the implementation milestone commit. S2–S6 remain pending implementation
and training after their dependencies; this queue is not limited to an R0 audit.
