# Validation review and model revision

State: completed_no_validated_improvement | Deadline UTC: 2026-11-01T05:17:35+00:00

C1 baseline reused. Parent screens are observed every5 seconds. No running recipe is edited in place.
Feedback proposals use validation only; changes create separate full-state3k branches.
One measured revision and at most one conservative retry; at most one additional50k promotion.

Latest: F2_conservative_retry / 20000 iter | PSNR delta vs C1 at same iter: +0.04834 dB
Quality guard: True | Integrity score: 0.011872836962477314
Latest: F1_feedback / 20000 iter | PSNR delta vs C1 at same iter: +0.50081 dB
Quality guard: True | Integrity score: 0.007225840528545251
Latest: H2_temporal_cohesion / 50000 iter | PSNR delta vs C1 at same iter: -0.38157 dB
Quality guard: True | Integrity score: 0.003972160060373721

F1_feedback: train
- Start from the best eligible20k validation configuration: H3_combined.
- Opaque appearance errors dominate alpha holes: add0.01 worst20%-foreground RGB emphasis.

F2_conservative_retry: train
- Follow-up20k failed quality/spill checks: halve auxiliary weights and disable cloning in a fresh branch.
