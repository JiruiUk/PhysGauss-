# I6: small shared motion rejected at original warmup gate

Actual training completed16,000 warmup updates, with the original12k800+
4k1600 width schedule, sampler and initial full state. The original
deformation/velocity/stress networks and Gaussian population are retained;
positions use16 canonical FPS controls from update1. No depth, tracks or extra
geometry losses were used.16 denotes capacity, not physical bodies.

Validation30: full PSNR26.575205, dynamic PSNR25.540285, dynamic alpha0.993656,
static PSNR34.065888. Frozen static-only PSNR35.643816: drop1.577928dB,
exceeding the original1dB admission guardrail. The static parameters themselves
remain frozen. Dynamic Gaussian occlusion/leakage affects their combined
render; this observation does not establish a unique cause.

Native1600x900 visual inspection: fixed188/t43 has substantial black holes,
detached particles and incomplete yellow surfaces; fixed188/t146 has stretched,
merged-looking, incomplete surfaces; fixed86/t86 is smoother but still lacks
shape detail. The first two already fail completeness. The numerical gate
failed independently. No formal physics updates,22k completion,1200-frame
acceptance scan or test-RGB evaluation is claimed.

The independent `warmup_state_audit.json` verifies16,000 optimizer updates,
zero physics calls, exact original initialization/sampling/RNG, finite Adam
state and unchanged static parameters. Preserve the original failed
`warmup_admission.json`; actual image review is recorded separately.

Shared capacity alone did not resolve the missing geometry. The I2 read-only
spatial-map audit also found negative local determinants despite rotation
blending being close to orthogonal. Next test a time-conditioned invertible
position map inspired by CaDeX/MoDGS, with original PIDG image/physics goals
and unchanged numerical/visual acceptance. Analytic invertibility is a motion
property and cannot itself certify complete shape.
