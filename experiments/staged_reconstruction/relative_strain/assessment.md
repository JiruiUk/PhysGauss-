# I9: reduced relative distortion did not restore complete objects

The original16k warmup completed from the pinned original S3-U step-zero state:
12k updates at width800 followed by4k at width1600. The one change over I8 was
the preregistered robust relative-Jacobian prior adapted from Nerfies. There
were no new depth, track, material, force, instance, density or static targets.

|Validation30 at native1600|I8 warmup16k|I9 warmup16k|
|---|---:|---:|
|Dynamic PSNR|23.875979|22.946332|
|Dynamic alpha|0.982865|0.963690|
|Static-region PSNR of combined rendering|32.815685|31.894931|

I9 full PSNR is24.688758. The unchanged frozen static-only reference scores
35.643816 on the same static-region protocol. Combined-render static PSNR drops
3.748886dB, exceeding the original1dB warmup guardrail. The dynamic PSNR>=20 and
alpha>=.85 warmup gates pass, but **formal physics is not admitted:0 updates**.
`warmup_admission.json` remains the original failed numerical receipt; no gate
or source was changed. Its `native_images_reviewed:false` is preserved; actual
subsequent image review is recorded separately in `visual_review.json`.

Final checkpoint SHA256:
`ab8a5c64d8ae9671ea23028beb3cba1612a79a7f4caddd424a1254a9f9c773d7`.
`warmup_state_audit.json` independently verifies exact original initialization,
optimizer groups, all16k camera samples and Python/NumPy/CPU/CUDA RNG, finite
Adam state, inactive original heads, and unchanged static tensors. The18 CPU
tests pass again. Native flags-off parity, disabled-prior I8 parity, actual
covariance rasterizer backward, and forced physics smoke had passed before
formal training (`preflight_assessment.md`). Source freeze still verifies.

The final I8/I9 comparison additionally verifies identical extended step-zero
tensors/initial Adam, final optimizer groups, complete16k sampler/resolution
and RNG (`comparison16000.json`). This is one run per arm, not a statistical
robustness result. Frozen static parameters did not change: the failed metric
measures their composition with learned dynamics, not corruption of S1 tensors.

An independent float64 diagnostic covers all15,507 centers at the fixed nine
times. Relative map J(t) J(frame2)^-1 quantiles below use all rows, irrespective
of opacity, avoiding a changing opacity-selected population:

|Frame|I8 min-stretch1%|I9 min-stretch1%|I8 max-stretch99%|I9 max-stretch99%|
|---|---:|---:|---:|---:|
|62|0.1904|0.2828|6.3727|3.5732|
|86|0.2077|0.5936|5.2677|2.1457|
|108|0.1682|0.7028|5.1788|1.3413|

These measured distortion statistics improve, while appearance worsens and
completeness fails. Frame2 remains model-conditioned, not a known stress-free
configuration; no material/force/constitutive identification is established.

Three native1600x900 images were actually inspected: c188/t43, c188/t146 and
c86/t86. They retain large hollow/missing surfaces, separated flattened lobes,
upper thin sheets and fragments. The final same-crop I8/I9 comparison was also
actually inspected; both fail. Original native images and all hashes are
preserved. No six-by-nine passing screen, full1200-frame acceptance scan, test
RGB selection or22k matched result is claimed. The user goal is unachieved.

The evidence argues against treating lower strain as a solution to missing
geometry. The separate MonST3R pilot investigates additional observations;
its adjacent-pair outputs also fail qualification and are not training targets.
