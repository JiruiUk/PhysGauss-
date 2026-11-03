# I8: transporting covariance with position did not restore completeness

The original 16,000-step warmup completed from the exact S3-U step-zero state,
using the original 12k at width800 + 4k at width1600 schedule. The shared I7
position map also transports each canonical Gaussian covariance by its exact
spatial differential. No independent original grid/deformation heads are active
in this arm; their tensors and Adam groups remain present. No new observation,
material, force or strain objective was added.

Validation30: full PSNR24.946294, dynamic PSNR23.875979, dynamic alpha0.982865,
static PSNR32.815685. Static-only PSNR35.643816 gives a 2.828131dB drop and
fails the unchanged 1dB warmup guardrail. **Formal physics updates: 0.**
`warmup_admission.json` remains the original failed numerical receipt; a separate
`visual_review.json` records actual image inspection. No 22k result, full1200
acceptance scan or test-RGB evaluation is claimed.

Final checkpoint SHA256:
`f7ce6470086cf441515fca27a0eda50d1b8079d3576ff5448086e53557133903`.
`warmup_state_audit.json` passes: original parameter initialization and optimizer
groups, exact16k camera sampler/RNG, finite Adam state, inactive original heads,
and unchanged static parameters. CPU tests and native disabled-path/gradient,
covariance-only rasterizer-backward and forced physics smoke checks passed
before formal training. State correctness is not geometry success.

Actual native1600x900 images at fixed188/t43, fixed188/t146 and fixed86/t86
were inspected. They show hollow/missing surfaces, elongated folded sheets,
detached support and a long thin bridge. These independently reject completeness.
The three exact inspected image hashes are recorded in `visual_review.json`.

The final independent float64 differential audit covers all15,507 centers at
the nine preregistered times (`warmup16000_strain.json`). For opacity>0.1 rows,
the relative map J(t) J(frame2)^-1 has min-stretch1% of0.190/0.210/0.171 and
max-stretch99% of6.516/5.336/5.495 at frames62/86/108. At frame62,4.92% have
min stretch<0.25 and5.25% have max stretch>4. Unlike I7's measured position
map, I8 has substantial local distortion. The reference is model-conditioned,
not a known stress-free material state. Determinant positivity is insufficient.

This result supports testing a direct pointwise relative-Jacobian deformation
prior as a separate new arm. It does not justify declaring constitutive stress
identified, repeating graph constraints across uncertain object boundaries, or
relaxing static/appearance/completeness acceptance. Correct support transport
alone neither supplies missing geometry nor establishes material identity.
