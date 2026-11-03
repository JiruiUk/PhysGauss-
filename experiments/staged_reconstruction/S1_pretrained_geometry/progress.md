# S1-D3 pretrained geometry

D2 archived6751b566; user explicitly selected fixed DA-V2 Small. Official source
and99,218,434-byte weights downloaded and SHA256 checked against HF LFS metadata.
Pinned code/weight revisions and license retained in upstream_receipt.json and
external/depth_anything_v2_small/source/LICENSE. Existing environment unchanged.
No predictions yet. Fixed plan/protocol precede inference; only this candidate.

Network first failed sandbox sockets, then Windows certificate-store decoding;
public download succeeded using installed certifi CA verification. No TLS
verification disabled. No automatic-approval rejection occurred.

Next implement/tests/freeze, then serialized two-case inference and independent
GT audit. No full Gaussian/real continuation is authorized by this evidence
stage. Check actual process, stop.request and lock before launch; do not restart
completed prior stages or touch concurrent unrelated training.

## Fixed two-condition audit completed

27 tests passed; frozen sources28ebb9f4...; all192 outputs and original tight
anchors replay exactly. Independent process reload of the official weights
reproduces both first-view predictions exactly. Model24,785,089 parameters,
FP32 official518 preprocessing, no finetuning/augmentation/dependency changes.

Only1/96 images per case calibrated (c26_t2);49 lack anchors and46/45 fail held
anchor checks, plus one glossy degeneracy. Even accepted view visible-static
MAE0.11596/0.11186, and100% of its true empty background receives finite depth.
Direct static-use gate rejected both. Original frozen evaluation/gates retained.
Raw relative ordering Spearman mean0.8957/0.8959 motivates a separate diagnosis,
not direct depth/normal supervision. All fixed and sole-accepted-view figures
actually reviewed. See v1/assessment.md. No Gaussian/real training or new86/188.

Next fixed diagnostic separates relative-order reliability from metric-shape
error with evaluator-only oracle alignment; no GT coefficient/selection may
enter training. Do not repeat D3 or relax its anchor/coverage/depth gates.

## D3-O independent shape/order diagnosis completed

Preregistered3fac945b, implemented/frozen5c311055,20 tests. Privileged GT affine
alignment still gives held-visible MAE0.07223/0.07215, thin0.08082/0.08171;
same D1 pixels0.06779/0.06766 versus D1's0.03565/0.03569. GT-fit coefficients,
depth arrays and masks remain evaluator-only and are not saved as assets.

Strong pair static order errors2.23%/2.16%, thin8.76%/8.60%. Only7.42% of strong
pairs have two visible-static endpoints;91.24% include true empty background.
Thus high whole-image rank is not safe static or thin supervision. Fixed image
panels and explicit matched-pixel supplements actually viewed in both cases.
No PIDG/Gaussian training, candidate/weight/threshold change or old stage resume.

Continue only one RGB-validated ordinal depth-proposal candidate, separately
preregistered under S1_rgb_prior_proposals. Use learned ordering to propose an
alternative RGB depth basin, not as a loss or static classifier; retain every
original candidate and accept only fixed fit/held RGB checks. No GT fit/mask or
coefficient may enter it. New Gaussian stage remains separately gated.
