# Same-budget7000-step I8/I9 comparison: less distortion, still incomplete

Pinned I9 snapshot SHA256:
`3ddcdea111c9bf94eb014647855e35b631f4d879121acc6bbbaa3df77c6a5ccc`.
Pinned I8 snapshot SHA256:
`6a498afc8b694bfaab94ed05927d11dd5fe82279f7baa1ba1eb00bd3a21d3994`.

The read-only comparison verifies exact extended step-zero model tensors and
initial Adam configuration, identical optimizer groups, complete first7000
camera samples/resolution schedule and checkpoint Python/NumPy/CPU/CUDA RNG.
Both runs have zero formal physics calls at this point. This is the same I8
architecture with one extra relative-strain objective, subject to CUDA numerical
variation; it is a single-run ablation, not a statistical robustness result.
Receipt/source: `comparison07000.json`, `scripts/compare_transport_previews.py`.

The independent float64 audit covers all15,507 centers at the fixed nine times.
For the relative map J(t) J(frame2)^-1, the following quantiles use **all rows**
to avoid changing the sample population through an opacity threshold:

|Frame|I8 min-stretch1%|I9 min-stretch1%|I8 max-stretch99%|I9 max-stretch99%|
|---|---:|---:|---:|---:|
|62|0.217|0.265|4.546|3.551|
|86|0.183|0.612|3.669|2.066|
|108|0.143|0.654|5.865|1.400|

Measured relative distortion is reduced in these cases. Outliers remain at62:
I9 all-row fractions min<0.25 and max>4 are0.761% and0.548%. Counts above
opacity0.1 also differ (I8:12,212; I9:10,151); their separate distributions and
render covariance anisotropy remain in the full receipts. Frame2 is a learned
configuration, not stress-free GT. These statistics do not certify every
continuous time, surface point, physical identity or constitutive stress.

All three native1600 I9 RGB images were actually inspected, together with the
same-crop I8/I9 comparison and enlarged c86/t86 comparison. c188/t43 andt146
still have major hollow/missing surfaces and detached or flattened support.
c86/t86 retains a long sheet/bridge and separated upper fragments. Both arms
fail completeness at this intermediate point. This is not repaired by showing
lower strain or a contiguous part of the silhouette. No new-view synchronized
GT, test RGB, image inpainting, early model selection or protocol change.

Original native images are preserved. The comparison applies the same crop
and resize within each row; crop boxes and every input/output hash are recorded.
`preview07000_review.json` records exact inspected files. The live I9 training
continues toward its original16k numerical/image admission and conditional6k
physics; no final appearance or success claim is made here.
