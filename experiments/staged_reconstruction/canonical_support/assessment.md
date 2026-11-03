# Observed coverage is high despite failed new-view geometry

Read-only audit of all120 observed training frames at width800. All model
tensors and the three exact checkpoints are unchanged; no training/density
updates or test RGB. Per-point arrays, checkpoint/source hashes and per-frame
records are in the corresponding JSON receipts. The three models have different
budgets/architectures and are diagnostic cases, not a matched-budget ablation.

|Model|Active opacity>0.1 points|Mean dynamic alpha on eroded observed foreground|Mean fraction of those pixels with alpha<0.9|
|---|---:|---:|---:|
|S3-U selected final6k|12,882|0.997267|0.4371%|
|I8 failed final16k warmup|10,563|0.992415|1.2225%|
|I9 early step1000 only|15,171|0.795065|23.2052%|

Every dynamic row has nonzero image view-space gradient in at least one of
the120 views. None reaches the existing canonical coordinate displacement
clamp. Among active points,30.63%/36.38%/19.72% exceed the original3DGS
gradient threshold0.0002 for these respective models. That uncalibrated
threshold is descriptive only: a large gradient also responds to appearance,
misplacement, static interference and over-reconstruction. It does not identify
missing true surfaces or automatically qualify densification.

S3-U and I8 already cover almost all observed foreground while their actual
new-view images fail completeness. This weakens the explanation that an
inability to cover observed training pixels is the principal bottleneck. It
does not prove that all3D support is sufficient or that density adaptation would
not help. I9 step1000 is still an early underfit snapshot, not evidence that
its prior improves or worsens final coverage.

Native I9 step1000 dynamic-only alpha at c188/t146 and c86/t86 was also
inspected. Detached/flattened support and the long sheet with gaps persist
without view-dependent color and static compositing. This narrows the artifact
to dynamic support/geometry/opacity rather than only RGB appearance. It does
not identify an exact material surface or a unique cause.

Decision: preserve live I9 and finish its original admission protocol. Do not
blindly enable point growth on the strength of screen gradients. If the final
model still fails, distinguish canonical shape/initialization and motion-depth
ambiguity from population capacity in a separately preregistered experiment.
Original static/appearance/native completeness gates remain unchanged.

Figure: `output/staged_reconstruction/canonical_support/support_diagnosis.png`.
Primary literature and repository paths: `research.md`.
