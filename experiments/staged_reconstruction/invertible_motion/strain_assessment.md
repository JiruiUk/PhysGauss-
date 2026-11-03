# I7: local position strain does not explain the missing geometry by itself

Read-only diagnosis of the actual rejected16k checkpoint
`e5eb8b708051a757112a60a9e58f27cea616ab290a4e1d589012c94f15c05190`.
No optimizer, renderer, image input or existing experiment source was modified.
All15,507 canonical rows were evaluated at the original nine screen times in
float64, using the exact spatial derivative of the independent affine-coupling
map. Analytical log determinants and fixed32-row central differences agree.
Five mathematical tests cover batch Jacobians, rigid rotations, determinant
versus shear, relative-gradient multiplication order and coupling derivatives.

Across these sampled points/times, spatial-map principal stretches span roughly
0.353–3.358. Relative to the earliest training frame2, they span roughly
0.330–3.120. No sampled point has a minimum stretch below0.25, a maximum above4,
or a negative determinant. This is descriptive evidence, not a new admission
threshold; the complete distributions and numerical checks are in
`strain_diagnostic.json`, with hashed per-point arrays in the runs directory.

These results do not demonstrate a physical deformation field or good geometry.
The derivative is evaluated at Gaussian centers, not throughout the volume or
all200 times. Frame2 is a model-conditioned configuration, not a known relaxed
state. Nonetheless the earlier hypothesis of severe local center-map collapse
is not supported at the measured rows/times. Do not add stronger strain/stiffness
penalties merely because the rendered surface is incomplete.

The center differential does not transport the independently predicted Gaussian
covariance. Diagnose that rendering support and the canonical surface before
another physics-training objective. I7 remains rejected; no formal physics
updates, test evaluation or full1200-frame acceptance is admitted.
