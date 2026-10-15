# PIDG constitutive extension

This implements `CODEX_TASK.md` on an independent copy of the disclosed original
PIDG baseline in `variants/constitutive_pidg/code`. Original baseline outputs and
`third_party/PIDG` remain unchanged. Optional modules are in `pidg_ext`.

## M1: local deformation gradient

Canonical-space KNN is cached, refreshed every 1,000 iterations and rebuilt whenever
densification/pruning changes the point count. The local fit uses current canonical
positions, rather than interpreting quaternion/translation outputs as displacements.
Rendered centers are reconstructed with the original quaternion utility.

For each query, let `dX = X_j-X_i`, `dx=x_j-x_i`, and let distance weights be
`w=exp(-||dX||²/mean(||dX||²))`. Define
`A=sum(w dX dX^T)` and `B=sum(w dx dX^T)`. Solve
`(A+lambda I) F^T = (B+lambda I)^T` using `torch.linalg.solve`.
The ridge term is scaled by `trace(A)/3`, with an identity prior. This makes
identity exact while introducing a small, explicit regularization bias for other
affine transforms. Unit tests use a smaller ridge to test affine accuracy.

F remains differentiable with respect to deformed centers. Validity uses the
unregularized canonical moment matrix, the count of nonzero neighbour offsets,
condition number and the **signed** determinant. Negative determinants are logged
and invalidated, never converted to absolute values. Confidence is detached so a
future constitutive loss cannot reduce itself by changing its own confidence.

A1 evaluates deterministic query IDs every 100 iterations under `no_grad`; it
consumes no training RNG and adds no loss. Disabled mode does not build a graph,
create parameters, consume RNG, or write diagnostic records. The original RGB,
velocity, stress, CMR and optional flow paths stay in place.

## Validation levels

The five completed 50k A0 runs are reused. New short integration runs are labeled
`engineering_smoke`, width 400, seed 0, no densification, early physics warmup.
These short runs cannot establish improvements over the 50k reference.
Each run captures code, extension, config and dataset hashes, commands, official
render/evaluation output and a checksum-verified optimizer/RNG checkpoint.

The original tree has pre-existing Windows/release compatibility fixes; M0 captures
them explicitly. A baseline tag records the audit, patches and measured metrics;
it is not a claim that the historical runtime source was fingerprinted.
