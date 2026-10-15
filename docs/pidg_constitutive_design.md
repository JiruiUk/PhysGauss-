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

## M2: constitutive convention

The initial law is compressible, isotropic Neo-Hookean elasticity with
`W=mu/2*(J^(-2/3)*tr(B)-3)+kappa/2*log(J)^2`, `B=F F^T`.
Its Cauchy stress is `mu*J^(-5/3)*dev(B)+kappa*log(J)/J*I`.
Here `kappa` is the bulk modulus. Positive shared parameters use softplus plus
epsilon; no per-Gaussian stiffness is introduced. Nonpositive J raises an error
in the constitutive module; the caller selects the explicitly reported valid F.
Six-component stress order is `[xx,yy,zz,xy,xz,yz]`, with tensor shear components.
M2 is a fixed-parameter stress diagnostic, with no constitutive training loss.

## M3: consistency branch and training state

The original velocity/stress network stays in place. Its six-component stress is
expanded to a symmetric matrix, including the original velocity-gradient correction
with `force_sigma`. The new loss is a confidence-weighted mean of componentwise
Huber distances from the Neo-Hookean stress. Invalid F receives zero weight.
Raw/weighted losses, valid fraction, both stress norms, difference norm and the
loss gradients with respect to F and predicted stress are recorded separately.
The extra loss backpropagates to deformation and the original stress prediction;
unit tests also check the shared material parameters and velocity-gradient term.

The new consistency branch queries predicted stress at `fid`, matching the F
computed from rendered deformation. The original CMR continues to query
`fid/train_count`; that pre-existing CMR convention is retained for baseline
compatibility and remains a limitation of the combined physical interpretation.
CMR log values retain the original sampled-loss scaling; they are not unscaled
continuum residual measurements. Initial modulus values use dataset units, with
no calibration to physical units or claim of parameter recovery.

Optional shared material parameters have a separate Adam optimizer (default
LR 1e-5). The original Gaussian/deformation optimizers remain in use. Complete
checkpoints additionally save extension configuration, shared material state,
its optimizer, and the cached canonical graph. Enabled extension restarts reject
checkpoints lacking these fields. Flags-off checkpoints keep the original format.
The wrapper finalizes the extension payload/checksum before the trainer exits;
pause verification must wait for process exit and validate the final manifest.

The first M3 integration trial used the legacy CMR time for the new branch. Its
logs remain in `runs/pidg_constitutive/m3_constitutive_fixed`; this trial was
superseded before acceptance by `m3_constitutive_fixed_fid`, with matching stress/F
times. The old trial is not used in the fixed-versus-learnable comparison.

## M5: known material regions

The shared bank has two positive parameters per region. Material IDs are supplied
as an integer array in initial-particle order, then indexed through original PIDG
particle IDs. Clones inherit their parent's ID and pruning preserves remaining IDs;
the extension does not learn segmentation or assign per-Gaussian stiffness.
Annotation hashes and labels accompany the extension checkpoint.

Official scene material annotations have not been supplied. A separate manufactured
two-region dataset tests the integration, using 128 material points, prescribed
Neo-Hookean parameters, affine maps per region, equal normal interface traction,
and a known inertial body-force density with unit density. Camera calibration is
stored separately; images are projected colored discs. This is an engineering
fixture, not a new official benchmark or photorealistic physical simulator.
Its initial shared moduli are deliberately 20% below the prescribed values.
Future-time images are held out, but official PIDG's test split is not relabeled
as a future-prediction benchmark. No official-scene material recovery is claimed.
