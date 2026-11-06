# Closed Gaussian mechanics implementation

The remaining phases were explicitly authorized together on 2026-10-15. They are
implemented in independent frozen variants and enabled incrementally by JSON
switches. A phase is not a completed experiment until its five long runs finish;
scientific acceptance is a separate review of those results.

## Geometry and derivatives

The actual trajectory is `Phi(X,t) = R(q(X,t))X + b(X,t)`, including the rotational
term in velocity and acceleration. Scalar time is expanded to independent `[N,1]`
coordinates before per-particle differentiation. No finite-difference acceleration
is used for physics. Derivative helpers assume no batch-dependent layers.

The released CUDA hash custom backward has no complete second derivative with
respect to input coordinates. `physics/hash_autograd.py` reproduces the released
cubic smoothstep interpolation and unsigned hash indexing with ordinary Torch
operations, sharing the original embedding tables. Only small mechanics batches
take this path; appearance retains the original CUDA encoder/rasterizer. A GPU
preflight checks both paths and parameter gradients through trajectory acceleration.
Smoothstep is C1 across cell boundaries, not globally C2; weak mechanics and
explicit gradient checks are particularly relevant for acceleration residuals.

## Phase 4 stability revision

The initial unconstrained hash trajectory produced 20 inverted Jacobians among 64
sampled particles at sanity iteration 100. Those sources and runs remain available
as revision 0. Revision 1 adds two disclosed trajectory constraints from Phase 4:

- Center motion is anchored by `motion = identity + t*(motion_raw-identity)`,
  so `Phi(X,0)=X`, `F(X,0)=I`, and `J(X,0)=1` exactly.
- Canonical hash levels are tapered by `(resolution_0/resolution_level)^2`.
  Both RGB and mechanics use the same tapered representation; no detached or
  alternative trajectory is used for physics. This is a representation change,
  not a pure F diagnostic ablation; account for it when interpreting comparisons.

The revised sanity sample had zero inversions, minimum J approximately 0.484.
This is not a guarantee for other times, points, or long runs. Original invalid
counts are logged. A positivity barrier trains all sampled particles; material
evaluation explicitly excludes invalid particles and reports their count. Density
itself rejects non-positive J and is never silently clamped.

## Mechanics

`v=partial_t Phi`, `a=partial_t^2 Phi`, `F=partial_X Phi`, `J=det(F)`.
Mass is closed by `rho=rho0/J`. Consequently the soft mass residual is an identity
up to rounding; it is not an independently recovered physical density error.
Volume preservation is an independent optional switch and is off for the mixed
benchmark scenes without an incompressibility assumption.

Linear stress is `lambda tr(epsilon)I + 2mu epsilon` with
`epsilon=(F+F^T)/2-I`. Neo-Hookean stress is
`[mu(FF^T-I)+lambda log(J)I]/J`. E uses softplus, nu uses `0.49*sigmoid`.
Phase 6 trains linear elasticity first, then switches to Neo-Hookean at iteration
25,000; the short sanity uses an earlier transition to exercise that path.
Both constitutive laws have identity/shear/energy-gradient tests.

Phase 7 predicts canonical E/nu and optional material features with a smooth MLP,
SH appearance features detached as conditioning, and a neighborhood prior.
Viscosity is parameterized for extension but the initial experts are elastic;
do not report it as an identified viscous coefficient.

The strong residual uses the correct reference-to-spatial chain rule:

`div_x(sigma)_i = sum_{j,K} partial_XK(sigma_ij) (F^-1)_Kj`.

The weak residual integrates `rho*a*phi + sigma*grad(phi) - f*phi` using Gaussian
tests and reference quadrature volume `1/N`, current volume `J/N`. kNN selects
local quadrature support and bandwidth; neighbor selection is fixed in backward.
Boundary traction quadrature is supported by the API. Benchmark presets have no
provided traction and assume a zero boundary contribution. This approximation and
the dimensionless reference volume must be disclosed; Gaussian sample volumes are
not calibrated physical volumes.

Phase 10 mixes objective Neo-Hookean elastic and near-rigid (10x stiffness)
experts using a canonical gate. Phase 11 adds a symmetric neural stress correction
with epsilon=0.01 and a structural per-particle norm bound below 0.2 of physical
stress, plus a residual penalty. Gravity, contact, actuation and residual force
inputs are explicit. Contact uses a configurable plane SDF with a penalty force.
The benchmark preset leaves gravity off and uses offset -1.3 in dataset units;
this is a disclosed scene prior, not a measured floor or SI calibration.

## Flow and velocity switching

The old velocity head remains available throughout. Phase 2 first adds a velocity
consistency loss. The gated source can switch flow to trajectory velocity after
three consecutive sampled median relative mismatches below 10%. A short-run
mismatch above this threshold is not marked as passing Stage 2B. Closed momentum
always uses trajectory velocity/acceleration, regardless of the legacy flow gate.

Phase 13 adds bounded spatial log variance and Gaussian flow NLL. It uses occlusion,
raw image endpoint bounds, predicted endpoint bounds, motion masks and valid
current/next rasterizer projections. Errors/variances use normalized image units
with component order `[width,height]`; diagnostic EPE remains in pixels. This also
avoids the released `calculate_flow_loss` routine's HWC row-wise in-place scaling.
That legacy behavior is preserved for earlier variants and the original baseline.
The resulting comparison includes changed masks/normalization as well as NLL.
Flow is pseudo supervision; train-flow EPE is not held-out ground-truth flow error.

## Optional motion basis and forecasting

Phase 14 replaces the position warp with canonical-MLP-predicted cubic B-spline
controls in se(3), exponentiated to SE(3). It is an explicitly labeled Lie-algebra
spline, not a cumulative/geodesic group spline. Subtracting the t=0 spline enforces
identity at canonical time. The old deformation network remains in the source and
is still available via `motion_source=hash`. Spline queries cover [0,1]; forecasting
uses the physical integrator instead of silently extrapolating spline knots.

Phase 15 uses symplectic Euler with reference masses and stress forces. Deformation
gradients are fitted from reference/current neighbor offsets, and stress forces
are pairwise antisymmetric to conserve total internal linear momentum. A small
four-step trajectory consistency objective begins at iteration 25,000. Gravity and
contact are explicit in rollout too. Translation, harmonic energy bounds, reference
fit and internal-force momentum conservation have tests. No real future-trajectory
GT is available, so extrapolation accuracy cannot be fabricated from train loss.

## Curriculum and checkpoints

RGB is active from the start; legacy deformation/flow warmup is retained. Kinematic
closure ramps from 3k, local mechanics from 5k, strong momentum from 10k, weak form
from 15k, advanced regularizers/contact from 20k, and rollout from 25k. Physics is
sampled every 100 iterations with 64 collocation particles. This is an explicit
compute choice for higher derivatives, different from legacy 20k CMR sampling.
Log fields unavailable in a stage are null, not fabricated zero measurements.

New full checkpoints include mechanics modules, gate counters, both optimizers,
spline/hash/deformation state, camera stack and RNG states, with a SHA256 sidecar.
A real-CUDA restart test compares the next step against uninterrupted execution.
Appearance snapshots also carry mechanics weights so renderer loads are strict.
As in the legacy trainer, the terminal iteration skips optimizer step; terminal
checkpoints must not be interpreted as an exact continuation to a larger budget.

Gaussian covariance still follows the released scale/rotation heads. The mechanics
closure concerns particle centers; affine covariance transport `F Sigma F^T` is not
silently claimed by these variants. Absolute material identification additionally
needs calibrated units, loads, boundary conditions and suitable ground truth.
