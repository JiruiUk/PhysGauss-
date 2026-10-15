# CODEX_TASK.md — Executable implementation plan

## 0. Objective
Extend the official **Physics-Informed Deformable Gaussian Splatting (PIDG)** implementation while retaining PIDG as the base model.

The implementation must support the following experimental ladder:

- **A0** Original PIDG baseline
- **A1** PIDG + deformation-gradient estimator
- **A2a** A1 + constitutive consistency with fixed material parameters
- **A2b** A1 + constitutive consistency with learnable material parameters
- **A3** A2b + known multi-material support
- **A4** A3 + structured unknown force
- **A5** A4 + identifiability diagnostics
- **A6** Full model

Do not implement everything at once. Complete and validate each milestone before continuing.

---

# M0 — Reproduce PIDG baseline

## Tasks
1. Locate the original PIDG training entry point.
2. Locate all official benchmark configs/datasets.
3. Run the original baseline without modifications.
4. Record commands and metrics.
5. Create `PIDG_REPO_MAP.md`.
6. Create a baseline git tag or commit.

## Output files
- `PIDG_REPO_MAP.md`
- `baseline_commands.md`
- `baseline_metrics.json`

## Acceptance criteria
- Original training launches successfully.
- Original evaluation launches successfully.
- At least one development scene trains for a short smoke run without NaN/Inf.

---

# M1 — Differentiable deformation-gradient estimator

## Goal
Estimate local deformation gradient `F_i` from canonical and deformed Gaussian positions.

## Inputs
- canonical positions `X: [N,3]`
- current positions `x: [N,3]`
- canonical-space neighbor graph `neighbors: [N,K]`

## Output
- `F: [N,3,3]`
- validity/confidence mask
- optional condition number

## Method
For each Gaussian `i`, solve weighted local least squares:

`F_i = argmin_F sum_j w_ij || (x_j - x_i) - F (X_j - X_i) ||^2`

Implementation requirements:
- KNN computed in canonical space and cached.
- Distance-based weights.
- Regularized least squares.
- Prefer `torch.linalg.solve`/`lstsq`; avoid explicit inverse.
- Differentiable w.r.t. `x`.
- Degenerate neighborhoods must be masked/down-weighted.
- Compute `J_i = det(F_i)`.
- Never silently replace negative determinant with absolute value.

## Suggested config
```yaml
physics:
  deformation_gradient:
    enabled: false
    num_neighbors: 16
    regularization: 1.0e-4
    min_valid_neighbors: 6
    max_condition_number: 1.0e6
    min_determinant: 1.0e-6
```

## Unit tests
- identity: `x=X -> F≈I`
- uniform scale: `x=sX -> F≈sI`
- rotation: `x=RX -> F≈R`
- affine: `x=AX+b -> F≈A`
- autograd: finite gradients on `x`

## A1 experiment
Run PIDG with `F` computed only for diagnostics. Do **not** add any new loss.

Expected result:
- reconstruction metrics approximately match A0,
- `F` statistics are finite,
- invalid-`F` fraction is reported.

---

# M2 — Neo-Hookean constitutive model

## Goal
Compute a constitutive Cauchy stress `sigma_const` from `F`.

Implement a clean interface:

```python
class ConstitutiveModel(nn.Module):
    def forward(self, F, material_params):
        # return Cauchy stress [N,3,3]
        ...
```

Initial model:
- compressible Neo-Hookean

Use positive parameterization:
- `mu = softplus(raw_mu) + eps`
- `kappa = softplus(raw_kappa) + eps`

Support fixed or learnable parameters.

## Required properties
- `F=I -> sigma≈0`
- rigid rotation should not generate elastic stress
- isotropic expansion produces isotropic normal stress
- gradients finite w.r.t. `F` and material parameters

## Stress convention
If PIDG stores six symmetric components, add utilities:
- `vec6_to_symmat`
- `symmat_to_vec6`

Document component order explicitly.

---

# M3 — Constitutive consistency loss

## Goal
Keep PIDG stress prediction and compare it with constitutive stress.

Compute:

`L_const = mean_i confidence_i * Huber(sigma_pred_i - sigma_const_i)`

Do not replace the PIDG stress network.

## Config
```yaml
physics:
  constitutive:
    enabled: false
    type: neo_hookean
    learn_parameters: false
    mu_init: 1.0
    kappa_init: 10.0
    loss_weight: 0.0
    robust_loss: huber
```

## Logging
Log at minimum:
- `loss_constitutive_raw`
- `loss_constitutive_weighted`
- `mean_det_F`
- `invalid_F_fraction`
- `mean_sigma_pred_norm`
- `mean_sigma_const_norm`
- `mean_sigma_difference_norm`

## A2a
Fixed constitutive parameters.

## A2b
Learnable scene/material-level parameters.

Do not create one stiffness parameter per Gaussian.

---

# M4 — Fixed vs learnable material-parameter ablation

Run and compare:

- A2a: fixed `mu`, `kappa`
- A2b: learnable `mu`, `kappa`

Use the same PIDG initialization and random seeds where possible.

Report:
- reconstruction metrics
- Cauchy residual
- constitutive residual
- material parameter trajectory during optimization
- future prediction metrics if supported by the benchmark

Purpose:
Separate gains from adding a constitutive prior from gains caused by extra degrees of freedom.

---

# M5 — Known multi-material support

## Goal
Allow one scene to contain multiple known material regions.

Each Gaussian receives:
- `material_id in [0, M-1]`

Each material region shares one parameter set:
- `theta_0, ..., theta_(M-1)`

Implement:

```python
params = material_bank(material_id)
sigma_const = constitutive(F, params)
```

First version assumptions:
- material IDs are known,
- no learned segmentation,
- all materials may use the same constitutive family with different parameters.

## Config sketch
```yaml
materials:
  enabled: false
  regions:
    - id: 0
      model: neo_hookean
      mu_init: 1.0
      kappa_init: 10.0
    - id: 1
      model: neo_hookean
      mu_init: 2.0
      kappa_init: 20.0
```

## Tests
- two regions receive different parameter tensors,
- gradients flow only to the correct shared material entries,
- invalid material IDs fail loudly.

---

# M6 — Interface traction loss

For neighboring Gaussians with different material IDs, define interface pairs.

Use canonical shared normal:

`n_ij = normalize(X_j - X_i)`

Traction:
- `t_i = sigma_i @ n_ij`
- `t_j = sigma_j @ n_ij`

Loss:

`L_interface = mean ||t_i - t_j||^2`

Do not add a strong displacement continuity loss if both regions already share one continuous deformation field.

---

# M7 — Structured unknown external force

## Goal
Extend the Cauchy momentum residual with a structured unknown force.

Do not create an arbitrary free `f_i(t)` for every Gaussian.

Support:
- `none`
- `known`
- `surface_normal`
- `low_rank`

### surface_normal
`f_i(t) = mask_i * p(t) * n_i`

`p(t)` should use a low-dimensional temporal parameterization.

### low_rank
`f(x,t) = sum_r a_r(t) phi_r(x)`

Default rank:
- 1–8

Regularizers:
- force magnitude
- temporal smoothness
- optional sparsity

## Modified Cauchy residual
Use the existing PIDG implementation and insert `f_ext` only at the external-force interface:

`r = rho * DvDt - div(sigma) - f_ext`

Log:
- inertial term norm
- stress-divergence norm
- external-force norm
- total residual norm

---

# M8 — Staged optimization

Do not optimize all modules from iteration zero.

## Stage A — PIDG warm start
Train original PIDG variables. New material/force parameters frozen.

## Stage B — constitutive warmup
Enable `F` and constitutive loss with fixed material parameters.

## Stage C — physical parameter fitting
Freeze or strongly reduce learning rate for appearance/canonical geometry. Enable material parameters and/or force latent.

## Stage D — joint fine-tuning
Unfreeze selected modules with reduced learning rates for physical parameters.

Suggested hierarchy:
- standard PIDG neural variables: base LR
- force latent: ~0.1 × base LR
- material parameters: ~0.01–0.1 × base LR

---

# M9 — Identifiability diagnostics

## Goal
Diagnose whether low-dimensional physical parameters can be separated by the current observations.

Parameter vector example:

`theta = [log(mu_1), log(kappa_1), ..., force_latent_coefficients]`

Select a physical residual vector `r(theta)` and compute:

`J = dr/dtheta`

Use autograd / `torch.func.jacrev` where feasible.

Report:
- singular values of `J`
- minimum singular value
- maximum singular value
- condition number
- effective rank

Optionally:

`H ≈ J^T J`

`Cov ≈ (J^T J + lambda I)^(-1)`

Only use this for low-dimensional parameter vectors.

## Required ambiguity experiment
Compare:
1. stiffness unknown, force known
2. stiffness known, force unknown
3. both unknown
4. both unknown + priors/structure

Expected behavior:
- case 3 should show worse conditioning and/or strong stiffness-force correlation.

Do not hide ambiguity with a single confident point estimate.

---

# M10 — Benchmark matrix

Use `experiments/pidg_experiment_matrix.yaml` as the source of truth.

## Development tiers

### Smoke
Run one solid scene for a few hundred/thousand iterations.

Preferred current scene:
- Rubber Duck

### Development
Run:
- Rubber Duck
- Balls Reaction

### Final
Run the complete official PIDG benchmark suite verified from the repository:
- PIDG physics-driven scenes
- D-NeRF scenes
- HyperNeRF scenes

Do not apply Neo-Hookean constitutive loss to clearly incompatible fluid/smoke scenes.

---

# Required experiment IDs

## A0 — PIDG baseline
All new flags off.

## A1 — +F diagnostic
`F` enabled, no new training loss.

## A2a — +fixed constitutive consistency
Fixed `mu/kappa`.

## A2b — +learnable constitutive consistency
Learnable scene/material-level `mu/kappa`.

## A3 — +known multi-material
Known material IDs and shared parameter bank.

## A4 — +structured unknown force
Low-dimensional force representation.

## A5 — +identifiability diagnostics
Training may match A4; add Jacobian/post-fit diagnostics.

## A6 — full model
All supported modules enabled for compatible scenes.

---

# Required metrics

## Reconstruction
Use whatever official PIDG benchmark metrics are already implemented, typically including:
- PSNR
- SSIM / MS-SSIM
- LPIPS

## Kinematics
Add when GT/diagnostics are available:
- trajectory error
- velocity error
- deformation-gradient error
- invalid-`F` ratio

## Physics
- Cauchy residual
- constitutive residual
- stress norm/statistics
- interface traction residual

## Material
If ground truth exists:
- relative parameter error
- per-material parameter error

If ground truth does not exist:
- do not claim parameter recovery; report consistency/stability only.

## Force
If ground truth exists:
- magnitude error
- direction error
- temporal force error

## Identifiability
- Jacobian singular-value spectrum
- condition number
- effective rank
- parameter covariance/correlation
- especially stiffness-force correlation

---

# Result tables to generate

## Table 1 — PIDG compatibility
Columns:
- method
- PIDG benchmark reconstruction metrics
- D-NeRF metrics
- HyperNeRF metrics

Rows:
- official PIDG if numbers available
- reproduced PIDG
- A2b
- A6

## Table 2 — Physics ablation
Rows:
- A0
- A1
- A2a
- A2b
- A3
- A4

Columns:
- reconstruction metric(s)
- Cauchy residual
- constitutive residual
- future prediction metric(s), if benchmark supports them

## Table 3 — Identifiability
Rows:
- force known
- stiffness known
- both unknown
- both unknown + prior

Columns:
- parameter error, if GT exists
- force error, if GT exists
- `cond(J)`
- minimum singular value
- stiffness-force correlation

---

# Safety against false claims
- Do not report material recovery without material GT or a clearly defined reference.
- Do not report force recovery without force GT or a controlled synthetic protocol.
- Do not call a scene multi-material merely because different benchmark scenes use different materials.
- Do not call identifiability solved because a point estimate converges.
- If force and stiffness remain ambiguous, report the ambiguity explicitly.
