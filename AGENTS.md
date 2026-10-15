# AGENTS.md — PIDG Extension Project

## Project goal
Use **Physics-Informed Deformable Gaussian Splatting (PIDG)** as the base model. Preserve the original PIDG behavior when all new features are disabled, then incrementally add:

1. deformation-gradient estimation `F`,
2. constitutive consistency,
3. optional learnable material parameters,
4. known multi-material support,
5. structured unknown external force,
6. identifiability diagnostics.

The first implementation target is **not** PhysGauss-ILU. PIDG is the baseline.

## Non-negotiable rules
- Do not rewrite the PIDG baseline.
- Every new module must be behind a config flag.
- With all new flags disabled, training/inference should follow the original PIDG code path as closely as possible.
- Do not introduce per-Gaussian free stiffness or per-Gaussian free force in the first version.
- Do not implement learned material segmentation in the first version.
- Do not use a solid constitutive model on smoke/fluid scenes.
- Add tests before integrating each new physics module into full training.
- Commit each milestone separately.

## Repository audit first
Before modifying code, inspect the repository and map these PIDG concepts to actual files/classes/functions:

| Concept | Actual code location |
|---|---|
| training entry point | TODO |
| config system | TODO |
| Gaussian model | TODO |
| deformation field | TODO |
| velocity field | TODO |
| stress/material field | TODO |
| Lagrangian trajectory / flow loss | TODO |
| Cauchy momentum residual | TODO |
| renderer | TODO |
| dataset loaders | TODO |
| metrics/evaluation | TODO |
| checkpoint save/load | TODO |

Record the mapping in `PIDG_REPO_MAP.md`.

## Required baseline reproduction
Before adding features, reproduce the original PIDG benchmark protocol available in the repository.

Current experimental plan assumes the PIDG project contains:
- PIDG physics-driven synthetic benchmark,
- D-NeRF benchmark,
- HyperNeRF benchmark.

Do not trust these names blindly. Verify dataset/config names from the checked-out PIDG repository and update `experiments/pidg_experiment_matrix.yaml` if needed.

## Development scene priority
For solid constitutive development, prefer scenes that are clearly soft/elastic solids.

Current planned development priority:
1. Rubber Duck — soft-body development scene.
2. Balls Reaction — elastic/interaction development scene.
3. Mechanics Cloth — use only for general reconstruction until a cloth/shell constitutive model exists.
4. Motion Kuro — reconstruction/generalization only.
5. Dry Ice — reconstruction/generalization only; do not apply Neo-Hookean solid constitutive loss.

If the official PIDG repository uses different scene names, replace the names with the verified equivalents while preserving the intent above.

## Physics state
For Gaussian/material point `i`:

- `X_i`: canonical/material coordinate
- `x_i(t)`: deformed position
- `u_i(t) = x_i(t) - X_i`: displacement
- `v_i(t)`: velocity
- `F_i(t)`: deformation gradient
- `sigma_pred_i(t)`: PIDG-predicted Cauchy stress
- `sigma_const_i(t)`: constitutive stress computed from `F_i`
- `m_i`: material ID
- `theta_m`: material parameters shared by material region

Keep `sigma_pred` from PIDG. Do not replace it initially.

## Core extension
Add an independent constitutive branch:

`X, x(t) -> F(t) -> sigma_const(F; theta)`

and enforce consistency with PIDG stress:

`L_const = robust_distance(sigma_pred, sigma_const)`

This must be independently switchable.

## Milestone order
Follow this order exactly:

- M0: reproduce PIDG baseline
- M1: differentiable deformation-gradient estimator + unit tests
- M2: Neo-Hookean constitutive model + unit tests
- M3: constitutive consistency loss
- M4: fixed vs learnable material-parameter ablation
- M5: known multi-material material bank
- M6: interface traction loss
- M7: structured unknown-force module
- M8: staged optimization
- M9: identifiability diagnostics
- M10: full benchmark/ablation scripts

After every milestone:
1. run unit tests,
2. run a short training smoke test,
3. verify baseline flags-off behavior,
4. save metrics/logs,
5. commit separately.

## Definition of done
The project is not considered successful until:
- PIDG baseline can still run unchanged with new flags off,
- affine synthetic tests recover `F`,
- rigid rotation produces approximately zero Neo-Hookean elastic stress,
- constitutive loss backpropagates into the intended variables,
- multiple material regions use distinct shared parameter sets,
- structured force receives finite gradients,
- ambiguous force/stiffness settings produce poor identifiability diagnostics rather than falsely confident estimates,
- experiment configs A0–A6 are independently runnable.
