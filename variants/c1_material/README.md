# C1 / fixed E / learned E

Three total50k runs on Rubber Duck, seed0,120 train/30 val/50 fixed test.
P0 is a newly repeated C1 control. P1 adds constitutive consistency with fixed
Young modulus E=1; P2 learns exactly one shared E while using the identical loss.
Poisson ratio stays0.3. E uses a bounded sigmoid in[0.05,20]. Units are normalized
effective stiffness, not calibrated Pascals. No per-Gaussian physical parameter,
learned material segmentation, interface module or free external force is added.

The dataset lacks true material-region labels and contains several ducks and
obstacles. This is explicitly a **global effective material prior ablation**,
not true single-material recovery or a correctly labeled multi-material model.
Canonical proximity neighborhoods are unsegmented. The rest configuration is
the learned canonical geometry, not a verified stress-free physical reference.

Retain the frozen C1 color/mask/foreground recipe, original optimizer, standard
densification and shared frame/physics clock. No isotropy or kinematic term.
Copy the same complete C1 random-init3k checkpoint to all arms, including the
control; do not continue the existing final50k weights for extra iterations.
The material parameters/optimizer initialize only at this pre-activation branch.

F uses the tested16-neighbor weighted affine estimator. Its canonical reference
basis is detached; the complete R(q)X+b trajectory remains differentiable into
the deformation network. Keep the PIDG predicted Cauchy stress, including its
force_sigma strain-rate correction, and compare it to the existing objective
compressible Neo-Hookean law with mu=E/[2(1+nu)], kappa=E/[3(1-2nu)].

Constitutive Huber weight1e-4, sampled every100 steps with100x interval
compensation; start after3k and ramp2k. Query256 anchors. Invalid/inverted or
ill-conditioned local F is masked by the existing estimator. A weak1e-3
log(E/E0)^2 prior sits inside the same compensated objective. Prior loss is zero
for fixed E. The learned scalar Adam rate is1e-4. All knobs are predeclared.

Neural stress, unknown loads, normalized time, unknown density calibration and
free canonical geometry confound material identification. Conditional E-loss
profiles freeze the neural fields and explicitly **do not prove identifiability**.
Log E, prior, F validity, stress/F gradients and material updates. Save E optimizer
and exact canonical graph in checksum-bound full-checkpoint sidecars.

Tests precede integration. Preflight includes flags-off C1 loss/RNG/renderer
parity, synthetic affine/rotation/gradient tests, state and tamper tests,20-step
flags-off training,100-step all-arm training, learned-E70-step resume, and30-val
PNG parity. Smoke uses reduced engineering schedules only. Main training starts
only after the current frozen-source preflight passes.

Validation alone selects four snapshots. Lock all arms before final test. Export
four official metrics, alpha diagnostics, individual render PNGs, GT/render pairs
and GT/P0/P1/P2 four-panel comparisons. Monitor refreshes every5 seconds.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File variants/c1_material/launch.ps1 -Smoke
powershell -NoProfile -ExecutionPolicy Bypass -File variants/c1_material/launch.ps1
```
