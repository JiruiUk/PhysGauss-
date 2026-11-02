# Corrected flow / shared deformation stress follow-up

L0_corrected_trajectory and L1_shared_stress use trajectory velocity and the same
corrected raw training-flow pipeline. L0 keeps the independent PIDG stress field;
L1 computes stress6 from the exact existing deformation Grid4D and shared trunk
latent. `stress_source` is their only runtime-config difference. The old physical
network is retained but frozen in L1; the unused new head is frozen in L0.

The optional `unified_stress_head=False` architecture default creates no new
parameters. Existing forward remains unchanged. Enabled runs add Linear(256,6),
seed20261102/std.001/bias0 without consuming trainer RNG. Both arms receive the
same extended full3k checkpoint and optimizer layout, with an empty new-head
optimizer group at the original velocity LR. All inherited tensors, optimizer
state and RNG are checked against the original. No extra fitting/distillation.

The raw source→target camera flow uses only120 training RGB/RGBA and119 actual
chronological pairs. It reuses hash-verified UniMatch forward/backward estimates,
builds new train-alpha/FB masks, and never uses H3 depth. Actual scene camera FoVs
are used, preserving the inherited loader convention. The CUDA `x_mu=mean-pixel`
offset is negated; pre-update alpha compositing weights are reconstructed from
opacity/conic because CUDA metadata writes weights after updating transmittance.
These contracts have CPU affine and real CUDA tests.

This correction is common to L0/L1. K2→L0 also changes flow target/projection and
weights, so its improvement cannot be attributed to sharing. Both keep weights,
finite-difference steps, local Lagrangian momentum, original rate correction,
image objectives, time split and budget fixed. A shared latent is not a material
law or evidence of correct stress. Large acceleration-gradient imbalance may
remain; this comparison intentionally does not simultaneously tune its weight.

Before full training:15 tests, two100-step GPU smokes with70→100 full resumes,
flags-off20-step smoke, and separate actual optimizer-step checks passed. Shared
head/trunk gradients and snapshot/optimizer reload are covered. Full checkpoints
and inference snapshots include the new head. Evidence is in
`experiments/unified_flow/{preflight,optimizer_update_checks,prefix_extension_checks}.json`.

Run with the existing CUDA/MSVC environment:

```powershell
. ./reproduction/scripts/enable_cuda.ps1
envs/pidg-cu128/python.exe -u -m variants.unified_flow.run --hours 12
```

Do not edit frozen source/config/cache after launch. Both arms finish50k before
validation selection locks and final test opens. The coordinator exports pixel-
verified GT|Render PNG/video/ZIP, runs the same corrected raw validation-flow
diagnostic, verifies unused weights, and writes a report. Historical models and
original PIDG files are unchanged. No external force, constitutive law, material
unknowns, segmentation or per-Gaussian force/stiffness is introduced.
