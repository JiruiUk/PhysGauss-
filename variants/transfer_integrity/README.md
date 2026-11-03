# Transfer integrity prototype

This is an optional reconstruction/correspondence experiment on repaired PIDG
C1, not a constitutive-physics milestone. It introduces no material model,
force, deformation-gradient estimator, or learned segmentation.

`transfer_integrity_enabled` defaults to false. `prepare.py` copies the C1
runtime and changes only its Tuning import; the parent runtime and historical
states are preserved. `runtime.py` delegates to C1 when disabled.

The enabled branch adds training-view alpha coverage, time-independent static
position/covariance consistency, conservative image-triangulated static
anchors, frozen short-track reprojection, and tolerant same-component local
edges. Extra deformation evaluations preserve attention and random-state
behavior. The static prior covers part of the blue structure; dynamic links
terminate at each independently checked short clip. These are conditional
constraints, not physical point identities or dense novel-view ground truth.

The predeclared protocol is `experiments/transfer_integrity/plan.json`.
Data preparation and its checksums are in `data_binding.json` and
`input_lock.json`. `run.py` tests actual gradients/updates, ten-step C1 parity,
and enabled smoke/resume before matched 2,000-update continuations from the
same complete 50k C1 state. All source, optimizer, RNG, camera sampler, and LR
states are retained; topology is fixed. Only the new main flag differs.

```powershell
. .\reproduction\scripts\enable_cuda.ps1
envs/pidg-cu128/python.exe -m variants.transfer_integrity.run --preflight-only
envs/pidg-cu128/python.exe -m variants.transfer_integrity.run
envs/pidg-cu128/python.exe experiments/transfer_integrity/evaluate.py
envs/pidg-cu128/python.exe experiments/transfer_integrity/export.py
```

Evaluation uses final complete states, 30 validation plus 50 test observations,
three fixed cameras over all 200 times, and the original moving trajectory.
Novel camera/time combinations have no independent ground truth; shape and
color-subset measurements cannot establish true 3D completeness or identity.
See the assessment for actual outcomes, including regressions and limitations.
