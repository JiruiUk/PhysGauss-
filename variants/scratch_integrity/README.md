# Cold-start transfer integrity (version 2)

Independent reconstruction/correspondence experiment, not a constitutive
physics milestone. The repaired C1 recipe is retained. There is no learned
material segmentation, new stress law, deformation gradient, or external force.

`prefix.py` uses the unchanged C1 trainer to capture a genuinely untrained
step-zero state (empty optimizer states), retains it, and trains a common
prefix with seed20261103. The dataset's original train-only untrained point
cloud initialization is retained. No historical trained Gaussian/network/
velocity/stress state is loaded. Full optimization uses50,000 actual updates,
including the common prefix, with the original LR schedule and no reset.

The original C1 densification/pruning stops before15k and its optional later
coverage cloning is disabled. `data.py` recomputes complete native-resolution
contributions and row bindings from an immutable copy of the new common state.
Only independent training images, image-component continuity, LK triangulation
and frozen raw optical flow are reused. Historical C1 Gaussian row IDs are not.

Activation eligibility is checked at predeclared15k/20k/25k/30k common boundaries
using the unchanged v1 reliability gates. Failures are retained. Validation or
test scores never select activation. No passing candidate is an explicit
experiment failure, not permission to lower thresholds. After eligibility,
S0 and S1 receive byte-identical full model/optimizer/RNG/sampler state; only
`transfer_integrity_enabled` differs. The enabled branch gradually introduces
the same tested v1 losses over2,000 updates. The disabled branch delegates to
repaired C1; the independent fork changes only its Tuning import.

Unit tests cover activation, correct relative sampling time, topology/source
guards, inherited baseline behavior and all reused v1 objectives. Full-state
GPU gradient/update, disabled-versus-original repeat controls, and enabled
checkpoint recovery precede full training. Frozen source/image/binding hashes
protect this run and historical experiments. Evaluation uses final full states,
30 validation and50 test observations, three fixed cameras over all200 times,
moving controls and each fresh own early bank. No novel-view GT or physical
identity accuracy is invented.

```powershell
. .\reproduction\scripts\enable_cuda.ps1
envs/pidg-cu128/python.exe -m variants.scratch_integrity.prefix
envs/pidg-cu128/python.exe -m variants.scratch_integrity.run
```

The second command can be started while the first is running; it waits for
that existing prefix process rather than launching a duplicate. Predeclared
protocol and evidence: `experiments/scratch_integrity`. Full states and data:
`runs/pidg_scratch_integrity`. Generated C1 fork code is reproducible and ignored.
