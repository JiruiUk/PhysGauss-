# Reliability measurement qualified; complete-object goal still open

This milestone validates a way to measure S3-U color support. It does not train
a new reconstruction, prove physical identity, or establish complete surfaces.
No original PIDG source, frozen historical experiment, optimizer state or
checkpoint is changed. New diagnostics require explicit opt-in flags.

## Actual numerical evidence

-22 CPU tests pass: analytic linear operators, exact sign enumeration,
  occluded/invisible points, weighted pixels, private RNG, insufficient/empty
  observations, independent rank/eligibility checks and recurring voxel selection.
- Native37-point test uses13 one-hot RGB renders as exact pixel-weight truth,
  explicitly including contributions after the first20 rows and an invisible row.
  Total native color VJP agrees with explicit weights within relative1e-4.
-128 RGB batches (384 independent channel probes) fail the original10% median
  relative-error gate:16.11% and11.11% for the two seeds. These failed budgets
  are retained.512 batches (1536 probes) pass:2.29% and4.55%;2048 batches also
  pass. No accuracy threshold is relaxed.
- The exported PIDG buffer has80.59% relative L1 squared-support discrepancy
  **in this deliberately overlapping37-point fixture**. This is not an error
  percentage for real S3-U or a causal share of its fragmentation. The diagnostic
  does not use that buffer to choose anchors or change the renderer.
- Actual train-only pilot: frames2/43/86/146/188/200 at width800 plus frame86
  at native1600. All7 cases pass independent-seed, exact one-hot sample and
  unchanged native RGB/alpha checks. Per-case exact-support median errors range
  up to3.65%, with2.01% median across cases and seeds.
- Separate eligibility per seed yields candidate-set Jaccard0.9667--0.9907
  on these7 cases. The original helper's common-mask rank overlap can be trivial
  when fewer than310 points qualify; independent eligibility is required in the
  subsequent all-time selection protocol.
- The native37-point test and observed pilot verify unchanged global Python,
  NumPy, torch CPU/CUDA RNG and model geometry/state. No model `.grad` is written,
  no optimizer step is made, and no test RGB is read. This is inference parity,
  not a claim that an unimplemented training intervention passed training tests.

`verification_pilot.json` independently verifies upstream blob hashes, source
and input hashes, explicit weights, native arrays, exact-support errors and
the7 original case receipts. Torch/NumPy squared-sum reductions differ at final
floating-point rounding; verification uses relative1e-14/absolute1e-12 for this
cross-library reduction only. The original exact-equality verification failure
is preserved in `verification_pilot.log`.

Two native wrapper errors are also retained: handling the renderer's -1 empty
slots, and detaching the diagnostic exported buffer before NumPy serialization.
The final versioned qualification passes under unchanged statistical gates.

## Actual visual review so far

Viewed `observed_pilot/t0086_w1600.png` and `t0002_w800.png` from
`output/staged_reconstruction/uncertainty_motion/`. These sheets resize the
measured RGB/contribution images to400 pixels per panel; they are observation
diagnostic figures, not native novel-view acceptance images. High-support
contributions fall on visible dynamic foreground regions, and sparsely cover
those regions. No conclusion about full3D coverage or same-object identity is
drawn from their white contribution maps.

## Next measurement is already registered

`observed_native120_protocol.json` fixes all120 real train times at native1600,
two independent1536-probe estimates and192 population-spanning one-hot sample
rows per frame. All120 original train RGB hashes were locked before the pilot.
The recurring selection gates, voxel size (twice original initialization
spacing), independent-seed intersection, temporal-third checks and observed
model-support neighborhood coverage are fixed before the all-time run.

Those are graph-feasibility gates, not completeness acceptance. Subsequent
motion training requires a separate opt-in implementation, actual smoke and
recovery audit, matched controls and the unchanged6x9 then6x200 native novel-view
acceptance. Depth and physical correspondence remain uncertain even if reliable
color anchors exist.
