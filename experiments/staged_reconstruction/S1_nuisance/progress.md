# S1-N implementation and synthetic verification

The user authorized continuing improvements toward clean, complete structure.
The protocol is predeclared in `protocol.json`; no result-driven weight/threshold
change is permitted. This is one nuisance-separation candidate, following the
completed C1 negative result and RGB admission audit.

Implemented in `variants/staged_reconstruction/nuisance_static`: an independent
native-renderer Gaussian plane for each training image, explicit premultiplied
composition, full-RGB objective, restricted static-support prior and per-view
Adam. A plane is an appearance explanation, not a physical dynamic object.
Static topology is fixed. Flags-off returns the native static RGB unchanged;
no baseline, material, motion or force field has been changed.

Implementation details fixed before training:1000 plane primitives per view,
camera depth0.3 (native near-point guard is0.2), fixed plane orientation/depth,
learned2D positions/scales, RGB and opacity. Private deterministic NumPy seeds
preserve global sampler/RNG. Per-view Adam updates only when that view is drawn.
No densification, pruning or object-count inference. Planar appearance is a
bounded DeSplat-inspired adaptation, not a reproduction of its free3D/ADC code.
Optimizer rates and bounds are recorded directly in frozen core.py.

Synthetic initialization uses only the previous train-RGB inferred point cloud,
common0.06 voxels and at most4000 points, spacing0.04. Both arms start from the
same unfiltered model; quarantine is not mixed into this candidate. Conditional
support priors come from multi-view geometric support, not GT/static labels.
The evaluator alone reads `evaluation_only` static RGB, depth, normals and
visibility. The infinite black background gets no static-alpha-one prior.

Preflight: tests, preparation and native four-update gradient/parity check,
then source/input freeze and independent40→80 process recovery for both arms.
Initial test failures are preserved: buffer name `half` collided with nn.Module
and was renamed `plane_extent`; one pytest temporary parent directory was absent
and was created. These were implementation/harness fixes before formal runs.
No loss weight, candidate or experiment acceptance gate was changed.

Next: four serialized6000-step synthetic runs (control/nuisance × diffuse/glossy),
full optimizer/RNG/sampler/per-view audit and independent static-GT evaluation.
Inspect actual static/decomposition PNGs before any real training. Gate file
requires manual review as well as both numerical cases; the pipeline does not
automatically launch real training. Real continuation remains18k→24k only if
both cases pass. No previous paused workload is resumed.

Check actual process, `runs/pidg_staged_reconstruction/S1_nuisance/v1/state.json`,
`pipeline.json`, `experiment.lock` and `pipeline.lock` before starting anything.
`stop.request` saves full training state and releases the GPU; never resume a
user stop. Preserve unrelated changes and all failed logs. Commit implementation
and each completed training milestone separately.

## Preflight completed

72 tests passed (10 new nuisance tests,62 inherited). Native four-step check
confirmed nonzero finite static/nuisance RGB/alpha/geometry gradients and exact
flags-off RGB/alpha/depth parity. Source/input freeze852952eed5d63071760bdc661d221bdf5f9c5bdbe0008348b840644284340d76.
Both arms passed independent40→80 process recovery: static Adam counters80,
per-view Adam counters equal view visits, unvisited planes unchanged, initial
static/RNG/camera sequence matched, all full-state and topology audits passed.
Prepared common4000-point synthetic initialization and96 per-image planes/arm.
Formal synthetic training can now start; no real training until both GT gates
and actual manual review pass.

## Diffuse full-RGB control completed

6000 updates completed with4000 fixed static points. Full-state audit passed:
six static Adam counters6000,96 view visit counts replayed, RNG/camera stack
exact, source/input/history unchanged. Checkpoint SHA256
`64b5bc00a6a1ada7957c4d1bd8087df59c7e4c00f994cf02ace30fa83e3fb45b`.
No quality conclusion before nuisance/glossy and independent GT evaluation.

## Diffuse full-RGB nuisance completed

6000 updates completed with4000 fixed shared static points and96 independent
1000-primitive nuisance planes. Full-state audit passed: all six static Adam
counters6000, per-image Adam counters exactly match62/63 selected-view visits,
unvisited planes unchanged, RNG/sampler/topology replay verified. Checkpoint
SHA256 `9dc4322df099a123de1fc18801dbca2dbb621820c5b1239053176990bc87db06`.
This records execution integrity only; static quality awaits matched independent
GT evaluation and actual image review. Glossy arms remain serialized.

## Glossy full-RGB control completed

6000 updates completed with4000 fixed static points. All six static Adam
counters6000 and the same96-view sampling/RNG/topology/source audits passed.
Checkpoint SHA256 `c4dd10f4ec2c45fa21a9fcaf296313e0b6ce476299fe21766fd29cfe9fe5fd3a`.
The glossy nuisance arm is the last formal synthetic arm; no real training has
started. Quality conclusions remain pending independent evaluation.
