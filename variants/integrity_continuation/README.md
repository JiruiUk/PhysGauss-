# Complete the four existing20k pilots to50k

User-requested continuation: H1 balanced coverage, H4 color-consistent density,
F1 feedback and F2 conservative retry. C1, H2 and H3 are already complete and
are reused. The optional, never-started H5 is not a continuation target.

This overrides the earlier pilot promotion filter for an explicit full-budget
ablation. Original20k selection/rejection reports are copied into the new
report directory. A50k completion must not be presented as evidence that a
previously rejected method improved the model.

Existing model code/configurations remain frozen. Full checkpoint SHA,
saved configuration, iteration, optimizer/RNG availability and optional density
sidecars are checked before training. The original optimizer, RNG, sampling
stack, Gaussian state and learning-rate schedule resume at20001; total target
is50000, not50000 additional iterations. No new physics module is integrated.
Training and metric/render jobs are sequential on the existing GPU.

A new eight-hour wall-clock ceiling applies to this explicit continuation.
The old experiment budgets remain unchanged. This queue's deadline persists
across restarts; `--renew-budget` is reserved for a later explicit user request.
At the deadline it saves and stops. Completed checkpoints are skipped.

All four validation selections are locked before their final-test renderings.
Exports reuse C1/H2/H3 images and include individual renders, GT pairs and
GT plus all seven models, preserving1600x900 source pixels. A new comparison
report records four image metrics and the2D completeness proxies. Previous
single-scene/seed and historical test-exposure limits remain.

Monitor: `experiments/integrity_continuation/monitor.html` (refresh5s).
The original dashboards and frozen sources are not edited.
