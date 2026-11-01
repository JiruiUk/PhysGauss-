# Model-only deformation repair on H3

Independent switches: `integrity_motion_affine_weight` and
`integrity_covariance_weight`. Default zero preserves the inherited H3 loss path.
The fork changes only the trainer's Tuning import. The renderer, image loaders,
camera sampling loop, seed, split and training RGB/alpha preprocessing are copied
unchanged. New regularizer queries use no additional image target or label.

- D1: motion-compatible, persistent/color-gated local affine consistency.
- D2: tolerant temporal curvature of normalized rendered covariance and log size.
- D3: both, with the same independent weights as the single-component arms.

The covariance calculation is checked against the actual CUDA rasterizer,
including nonunit additive quaternions and signed scale offsets. The motion graph
is a heuristic affinity, not semantic ObjectID or learned material segmentation.

All arms share the existing random-training3k full checkpoint and train to50k
total; completed H3/C1 are reused. No frame sampler, image weighting or data
augmentation is introduced. See `experiments/deformation_repair/plan.md` for
formulas, tolerances, gradient scope and limitations.

Results/monitor: `experiments/deformation_repair/`.
Models/GT-render galleries: `runs/pidg_deformation_repair/`.
The original8-hour deadline is retained, including time spent on the superseded
queue. New test exports remain locked until validation selections are fixed.
