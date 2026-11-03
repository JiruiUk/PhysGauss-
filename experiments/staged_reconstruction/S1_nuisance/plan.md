# S1-N: one bounded nuisance-separation experiment

Status: research/protocol prepared; implementation and full Gaussian synthetic
training have not started. Do not describe the completed point-admission audit
as this experiment. This is the next authorized stage, not an extra candidate
parallel to active C1 training (C1 is complete and not promoted).

The completed visibility experiment did not reduce real native artifacts.
Independent RGB geometric admission benefited modestly from quarantine, while
about14% of inferred points remained away from static GT. The existing
confidence-masked image loss gives no image gradient outside admitted pixels;
surface SH can also absorb inconsistent foreground appearance. The next single
hypothesis is whether explicitly explaining per-image nuisance with full-RGB
supervision can reduce these residues without discarding thin static surfaces.

Primary method read: [DeSplat, sections3.1 andB.3](https://arxiv.org/html/2411.19756v2).
Render shared static and selected-view nuisance Gaussians separately, then
compose premultiplied RGB as `C_d+(1-alpha_d)*C_s`. Per-view nuisance parameters
update only for that view. The paper uses alpha regularization and1000 initial
nuisance primitives per image. This budget is unrelated to the number of
physical objects. Our bounded adaptation keeps a fixed budget and native PIDG
rendering; it is not a full paper reproduction.

Both arms use the same full-RGB objective. Keep static support regularization
only on inherited reliable train-inferred geometry: a full-image alpha-one
prior would manufacture geometry against this dataset's infinite black
background. Nuisance can explain reflections or uncertain content; it is never
called an identified dynamic object or used as a physical force/material field.
Preserve S3-U and its unified dynamic pool.

Required execution order:

1. Add a new default-disabled branch with compositing, camera/native-renderer,
   selected-view-only gradient, optimizer/RNG/sampler and exact reload tests.
   Native smoke must check static and nuisance RGB/alpha gradients and flags-off
   parity, with separate full-state40→80 process recovery.
2. Full Gaussian synthetic train/evaluate both arms on the fixed diffuse/glossy
   v2 RGB data. Initialize only from train-RGB reconstruction, not evaluator GT.
   Evaluate static-only and composite images separately. Reuse the saved camera
   sequence. Independently evaluate static RGB, depth/alpha, geometry coverage
   and thin-structure preservation; expected depth is accumulatedD/alpha, notD.
3. Only if both synthetic conditions pass the predeclared gate, run one matched
   real18k→24k continuation from complete control18k model/Adam/RNG/camera stack.
   Keep topology fixed. No additional loss-weight/threshold sweep or physics.
4. Native fixed86/188 and original trajectory, common frozen proxy regions,
   reliable support, exact PLY, decodedH264, actual manual PNG review, separate
   milestone commits and assessment/plan revision. Preserve all failed attempts.
   Refuse promotion if composite fit improves by hiding static structure.

Do not skip the synthetic training gate. Do not restart C1 or paused historical
tasks. Check actual processes/locks first. Honor user stop by saving full state
and releasing GPU. New module source/protocol/input must be frozen before any
formal training; choose routine implementation details once before results,
record them, and do not adapt weights to desired outcomes.
