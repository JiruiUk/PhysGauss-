# V0: Rubber Duck camera/time coupling diagnostic

Question: does apparent completeness concentrate around the observed camera-time
diagonal, while the same frozen geometry becomes fragmented or develops dark
high-alpha support as its camera departs? This is a read-only diagnostic, not
training, model promotion, an accuracy benchmark without GT, or proof of one cause.

Compare exact retained J7 and both completed J8 continuations. Use existing120
training camera poses/intrinsics/times only. Anchors are training-list indices
24/60/95; local offsets are -24/-12/-3/0/3/12/24. Fix time at each anchor and render
all120 cameras. Separately render all7x7 camera/time combinations per anchor;
retain diagonal actual RGB observations, never substitute them as off-diagonal GT.
All poses are real dataset metadata; no orbit interpolation, recentering or crops
chosen by results. Report actual view-direction and SO(3) angles, translation,
time difference and denominators. Local graph/node identities are unknown.

Native PIDG width800 renders provide joint RGB and dynamic-only RGB/alpha/depth.
Shape descriptors at fixed400x225, inherited alpha.5/thin2/island20/hole4, are
unreferenced proxies, not physical object counts or completeness truth. Main
diagnostic is the fraction of rendered dynamic alpha>.5 pixels having max dynamic
RGB<.05; report support area, alpha mass, missing/clipped and shape descriptors
alongside it. Darkness never creates a background/dynamic label or deletes points.
Angle bins are [0,5,15,30,60,90,180.000001]. Keep all results, null empty ratios,
per-anchor values and disagreement; do not tune definitions or select best cases.

For every seven-camera fixed-time row, also render constant-gray dynamic-only
support; alpha/D must match the ordinary dynamic render. This removes SH color
and static compositing as sole explanations, not all ambiguities. Export those
63 model/anchor/camera combinations at native1600 for visual review as well.
Produce all nine7x7 joint/alpha contact sheets, observed diagonal RGB controls,
camera-angle curves and same-pose time controls. No fixed-time novel-view GT is
available; never calculate off-diagonal PSNR against the wrong observation.

Before measurement, test pose distance, pairing/duplicates, null/border behavior,
proxy calculations and default-off CLI. Verify native direct-render identity,
gray alpha/D parity, full model/twoAdam/RNG/camera/physics/controller no mutation
including exception context. Freeze new source/protocol and exact inputs first.
Then one serial full audit; independently rerender all metrics800 in another
process and compare all rows/render hashes, aggregate bins/paired controls and
native1600 image receipts. Review fixed panels before concluding. Save failures,
locks and stop.request; stop releases this task GPU and never resumes training.

Preserve original PIDG, historical weights, J8 failure/status and unrelated work.
No training, new labels/blue threshold/object count/depth prior/physics/force,
candidate sweeps or subagents. Commit protocol, audit implementation/results and
final reviewed report separately. Shared docs get only task-specific cached patch.

Interpretation: concentration of visible-support quality near the diagonal plus
rapid off-diagonal deterioration supports camera-time conditional overfitting /
underconstrained geometry. Natural perspective/self-occlusion, pose distance and
actual motion can also change silhouettes, so proxy changes alone cannot establish
accuracy or separate these causes. This audit cannot prove that a stronger prior,
more capacity or longer training resolves the problem.
