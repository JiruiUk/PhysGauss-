# J5: target-data camera-time coverage and dark dynamic support audit

J4's paired RGB improvement is retained, but fragmented off-trajectory dynamic
geometry and dark high-alpha support remain. This is one read-only diagnostic
on Rubber Duck, not another training candidate. Do not repeat J0's gradient
audit or the existing static-attenuation probes. Reuse their conclusions and
the failed motion-integrity/local-rigidity/shared/invertible-motion results.

Question: how much same-time camera evidence actually exists, and how do the
Gaussians contributing to dark high-alpha fixed-view dynamic pixels appear
along the supplied training camera-time trajectory? Neither darkness nor
visibility assigns physical identity or establishes incorrect geometry.

Pin the S3-U final6k parent and both J4 final2000 states. Use only the120 supplied
training RGB/cameras/times. No validation/test RGB or semantic/instance/depth
labels. Use the already fixed cameras86/188 and target times2/100/200 at native
1600. Count exact metadata times/camera transforms and report time gaps rather
than inventing simultaneous views. For each model and target, define the
diagnostic pixel set as dynamic-only alpha>0.5 and mean clamped RGB<0.05.
This is an evaluation subset, never a training mask or deletion rule.

Select the union of valid Gaussian IDs in the native renderer's first20
contributing IDs on those pixels; never treat the capacity-limited list as all
contributors. Do not use weight_per_gs_pixel as compositing weights: source
inspection shows it may be written after updating T. Test that contract with
one fixed single-Gaussian native fixture, preserving the original renderer.
Measure the selected group's actual alpha contribution using the unchanged
native override-color pass with original geometry/opacities. Quantify the
fraction of target dark-pixel alpha explained; unrepresented mass stays unknown.

At each of the120 source camera-time pairs, deform the same selected canonical
row IDs at that source's own time and render them jointly depth-sorted with the
original static bank and all other dynamic rows. Static/other rows are black;
each selected group is white in one channel. Pack3 independent group queries
per RGB render (6 target groups need2 renders), and verify this against an
unpacked native pass on train index0. Groups can overlap and their masses must
not be summed as a partition. Frozen canonical row correspondence is only the
model's hypothesis, not independently identified material points.

Report per-source visible weight integral, observed mean RGB weighted by that
integral, and weight falling on observed mean RGB<0.05. No depth consistency or
segmentation assertion follows from those statistics. Source times differ;
the same-world-point projection across times is explicitly forbidden. Missing
visible weight has an explicit zero denominator/unknown result. Do not infer
that black observations are empty, or bright observations certify a surface.

Save aggregate/per-source statistics, denominator counts, fixed native target
diagnostic images and selected fixed source index0/24/72 contact images. Do not
export per-Gaussian selections as training assets. Compare model+bothAdam+RNG+
sampler+physics-graph state before/after exactly; zero updates/gradients. Freeze
inputs and sources, use a task lock, honor stop requests, default off. Test
subset/invalid-ID/zero-denominator/packing arithmetic before the full audit;
native flags-off parity and two independent index0 measurements qualify the
worker. Then run one serial three-model audit and independently replay metadata
counts plus fixed index0/24/72 measurements before finalization. Any evaluation
repair gets a separate file/failed record, without changing frozen history.

No quality promotion, new training, threshold/weight sweep, point deletion,
static bank graft or physics/material/force module follows automatically.
The diagnostic must choose at most one next mechanism based on actual support
and coverage evidence; it must not claim RGB is unsolvable.
