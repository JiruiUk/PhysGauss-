# Static extraction affects dynamic reconstruction through two distinct inputs

Current evidence supports meaningful coupling, not a measured percentage of
failure caused by incorrect static extraction. No static-quality training
ablation has been performed in this audit. I8 formal training is unchanged.

## Actual code paths

1. **Initialization geometry:** `union/prepare.py` reads the original
   `runs/pidg_staged_reconstruction/static_metadata_initialization.npz`, not the
   fitted S1 checkpoint's rendered depth. `union/core.py:initialize_union` samples
   foreground pixels only in the earliest training frame. For each sampled
   pixel it takes the minimum camera depth among64 nearby projected static
   points, subtracts0.25, and adds uniform[-0.12,0.12] uncertainty. This is a
   conditional depth guess, not a ray-certified surface or metric GT. Static
   coverage/depth errors can bias the dynamic canonical initialization. Later
   optimization can move canonical points and deform them; this is not a hard
   permanent attachment to the static surface.

2. **Frozen compositing background:** `union/model.py` loads the separately
   trained S1 Gaussian checkpoint and freezes all its parameters. The unchanged
   renderer jointly depth-sorts static and dynamic Gaussians. Incorrect static
   depth/opacity can attenuate visible dynamic content, and static RGB residuals
   cannot update that frozen branch. `union/train.py:reconstruction` still sends
   gradients to dynamic geometry/appearance through the full-image objective,
   weighted0.5 dynamic foreground term, weighted0.5 blue/static image term and
   weighted2.0 foreground coverage. Dynamic points may receive pressure to alter
   the composition on static-colored pixels. This does not prove such pressure
   is erroneous: dynamic leakage onto correct static regions also activates it.

3. **Observation masks:** the dynamic image mask comes directly from the RGB
   yellow-foreground rule, not the complement of rendered static segmentation.
   Static errors therefore do not automatically relabel every observation.

4. **Physics:** current S3-U/I8 use velocity–trajectory and dynamic same-bank
   momentum terms. Static collision/contact/SDF losses are disabled. No claim
   that static collision forces are currently causing fragmentation.

Improving only the fitted S1 render does not retroactively regenerate the
already frozen dynamic initialization. The two inputs must be distinguished in
a causal experiment.

## Actual read-only checks

The exact user-selected S3-U final6k and immutable I8 step7000 preview were
loaded by their pinned hashes. No training updates, test RGB or protocol/source
changes. Original native1600 renderer was used at c188/t43, c188/t146, c86/t86.
Dynamic-only RGB was compared to the combined scene. Reviewed S3-U c188/t146
comparison and I8 c188/t43 dynamic-only native image, plus the already reviewed
I8 dynamic alpha. Both arms retain missing/incomplete surfaces and detached or
elongated support after removing static rendering.

With the same opacities/geometry and black background, an override-color pass
assigns black to static rows and white to dynamic rows. Its output measures
visible dynamic compositing weight. Compare its image integral to dynamic-only
alpha to describe attenuation by static Gaussians:

|Probe|S3-U final6k attenuation|I8 step7000 attenuation|
|---|---:|---:|
|c188/t43|0.00013%|4.00%|
|c188/t146|2.81%|2.28%|
|c86/t86|0.019%|0.00%|

These probes do not support interpreting their large missing regions primarily
as current static compositing occlusion. They do not measure earlier training
damage, occlusion accuracy, local worst cases, all views/times or static-geometry
truth. Some attenuation can be physically correct.

Three observed **training** frames43/86/146 at width800 were also differentiated
without optimizer updates. On S3-U canonical coordinates, the weighted static
image term's gradient norms are0.221/0.647/0.052 times those of the weighted
dynamic foreground term; their gradient cosines are-0.476/+0.125/+0.024.
On I8 they are0.315/0.427/0.245, with cosines-0.651/-0.067/-0.038. Raw within-group
gradient norms are not optimizer-step shares or causal percentages. The
static term responds to both static reconstruction residuals and dynamic
misplacement; opposing gradients alone do not identify which geometry is wrong.
The raw per-group values and same-mask static-only/combined RGB errors are in
`audit.json`. Different arms/budgets are diagnostic cases, not matched outcomes.

## Implication for the next experiment

Separate uncertainty in static-derived initialization from gradients caused by
the frozen background. Keep the selected S3-U reference and the current I8
protocol intact. A future static-background loss isolation should preserve
observed dynamic foreground and known empty-background supervision while
treating uncertain occlusion as unknown; simply labeling all static-colored
pixels as dynamic-empty would introduce false supervision. It must retain the
original static-preservation gate and actual new-view checks. A higher-quality
static reference must first be independently qualified, not assumed from RGB
fit. Replacing both initialization and background together would confound them.

Artifacts: `output/staged_reconstruction/static_dynamic_coupling/` contains native
combined/dynamic-only images and side-by-side sheets. Exact checkpoint/source/
image hashes, train-frame gradients and attenuation statistics: `audit.json`.
