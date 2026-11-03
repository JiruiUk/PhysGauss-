# Literature and experiment decisions — 2026-11-03

The objective is less visible contamination while preserving thin surfaces,
boundaries and completeness. PSNR alone cannot decide success. The current
G1/G2 outputs contain foreground-like smears and floaters despite recovering
the column. No dataset-independent static or depth GT has been established.

| Primary source | Verified mechanism | Application and limitation here |
|---|---|---|
| [RobustSplat, ICCV2025](https://arxiv.org/html/2506.02751v1) | Delayed densification and progressively refined transient masks; its appendix also discusses contaminated initial SfM points. | Keep growth disabled while testing admission. Our geometric visibility audit is an adaptation, not their DINOv2 mask implementation or a paper reproduction. Delaying growth alone cannot remove existing contamination. |
| [SpotLessSplats, arXiv2024](https://arxiv.org/html/2406.20055v1) | Feature-based robust inlier estimation, gradual mask use and utilization-based pruning; differentiates rendered-image derivatives from loss gradients. | Initial points must be eligible for evidence-based quarantine. Simple residual thresholds or low opacity are insufficient. The present experiment does not implement diffusion features or claim its exact pruning metric. |
| [RobustNeRF, CVPR2023](https://robustnerf.github.io/) | Robust estimation treats transient observations as outliers without specifying their object class. | Useful baseline for robust supervision; RGB residuals alone can confuse reflections, missing static surfaces and motion. Do not use residual alone to delete structure. |
| [DeSplat, CVPR2025](https://aaltoml.github.io/desplat/) | Splits persistent scene and per-view distractor Gaussians within alpha compositing, without a pretrained semantic detector. | A later option to explain transient foreground separately without assuming a body count. Per-view distractors do not establish physical trajectories or identity; first test static admission. |
| [CoR-GS, ECCV2024](https://jiaw-z.github.io/CoR-GS/) | Two jointly trained fields use point/rendering disagreement for co-pruning and pseudo-view regularization. | Disagreement can flag uncertain geometry; agreement cannot prove truth. Our current inherited depths reuse RGB and are not two independent CoR-GS fields. A two-field experiment would be a separate bounded stage. |
| [2DGS, SIGGRAPH2024](https://surfsplatting.github.io/) | Oriented surface disks, ray/splat rendering, depth distortion and normal consistency. | Motivates a later surface-geometry ablation. Replacing the native PIDG renderer would break the current matched control, so it is not bundled into visibility cleaning. |
| [DN-Splatter, WACV2025](https://maturk.github.io/dn-splatter/) | Depth/normal supervision and local geometric regularization for 3DGS. | Test reliable train-derived depth/normal constraints next if needed, with a separate switch and an image/structure control. Estimated depth remains a prior, not GT; thin/discontinuous geometry needs protection. |
| [Cross-Temporal3DGS, arXiv2025](https://arxiv.org/html/2512.00534v1) | Confidence-guided reuse of historical well-observed scene information for sparse updates. | Retain revisable confidence, but our cold-start setting lacks its historical scene prior. Self-render agreement cannot independently admit missing geometry or certify staticity. |

Reading coverage: RobustSplat and SpotLessSplats method/appendix HTML; the
other linked primary project pages/abstracts and official publication records.
Some CVF pages returned403 and the AAAI page denied automated access; the
accessible author/arXiv versions above were used. No inaccessible PDF is
represented as fully read, and no numerical paper result is imported as ours.

## Current bounded test: C1 visibility quarantine

Start from the complete G1 twelve-thousand-step checkpoint, retaining full
model/Adam/RNG/sampler state. Compare an unmodified six-thousand-step continuation
against the same continuation with conditional ray-evidence quarantine. No new
loss, body count, growth or parameter sweep is included.

A Gaussian in front of a reliable train-inferred surface contradicts that
surface's observed ray, while one behind it is potentially occluded. Three or
more contradictory views, at least20 frame indices apart overall, and a
contradiction fraction of at least0.6 among agree/contradict votes are required.
Missing/occluded depth supplies no negative vote. RGB error, color labels and
opacity do not determine this quarantine. All initial rows are eligible; full
parent state and quarantined hypotheses remain recoverable and are not called
dynamic objects or empty background. These thresholds were fixed before the
actual evidence scan; they are not a paper's published hyperparameters.

Analytic ray fixtures test the mechanism with known geometry. They are not an
end-to-end RGB benchmark. The real-data evidence comes from inherited RGB depth
estimates and is conditional. Keep old and broad inferred masks frozen for
evaluation; inspect actual native early/middle/late86/188 images. Refuse
promotion if apparent cleanup removes structure. Preserve negative results.

## Continued sequence, one bounded experiment at a time

1. Finish C1 matched continuation, numerical and native-image review.
2. Add a small independent RGB synthetic audit with static/moving surfaces,
   specular appearance and independently varying time/camera; evaluator-only
   static/depth GT must not enter the reconstructor. Measure false inclusion,
   surface coverage, depth error and static texture preservation.
3. If C1 leaves surface noise, test reliable depth/normal/surface constraints
   inspired by DN-Splatter/2DGS as one independently switchable stage. If the
   dominant issue is transient appearance, prioritize DeSplat-style nuisance
   separation or feature-based masking instead. Do not start both branches.
4. Consider two-field uncertainty only if the preceding evidence warrants the
   extra compute. Keep unknown identities and one unified physical dynamic pool.

No clean-static claim follows from deleting points, increasing alpha, unchanged
fixed-time output, or self-selected PSNR. Historical baselines remain frozen;
material/constitutive/contact/force extensions are outside this effort.

## Result-driven decision after C1

Real matched continuation did not clean native artifacts: CT inferred validation
25.7836/control vs25.7377/quarantine. Independent RGB-derived point admission
precision improved81.9%→86.1% with visible-surface coverage about99.4%, but this
is not full Gaussian training. Existing surface SH contamination is the next
mechanism to test rather than repeating point deletion.

Read [DeSplat method and appendix](https://arxiv.org/html/2411.19756v2),
sections3.1 andB.3: shared static and per-view nuisance Gaussians are rendered
separately and alpha composited. Only the selected view nuisance set updates;
alpha regularizers discourage explaining everything as nuisance. Our next
bounded adaptation must preserve the native PIDG renderer and unified physical
dynamic pool. Its background is black/infinite, so a global static-alpha-one
prior would create false geometry; restrict structural support to independently
inferred reliable regions. Test full-RGB control versus the same objective with
per-view nuisance, with no semantic labels, body count, growth or weight sweep.
Nuisance masks are appearance explanations, not dynamic IDs or force fields.

First require full Gaussian synthetic training and evaluator-only static/depth
checks. Then run one matched real pilot from the preserved complete parent,
with full-state recovery and unchanged camera sequence, rendering static-only,
nuisance and composite outputs separately. Reject improved composite fitting
if thin/static surfaces disappear. No improvement is claimed in advance.

## S1-N negative result and depth-prior audit

Full Gaussian synthetic nuisance training failed both fixed static-GT gates:
novel-camera static PSNR fell0.739/0.644dB and RGB spill increased6.05/6.44%.
See S1_nuisance/v1/assessment.md. No real nuisance continuation or stronger
alpha regularizer follows. This bounded camera-plane adaptation is not the
complete DeSplat method and does not refute that paper.

Re-read [DN-Splatter full method, sections4.1–4.4](https://arxiv.org/html/2403.17822v3):
it combines gradient-aware depth regularization with sensor/scale-aligned
monocular cues and Gaussian-derived normals supervised by monocular estimates.
The authors explicitly discuss noisy normals derived from rendered-depth
gradients. Our RGB-MVS depth is neither their sensor prior nor their monocular
normal prior. Reusing noisy finite-difference normals would not reproduce their
method. [2DGS](https://surfsplatting.github.io/) also relies on its own oriented
disks/ray intersections plus geometric regularizers; simply copying a loss into
the unchanged native PIDG renderer is not an equivalent implementation.

Read-only S1-D0 fixes the old RGB-only depth selection before accessing GT.
Same-ray depth agreement is only79.4% at0.05 and52.6% at0.02; checked-depth
normal error averages30.8–30.9deg on the single-face subset. These differ from
point-to-nearest-surface precision and must not be conflated. Thin pixel depth
support is only56.8%, even though spatial point coverage was near99%.
Do not attach strong depth/normal supervision yet. Next single mechanism is
continuous inverse-depth refinement with disjoint held-source cost validation,
first inference/evaluation only; no GT-driven threshold sweep or new losses.
