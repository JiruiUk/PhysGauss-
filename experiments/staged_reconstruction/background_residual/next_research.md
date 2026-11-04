# Additional source review while the matched I10 arms train

Research only: no second intervention, graph, target or training process was
added while the I10 protocol is frozen.

Primary source: USplat4D, ICLR2026,
<https://arxiv.org/html/2510.12768v3>, official project
<https://tamu-visual-ai.github.io/usplat4d/>, official code
<https://github.com/TAMU-Visual-AI/usplat4d>.
The paper estimates time-specific Gaussian color uncertainty from inverse
squared compositing support, gates by photometric convergence, selects anchors
with recurring support and propagates motion through a directed graph. This
differs from our I1 graph's uniformly weighted trajectory-distance neighborhoods.
Its color-uncertainty estimate is not independent geometry or object identity GT.
The paper's depth-axis scaling text and supplementary reference labels are
inconsistent; official implementation must be audited before adopting them.

## Local implementation constraint discovered before any adaptation

The original PIDG renderer returns `gs_per_pixel` and `weight_per_gs_pixel`,
but these cannot be assumed to be the full color Jacobian:

- `third_party/PIDG/submodules/flow-based-diff-gaussian-rasterization/`
  `rasterize_points.cu` fixes K=20.
- `cuda_rasterizer/forward.cu` accumulates RGB using alpha*T, then updates
  T=test_T, and only afterwards exports alpha*T for the first20 contributions.
  Thus exported weights use post-update transmittance and omit later rows.

This source inspection changes the safe next action: do not directly scatter
those exported squared weights and call them exact uncertainty. Preserve the
original rasterizer; no baseline correction is made in I10. An independent
native color-Jacobian backward with override colors could estimate total
squared support using deterministic local Rademacher probes. A tiny-scene
explicit per-row rendering check and convergence/ranking stability tests would
be required before any anchor selection. Such a stochastic estimator is an
adaptation, not a full reproduction of USplat4D.

If I10 does not restore complete geometry, the next question is whether the
selected S3-U has reliable recurring anchors at all, rather than trusting every
well-rendered training pixel or every nonzero screen-space gradient. Any graph
still needs to avoid treating proximity as certified same-object membership.
