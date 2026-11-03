# I3: ordinal geometry prior after I1/I2 fail cross-view integrity

MoDGS (ICLR2025) supplies the hypothesis: monocular depth ordering can help
geometry even when metric depth scales are inconsistent across frames.
Primary source: <https://arxiv.org/html/2406.00434v4>, Section3.3 Eq3.
This is an adaptation, not a reproduction: no inverse deformation field,
3D-aware Gaussian reinitialization or GeoWizard estimator is introduced.

Checked-out official PIDG README identifies the supplied synthetic-scene
depth maps as Distill Any Depth outputs. They are learned pseudo-depth, not
Blender GT or independent geometric evidence. An actual six-training-frame
audit found all arrays finite and negative rank correlations with the frozen
static reconstruction (Spearman -.40 to -.84), supporting larger-is-closer
direction. This audit is model-conditioned, not proof of depth accuracy.
Visual review of the six-row depth qualification sheet also showed errors
and smooth boundary bias. Ordinal labels discard near-equal pseudo-depth
pairs and erode the existing dynamic union mask; neither step proves labels.

Use the cached training120 maps only. No validation or test depth is read.
Global and local pixel pairs are deterministic and fixed before training.
Loss compares tanh(10*(normalized-render-depth[a]-depth[b])) with
sign(pseudo[b]-pseudo[a]); larger pseudo values indicate nearer geometry.
Depth is dynamic-alone alpha-normalized camera-z from the original renderer.
The mask and pair denominator never depend on current opacity. Opacity,
scale, rotation and SH attributes are detached for this auxiliary rendering;
the depth loss can train centers and the original deformation grid/network.
Actual native backward testing confirmed finite nonzero center/field gradients.

I3 starts from the identical immutable S3-U final6k model and full Adam/RNG
state as I0/I1/I2. It retains the I1 frozen graph prior and original image and
physics losses, adds ordinal weight .03 ramped for100 updates, then runs2k
actual steps. I1 is the matched no-depth control. Dense original PIDG motion
is used; the failed I2 representation is not combined with this new prior.
This is a different supervision condition and must be reported as such.

Acceptance stays unchanged: train/val selection, same six cameras and nine
screen times, then all200 times only for a visually promising candidate.
Native1600 validation dynamic-PSNR drop <=.3dB vs S3-U and dynamic alpha>=.95
are necessary guardrails; persistent sheets, missing parts, merges or
detached fragments fail regardless of scalar gains. No new-view GT exists.
MoDGS itself notes unseen-surface/occlusion limitations; success is not assured.
