# Completeness-first C1 experiments

Focus: preserve objects during motion, reduce alpha holes and fragmentation,
and distinguish these from opaque dark/color errors. No semantic/material
segmentation or instance labels are added. Existing C1 foreground RGBA
supervision remains; this is not an RGB-only experiment.

## Budget and queue

Eight hours from2026-10-31 21:17:35 UTC, ending2026-11-01 05:17:35 UTC,
including preparation and waiting for the current Object-ID queue. Existing
Object-ID training and its renders finish first; new GPU work is sequential.
No C1 baseline training or baseline GPU rendering is repeated. Existing
validation30 images, metrics and final50 test renders are reused.

Four candidates start from the retained identical C1 random-init full3000
checkpoint, with model/optimizer/RNG/camera-stack state preserved. Each screen
stops at20000 total while retaining the original50000 learning-rate schedule.
At most two candidates continue the same state to50000 total. If every screen
fails the image-quality guard and time permits, a quarter-weight combination
is screened once. Stop requests/deadline save full checkpoints; no claim of
completion is made for partial runs.

## Candidates

| Arm | New component |
|---|---|
| H1 | Foreground-normalized interior coverage deficit + background guard |
| H2 | Weak, tolerant short-term local strain consistency |
| H3 | H1 + H2 |
| H4 | Existing color-consistent late densification, isolated from affine loss |

H1 adds0.002 times the mean positive alpha deficit below0.95 in the eroded
training foreground, plus0.25 times mean alpha in the GT background. It ramps
on after3k over4k. C1's RGB, foreground RGB and perceptual losses stay active
to discourage fixing coverage by corrupting radiance. Opacity inflation is
also checked with background-spill and high-alpha appearance-error metrics.

H2 compares neighbor lengths at a training timestamp and its adjacent training
timestamp, penalizing excess absolute log length ratio with robust Huber.
The allowed strain is3% per original frame gap, capped20%; rigid motions
produce zero strain. Weight1e-4, every20 steps with interval compensation,
start8k/ramp4k. It does not lock distances to a canonical rigid template.
Canonical neighbors have radius0.05, opacity>0.05 and DC-color distance<0.25.
They must stay within0.15 at first/middle/last TRAINING timestamps under the
current detached motion prediction.96 anchors x8 neighbors are sampled
deterministically. These gates reduce transient-contact edges but do not
guarantee correct object association; same-color contacting objects remain a
failure mode. No validation/test timestamps or images enter this loss.

H4 isolates the already-tested color-evidence clone mechanism from the affine
regularizer that previously reduced PSNR. It observes deficient footprints
after16k, requires3 training times and color consistency, adds at most256
points per2k window through36k, keeps parents and resets higher-order SH for
new points. It cannot seed truly empty rays with no3D contributors. This is
an isolated ablation, not a claim that late cloning must help.

## Evaluation and adaptation

Only the120 training frames supervise losses. The30 validation times evaluate
four image metrics and the following2D completeness proxies: eroded interior
alpha<0.5 and<0.9, foreground missing coverage, background false coverage,
largest covered component loss, extra projected fragments, and high-alpha
appearance error. GT connected image regions are not semantic object labels;
the metrics do not establish3D watertightness or correct hidden surfaces.

Score =0.45*core_low90 +0.25*foreground_missing50 +0.2*fragment_loss
        +0.1*opaque_appearance_error +0.1*background_false_coverage.

Screens must stay within0.5dB PSNR,0.001 SSIM,0.003 LPIPS and0.001 MS-SSIM
of the existing C1 record at the same20k iteration, and background spill within
+0.5 percentage point of completed C1 validation. Eligible screens are ranked
by mean completeness score, then95th-percentile score. Final integrity-best
snapshots additionally pass the quality guard against completed C1 validation.
Final improvement claims require lower validation score and background spill
within+0.2 percentage point. All models/snapshots remain available if metrics
conflict; a low hole score alone is insufficient. Historical test exposure
and single-scene/single-seed limitations remain explicit.

All promoted50k selections are locked before new final-test rendering. Final
exports include independent RGB images, GT/render pairs and GT/C1/promoted
models in one folder. The baseline is read-only. A separate source fork changes
only the training Tuning import; flags-off inference files match C1 byte-for-byte.

## Research basis and differences

[Dynamic3D Gaussians](https://dynamic3dgaussians.github.io/paper.pdf) uses local
rigidity, rotation similarity and long-term isometry to encourage persistent
motion. [SC-GS](https://yihua7.github.io/SC-GS-web/) uses sparse motion control
and ARAP for continuity. [DynaSurfGS](https://open3dvlab.github.io/DynaSurfGS/)
combines planar Gaussians, surface-normal and ARAP regularization. H2 borrows
the short-term neighborhood idea, relaxes it for stretching Rubber Duck, and
retains PIDG/C1 representation and physics. It is not a reproduction of these
methods. A mesh/SDF/surfel replacement would be a separate architecture study
and is not silently introduced here.

CPU unit tests run before integration. The queue requires real CUDA gradient/
flags-off tests,100-step combined/density training,70-step resume, finite
validation completeness and PNG parity before any20k screen. It records a
separate milestone commit using only this variant's paths after preflight.
