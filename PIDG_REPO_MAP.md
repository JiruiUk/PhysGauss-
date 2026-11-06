# PIDG repository map — M0

## Optional sparse shared dynamic motion (2026-11-03)

`variants/staged_reconstruction/sparse_motion/core.py` implements canonical
farthest-point control selection, normalized four-neighbor RBF weights,
hemisphere-aligned quaternion interpolation and a SparseDeform adapter. Its
complete center mapping is shared by UnionScene rendering and inherited
finite-difference physics queries; mix0 returns the original deformation path.
`prepare.py` binds256 fixed controls to S3-U final6k. `train.py` retains full
original optimization/sampling state, transitions dense to sparse over400 steps,
then evaluates only control motions; `preflight.py` verifies geometry, gradients,
flags-off rendering and independent40/80 resume. `evaluate.py` uses identical
six fixed cameras, native RGB/alpha and validation-only iterative selection.
`freeze.py` protects both this branch and the previous motion-prior sources.
No new physical unknowns, learned segmentation, instance count or topology;
this is an adapted reconstruction representation, not a full SC-GS replication.

## Opt-in literature-guided dynamic integrity (2026-11-03)

`variants/staged_reconstruction/motion_integrity/core.py` constructs frozen
maximum-trajectory-distance mutual neighborhoods and a robust, rotation-invariant
local strain prior. `prepare.py` audits the exact S3-U final6k parent using only
training times. No graph edge is certified as physical/material correspondence.
`train.py` reuses UnionScene, full Adam/RNG/camera state, inherited reconstruction
and original guarded kinematic/CMR terms; only its opt-in I1 arm adds the prior.
`preflight.py` checks flags-off native/loss/gradient parity, actual100-step smokes
and independent-process40/80 recovery. `evaluate.py` renders native1600 fixed
cameras/time combinations, full RGB and dynamic-only alpha, reports descriptive
shape proxies separately from validation appearance, and does not read test RGB
for iterative selection. Reports: experiments/staged_reconstruction/motion_integrity.
No original PIDG file, constitutive model, force, material segmentation, object
count or topology is changed. This is a reconstruction-prior branch, not a new
physics milestone or admission of the failed surface-trajectory observation set.

## Opt-in cross-temporal static confidence pilot (S1-CT, 2026-11-03)

`variants/staged_reconstruction/cross_temporal_static/prepare.py` combines
train-only SIFT/LK fixed-point evidence with color-compatible cross-time
projections to propose generic regions; it never reads semantic static colors,
old hulls or model weights. `core.py` caps confidence by independent geometric
support and permits admission/revocation without turning unknown into empty
background. `train.py` runs matched fixed/progressive12k static arms using the
unchanged native renderer and saves confidence/Adam/RNG/camera state together.
`audit.py` freezes inputs/sources/history and independently checks actual
sampling, optimizer and confidence recovery. `evaluate.py` compares common
inferred regions and native86/188/original-camera images, exports standard PLY
and H264. This borrows the confidence-update idea, not a replication of the
paper's pretrained historical3DGS/DUSt3R setup. No new physics or dynamic pool
integration. Reports: experiments/staged_reconstruction/S1_cross_temporal.

## Unknown-static RGB-geometric discovery (S1-A, 2026-11-03)

`variants/staged_reconstruction/automatic_static/geometry.py` tests a fixed
world-point hypothesis using separate fit/check observations and rejects low
parallax/short evidence as unknown. `discover.py` owns unmasked feature tracking
and generic region propagation; it consumes training RGB and supplied cameras,
not S1 masks/weights, blue/yellow rules, old selected tracks or material labels.
The opt-in protocol and limitations are in experiments/staged_reconstruction/
S1_automatic. This is reconstruction preprocessing, not learned material
segmentation or a physics milestone. Existing frozen trainers remain unchanged.
`initialize.py` builds a conservative shell near automatically triangulated
points; `train.py` reuses the original renderer/StaticGaussian optimizer through
separate complete-state checkpoints. Unknown pixels have zero direct loss.
`experiments/staged_reconstruction/verify_automatic.py` audits all samplers,
Adam/RNG and native parity; `review_automatic.py` exports native images, standard
3DGS PLY and a moving-camera H264 video without a time deformation.

## Opt-in S3-U surface trajectory research (2026-11-03)

`variants/staged_reconstruction/surface_trajectory/observations.py` and `prepare.py`
extract count-free train-image tracks from raw bidirectional RGB flow, with
appearance, visibility and full-fragment closure checks. Versioned observation
receipts retain the insufficient first attempt. `binding.py` fits frozen local
support weights using only three bootstrap observations; `bind.py` audits the
immutable S3-U final6k state with full forward contribution renders and rejects
ambiguous depth hypotheses. No original group/row bindings are reused.
`split.py` groups image duplicates and overlapping weighted supports before
supervision/track-loss-holdout assignment. `review.py` exports native-image crop
galleries without claiming that generating images constitutes visual review.
The parent is `runs/pidg_staged_reconstruction/S3_union/physics/formal/checkpoint.pt`.
Reports live in `experiments/staged_reconstruction/surface_trajectory`; these
R0/R1 modules perform zero optimizer updates. Training integration and physics
expansion require the separately recorded observation, support and preflight
gates; no material identity or cross-occlusion reidentification is certified.

`diagnostics.py` queries the frozen supports at later observations and reports
reprojection plus complete visible support without filtering failed observations.
`admission.py` keeps the evidence gates distinct from user authorization and
training preflight. `experiments/staged_reconstruction/verify_surface_trajectory.py`
independently replays raw-flow interpolation, bootstrap bindings and grouped
splits; `summarize_surface_trajectory.py` writes the result and static figure.
Completed R0/R1: conservative observations66, fixed supports23 (12/3/8 over
early/middle/late), duplicate/support groups12. Evidence admission fails; no new
training integration, optimizer update, constitutive module or force is added.

## Independent count-free S3-U branch (2026-11-03)

`variants/staged_reconstruction/union/prepare.py` freezes a training-image union
initialization and separate protocol; `core.py` owns count-free pixel/depth
sampling and union coverage. `model.py:UnionScene` uses one canonical dynamic
population, one original DeformModel and one original VelocityNet/stress field.
It shares only frozen S1 static support. `train.py` owns opt-in16k reconstruction
warmup/6k S3 continuation, existing image/CMR weights and lock/stop handling;
`runtime.py` saves model/Adam/RNG/camera/derived graph state. `evaluate.py` writes
isolated native image metrics. `experiments/staged_reconstruction/verify_union.py`
checks real40→80 recovery and renderer parity; `review_union.py` exports native
fixed-view images and H264 comparisons to historical S3. No instance numbering,
physical body-number estimate or learned material segmentation is introduced.
Spatial dynamic-only neighborhoods cannot certify physical-boundary exclusion.
Original PIDG and frozen S1–S6 sources/checkpoints are not edited by this branch.

## Independent S1--S6 reconstruction curriculum (2026-11-03)

This opt-in branch does not modify `train_pidg.py`, baseline `GaussianModel`,
`DeformModel`, `VelocityNet`, renderer or historical checkpoints. Root
`variants/staged_reconstruction/{train_static,model,static_gaussian}.py` owns S1;
`stage2` owns immutable per-object Gaussian banks and original deformation
networks. Sequential `recovery`/`geometry_recovery` corrections preserve their
negative results and frozen sources. The camera contract is metadata CV and
must not be mixed with legacy PIDG checkpoint coordinates.

`stage3/train.py` retains the independent S3 image/physics schedule, optimizer and
complete state entry. `state_recovery/train.py` is the actual corrected S3 entry;
its `physics.py` uses original per-object PIDG velocity/stress fields with
same-bank current-world kNN and tested quadratic momentum derivatives. Invalid
derivative neighborhoods fail rather than report zero residual. The original
S3 smoke failure and its frozen source are preserved. `state_recovery/runtime.py`
serializes derived graph state as well as the model/Adam/RNG/camera payload.

`stage4/{tracks,prepare_tracks,bind,constraints}.py` extracts train-only image
fragments, binds one immutable model row with full visible alpha*T integrals and
filters uncertain/conflicting associations. `stage5/{prepare_sdf,nonpenetration,
controlled}.py` owns the conditional S1-interior prior and manufactured optimizer
validation. `continuation.py` owns separately gated S4/S5/S6 stages and protocols;
S6 unfreezes static appearance while keeping static geometry fixed. All are
explicitly disabled without their CLI enable flag. No contact force, material
law, unknown force or learned material segmentation is enabled by this branch.

Independent evaluation/review lives under `experiments/staged_reconstruction/`:
`verify_smoke.py` checks actual optimizer counters and exact sampler recovery;
`audit_stage.py` checks parent hashes, final sampling and internal objectives;
`review_objects.py` exports native observed/group/fixed-view images and decodes
videos; `finalize_stage.py` requires a separate actual image-review receipt.
These diagnostics do not certify physical IDs, complete geometry or contact
surfaces. See `runbook.md`, `progress.md` and the stage-specific assessments.

## Cold-start transfer-integrity version 2 (2026-11-03)

`variants/scratch_integrity/prefix.py` captures and verifies a fresh step-zero
repaired-C1 state, then runs the unchanged C1 trainer through common preheating.
`data.py` reuses only image/flow/triangulation evidence and rebuilds every row
binding from immutable fresh common checkpoints after original topology changes
stop. `stage.py` checks activation/source/topology; `runtime.py` reuses the v1
reconstruction objectives with a new relative clock and2k ramp, delegating to
C1 when disabled. `prepare.py` changes only the C1 Tuning import. `run.py`
checks original-path numerical repeat controls, real gradients and resume, then
continues matched arms through50k actual total updates with no LR reset.
Protocol and records are isolated under `experiments/scratch_integrity`, full
states under `runs/pidg_scratch_integrity`. This is not a new physics milestone.

## Optional fixed-view completeness metrics (2026-11-03)

`variants/view_completeness/metrics.py` separates independently observed alpha
coverage metrics from target-free thin-support/island/hole/boundary descriptors.
Its evaluation entry is disabled by default and never creates novel-view GT or
a full-object completeness score. `audit.py --enabled` consumes the frozen
fixed-camera PNGs with source/checkpoint hashes; it reports every camera over
all200 times and validation30 times separately. Twelve new semantic tests plus
ten existing integrity tests cover the diagnostic and supervised-loss behavior.
Reports live in `experiments/view_completeness`; no trainer/physics integration
or optimizer update is introduced. Future geometry priors remain unimplemented.

## Fixed-camera temporal completeness diagnostic (2026-11-03)

`output/fixed_camera_temporal/render_sequences.py` restores frozen final C1/T1/O1
states through the audited association evaluator, reuses actual trained cameras,
and varies only deformation time for three fixed poses plus a moving-camera
control. All2400 renders use native1600x900 before downsampling. Diagonal native
RGB parity, fixed-camera fingerprints and source/checkpoint hashes are checked.
`finalize.py` independently decodes all16 videos and verifies selected native
RGB downsampling; outputs a gallery and GT/original/fixed-view/alpha figure.
Reports, descriptive masks, exact manual review scope and verification are in
`experiments/fixed_camera_temporal`. No training integration or physics module,
optimizer updates, camera recalibration, novel-view GT or physical-ID claim.

## Observable association training pilot (2026-11-02)

`variants/association_training/prepare.py` forks repaired C1 and changes only
the Tuning import. `runtime.py` inherits original image/physics methods, adding
an opt-in fixed-row early-object anchor and symmetric raw-flow contribution
loss via `coverage_loss`. `model.py` contains tested differentiable objectives;
`prepare_data.py` binds the independently verified early membership to C1 SHA.
`run.py` compares identical full50k optimizer/RNG/camera states for2,000 actual
updates; `freeze.py` locks all training inputs. Full-state support and validation
evaluation is in `evaluate.py`. No learned segmentation/material physics is
introduced. Runs/reports/output are isolated under `pidg_association_training`
and `association_training` respectively. See the variant README and plan.

## Conditional local object membership (2026-11-02)

`variants/object_membership/model.py` is a disabled-by-default non-neural,
causal observational ledger: historical association, current visible support
and latched cross-object conflict are separate. `audit.py` verifies forward and
reverse visible-FB links on each prefix of the fixed early/late clips, consumes
the existing native1600 complete-contribution caches and center/motion gates,
then connects A2/A15 banks only with independent image continuity. It compares
causal bank additions after the A15 seed period with identical per-point gates.
`verify.py` independently recomputes all point/ledger arithmetic and343 hashes;
`visualize.py` writes six-frame center overlays and evidence plots. Protocol,
results and assessment: `experiments/object_membership`; arrays/figures/gallery:
`output/object_membership`. NPZ embeds exact historical checkpoint SHA. There
is no PIDG training integration, model update, learned material segmentation,
cross-gap identity claim or material correspondence/constitutive evidence.

## Fixed short-clip C1 diagnostic (2026-11-02)

`output/short_clip_diagnostic/image_camera.py` computes independent sparse LK
and raw-flow static-blue epipolar checks for four fixed camera hypotheses.
`camera_math.py` explicitly separates loader conventions and metadata hypotheses;
six CPU tests cover projection, epipolar geometry and camera/object decomposition.
`model_check.py` uses the existing frozen C1 loader/renderer and complete color
adjoints to select local fixed canonical banks from each clip's first3 views.
`resolution_control.py` repeats all24 observations at the actual trained width1600,
with unchanged masks/gates and both original400 and native-selected banks.
`summarize.py` reports post-result cohort-gate sensitivity without threshold tuning.
`finalize.py` produces plots/GT|Render gallery; `verify.py` independently checks
camera arithmetic, bank membership, source/target unknowns, coverage, input/model
SHA and48 pixel concatenations. Evidence is in `experiments/short_clip_diagnostic`;
full numerical caches in `runs/pidg_short_clip_diagnostic/cache` must be retained.
No model, optimizer, PIDG runtime, camera calibration or physical ID is modified.
Direct400 splatting differs materially from trained1600 rendering; diagnostic
coverage is not a geometry or material-identity accuracy score.

## Training-data short-clip selection (2026-11-02)

`output/short_clip_audit/scan.py` searches all eight-observation Rubber Duck
training windows using visible component separation, frozen raw FB support and
camera metadata. It never loads C1 or uses model scores for selection. It exports
contact sheets/native images/crops and a gallery. `verify.py` independently
checks selected-window FB support, camera angles, RGB pixels and input SHA.
Plan/results/manual evidence are in `experiments/short_clip_audit`. Candidate
clips are local diagnostic data, not physical identity/3D calibration GT or new
heldout benchmarks. No neural training or new physics module is introduced.

## Visible image-fragment / fixed3D surface bank (2026-11-02)

`variants/surface_correspondence/model.py` adds disabled-by-default, non-neural
joint center/footprint/observed-motion support gates, three-observation fixed
canonical row selection, and conservative image-fragment qualification. It
does not edit PIDG paths or assign material/object labels. `output/surface_correspondence`
contains `common.py` for complete RGB color-adjoint contributions (beyond the
first20 diagnostics), `contracts.py` for real GPU/full-checkpoint integrity,
`audit.py` for five local references and all120 frozen-row forward evaluations,
`finalize.py` for per-point unknown retention/unique coverage/actual compositor
images/3D center paths, `bind_banks.py` for exact full-checkpoint/data/row SHA
binding, and `verify.py` for independent CPU flow and gate checks.
Protocols/results are in `experiments/surface_correspondence`; fixed banks and
NPZ trajectories/assignments in `output/surface_correspondence`; seed evidence
caches in `runs/pidg_surface_correspondence/cache`. Tests cover disabled behavior,
missing support, ambiguity and fixed-row selection. No new Gaussian training,
shape/contact/force/material objective or cross-occlusion physical-ID merge.

## Motion-history correspondence prototype (2026-11-02)

`variants/motion_correspondence/model.py` supplies an optional non-neural image
position/velocity tracker with irregular normalized time, covariance prediction,
one-to-one multiple hypotheses, merge rejection, expiry, persistent ambiguity
and JSON state restore. Flags off delegates exactly to the prior visible-only
`output.surface_support.core.propagate`. Candidate image seed identities are not
Gaussian/material labels. No PIDG optimizer or training/inference source changes.
`run.py` evaluates fixed synthetic hidden pass/bounce ambiguity, 36 artificial-gap
trials against frozen visible FB pseudo-references, and all120 train frames.
`output/motion_correspondence/rolling_test.py` adds local rolling visible tracklet
checks across the sequence; these do not provide global physical identities.
`render_support.py` checks complete C1 early-patch alpha compositing and centers;
`finalize.py` creates figures/browser; `verify_results.py` independently checks
inputs, metrics, masks, state replay and protected C1 hashes. Protocol, initial
failed uncertainty probe, corrected results and assessment are retained in
`experiments/motion_correspondence`; tests in `tests/test_motion_correspondence.py`.
This diagnostic does not resolve the rejected persistent-surface gate or resume P1.

## Full train-sequence surface-support audit (2026-11-02)

`output/surface_support/{core,audit,contracts,finalize,verify}.py` adds read-only
support diagnostics for historical C1 final50k and rejected P1 at16k. It checks
all120 train frames, full center trajectories, effective additive covariance,
full compositor group contributions, isolated/fixed-footprint/fixed-camera
counterfactuals and conservative raw bidirectional image tracking. The CUDA
RGB/alpha compositor uses all contributors but its diagnostic IDs/offsets store
only the first20 per pixel; metadata-normalized coverage is not full coverage.
No frozen renderer or trainer was changed. The early patch graph still fails
persistent visible support under complete compositing; unknown IDs stay unknown.
`experiments/surface_support/assessment.md` records limitations and required
identity/representation evidence before any new shape/contact training.

## C1 local shape-coherence candidate (2026-11-02)

`variants/shape_coherence/prepare.py` copies C1 and changes only the optional
Tuning import. `model.py` estimates local relative F with rank-gated FP64 fits,
separately switchable affine and broad stretch/collapse terms, and no material
law. `observations.py` lifts manually reviewed early train-image components to
partial reference patch labels using corrected alpha*T contributions. Unknown
points never connect, and these labels are not physical material IDs.
`runtime.py` builds a frozen graph only after original densification and binds
its state to complete checkpoints; disabled flags delegate to original C1.
`audit.py`, `verify_update.py`, and `tests/test_shape_coherence.py` check the
train-only evidence, intended gradients, actual optimizer update, affine/rigid
behavior, label isolation and checkpoint integrity. `run.py` pauses at16k for
manual graph review before enabling the objective. Historical C1 is reused.
`output/shape_coherence/{continue_run,review_graph,finalize,verify_results}.py`
provide launch receipts, a coverage gate, locked-selection comparison and media
verification. No collision force, constitutive law or learned segmentation is
introduced. Early points lose visible contribution later in C1, so this weak
partial prior must not be presented as complete persistent object tracking.

## C1 optical-flow-only ablation (2026-11-02)

User correction: reuse historical C1; do not retrain the original. The former
new O0 was safely stopped and excluded. `output/c1_flow_ablation/reuse_control.py`
trains only O1 in a separate coordinator queue; `compare_reuse.py` reuses the
historical locked best/final snapshots and existing PNGs without altering them.
`reuse_control_plan.json` supersedes the original two-new-arms protocol below.

`variants/c1_flow_ablation/prepare.py` copies the frozen C1_repaired trainer and
renderer, changing only the optional runtime import and additive observed-flow
hook. `runtime.py` inherits C1 image/alpha losses, frame-index clock, independent
velocity/stress and original Cauchy momentum unchanged. Its single treatment
flag `observed_flow_enabled` toggles corrected raw train-only geometric and
independent-velocity flow supervision; no M1/N1 structural or momentum changes
are inherited. `run.py` branches the identical full pre-flow3k checkpoint into
two newly trained50k arms. `tests/test_c1_flow_ablation.py`, CUDA smoke/recovery
and `verify_update.py` validate flag isolation, parent delegation, and actual
flow-only optimizer updates. `results.py` locks validation selection before test,
exports GT|Render, evaluates the frozen V2 validation observation cache, checks
camera schedules, full checkpoints and rendered artifacts. Reports reside in
`experiments/c1_flow_ablation`; historical C1 is a descriptive reference only.

## Coherent discrete trajectory momentum (2026-11-02)

`variants/coherent_momentum/operator.py` derives interval chord velocities,
acceleration and the physics rollout from the same complete center trajectory.
It retains stress_pred and the original deviatoric rate term, using one
normalized quadratic current-space fit for the stress Jacobian / velocity
Hessian instead of successive linear fits. `runtime.py` gates the new operator
behind `coherent_momentum`, preserving M1's gradient policy and exact new-flag-off
path. `tests/test_coherent_momentum.py` checks analytic fields, degeneracy,
gradients and parent parity. `prepare.py` creates an independent trainer copy;
`run.py`, `evaluate.py`, `export.py` control recovery, validation selection,
common-anchor dual-operator diagnostics and GT|Render. The frozen old PIDG,
L1 and M1 trees are unchanged. A shared trajectory was already present in M1;
the new objective improves discrete compatibility rather than claiming a new
independent-velocity unification or exact physical integration.

## Momentum-gradient repair follow-up (2026-11-02)

`variants/momentum_guard/runtime.py` optionally guards the inherited shared
momentum loss with same-batch RGB gradients. `balance.py` defines the explicit
first-order per-group policy, leaving private stress/rate-coefficient gradients
unchanged; `tests/test_momentum_guard.py` checks the policy and exact guard-off
parent path. `prepare.py` copies the frozen unified architecture and changes only
the trainer runtime import. `run.py` / `verify_update.py` provide one matched
full-state experiment, smoke/recovery/flags-off and real optimizer checks.
`experiments/momentum_repair/{diagnosis,plan}.json` record evidence and prospective
rules. This adds no constitutive or external-force module.

## Unified trajectory/stress follow-up (2026-11-02)

`variants/unified_flow/code` is an independent frozen copy of the flow-comparison
trainer. Its optional `DeformNetwork.shared_stress` is Linear(width,6), stored in
deformation snapshots/full state. `model.predict_stress` evaluates the same
deformation Grid4D and exact shared trunk latent as `gaussian_warp`; full centre
velocity remains in `variants/flow_comparison/temporal.py`. `runtime.Tuning`
selects shared versus original predicted stress in the same local momentum
operator and integrates corrected raw source→target-camera flow. New optimizer
group, checkpoint extension, input hashes, matched controls and validation-only
selection are in `checkpoint.py`, `prepare_data.py`, `run.py` and
`experiments/unified_flow/plan.json`. Flags off preserve the original paths.
Original PIDG/K1/K2/source locks and weights remain unchanged. The additional
head has no constitutive or external-force interpretation.

The execution pack in `pidg_codex_plan.zip` is the current plan. The previous Phase
0–15 experiments were removed at the user's request and are not resumed.

The baseline is the released PIDG code at commit
`b77fd2107a50c875d7219bb4f82388d5e5ea1319`, under `third_party/PIDG`, with the
existing, disclosed Windows/release compatibility patches. It is not the separate
CPU research scaffold in `src/physgauss`. No new physics is added to the original tree.

| Concept | Actual file / symbol (relative to `third_party/PIDG`) |
|---|---|
| Training entry | `train_pidg.py::training`, CLI at file end |
| Configuration | `arguments/__init__.py::{ModelParams,OptimizationParams,PipelineParams,merge_config}`, `arguments/{pidg,dnerf,hypernerf,neu3d}/*.py` |
| Canonical Gaussian model | `scene_PIDG/gaussian_model.py::GaussianModel` |
| Deformation field | `scene_PIDG/deform_model.py::DeformModel.step`; `scene_PIDG/network.py::{Grid4D,DeformNetwork}` |
| Velocity field | `motion_utils/pidg.py::VelocityNet.velocity_head`, `GaussianModel.get_velocity_and_stress` |
| Predicted stress/material field | `VelocityNet.stress_head`; `GaussianModel.force_sigma`; `compute_stress_divergence` adds a velocity-gradient stress correction |
| Lagrangian trajectory / flow | `utils/flow_utils.py::motion_flow_velocity`, called in `train_pidg.py` every 100 iterations after warmup when `use_flow` is true |
| Cauchy momentum residual | `motion_utils/pidg.py::{cauchy_momentum_constraint,compute_acceleration,compute_stress_divergence}`; `train_pidg.py::cauchy_momentum_loss_sampled` |
| Renderer | `gaussian_renderer/__init__.py::render`, installed flow-capable CUDA rasterizer |
| Dataset loaders | `scene_PIDG/__init__.py::Scene`, `dataset_readers.py::sceneLoadTypeCallbacks`; `cameras.py::Camera` |
| Evaluation | `render.py::{render_sets,render_set}` then `metrics.py::evaluate` |
| Inference checkpoint | `Scene.save`, `GaussianModel.save_ply/load_ply`, `DeformModel.save_weights/load_weights` |
| Full training checkpoint | `reproduction/scripts/training_checkpoint.py::{save_training_state,restore_training_state}`, called by patched original trainer |

## Verified datasets and protocol

The local five PIDG scene directories are `rubber_duck`, `balls_reaction`,
`mechanics_cloth`, `motion_kuro`, `dry_ice`. The released Rubber Duck **config** is
spelled `arguments/pidg/rubber_dack.py`; the dataset remains `rubber_duck`.
D-NeRF has 8 config files, HyperNeRF 11, and Neu3D 6. Only PIDG and the independent
toy synthetic data are currently installed. A config's existence does not imply
its dataset, material annotations, future trajectory GT, or force GT is available.
The experiment matrix records the exact config names and local availability.

Five completed original baselines exist in `runs/pidg_official/<scene>`: 50,000
iterations, width 1600, seed 0, official train/test splits, original best-test-PSNR
snapshot selection and official PSNR/SSIM/VGG-LPIPS/MS-SSIM evaluation.
On the user's instruction, M0 reuses these runs instead of training again.

## Important implementation conventions

- `d_xyz` contains quaternion + translation, **not** a displacement. Rendered
  centers are `R(q) X + b`; covariance rotation/scale offsets are separate.
- Six stress components are ordered `[xx, yy, zz, xy, xz, yz]`. The full Cauchy
  stress includes the original `force_sigma * dev(sym(grad(v)))` correction.
- CMR uses canonical positions and `fid / train_count`; rendering uses `fid`.
  The legacy convective contraction and sampled `M/N` weighting are retained in A0.
- Rubber Duck's official config has `use_flow=False`. Cloth is for reconstruction
  only in the new solid Neo-Hookean extension; Dry Ice and Motion Kuro are also
  ineligible for that solid loss. These settings must not be silently changed.
- The original release imports a missing material file. Existing compatibility
  patches use the released `motion_utils.pidg.VelocityNet`; identity with the
  missing implementation cannot be proven. Other existing patches support MSVC,
  camera device transfer, released mask paths, material snapshot loading and full
  checkpoints. See `reproduction/README.md` and the captured baseline patch.
- Historical training did not capture runtime file hashes. M0 validates completed
  outputs and captures current source/data hashes; it does not retroactively prove
  historical source identity or claim paper-table reproduction.

Material/force recovery and future-prediction errors remain unavailable without
appropriate ground truth. Synthetic tests must be labeled separately from official
scene measurements.

## Metric tuning (separate fork)

`variants/metric_tuning/code` copies the audited original runtime. Its optional
hooks live in `variants/metric_tuning/{model,runtime,checkpoint}.py`; preparation
is recorded in `source_manifest.json`. The original trainer, deformation network,
renderer and data loaders remain at the locations above. The fork adds a
disabled-by-default residual adapter to `DeformNetwork`, optional training-image
objectives, and four-metric no-grad checkpoint selection. New runs/metrics stay in
`runs/pidg_metric_tuning` and `experiments/metric_tuning`. No baseline files or
previous experiment weights are edited.

## Isotropic Gaussian shape regularization (separate fork)

`variants/isotropic_tuning/model.py` defines the dimensionless principal-axis
isotropy penalty. `runtime.py` subclasses the frozen metric-tuning runtime;
`prepare.py` copies its code into a new `code/` directory and adds one optional
training-loss hook. The renderer remains unchanged: the default loss uses
`abs(gaussians.get_scaling + d_scaling)` and follows the canonical-only path when
Python covariance is enabled. Default weight zero preserves the T3 loss path;
all tuning disabled preserves the baseline path. New configs, validation logs
and results live under `experiments/isotropic_tuning` and
`runs/pidg_isotropic_tuning`. This is a Gaussian shape prior, not a material law.

## Shared time and kinematic consistency (separate fork)

`variants/kinematic_tuning/code` independently copies the frozen metric runtime.
`timeline.py` keeps the legacy `fid/train_count` physics path when disabled;
`shared_physics_time` switches the physics input to the same normalized `fid`
used by deformation/rendering. An additional `timeline=frame_index` option maps
the verified 200 numeric filenames jointly to `(frame_id-1)/199`, without editing
source JSON or changing the 150/50 split. Uniform capture cadence is an explicit
assumption; these retimed tests must be compared separately from historical
metadata-time metrics. Real elapsed seconds remain unknown.

`model.py` differentiates the full rendered center `R(q(X,t)) X+b(X,t)` using
second-order finite differences, including one-sided endpoint stencils.
`runtime.py` adds optional joint Huber consistency with `VelocityNet(X,t)`;
the canonical sample positions are detached, but deformation-grid, deformation
head and velocity-network parameter gradients remain enabled. This avoids
depending on unsupported coordinate second derivatives in the custom hash CUDA
encoder. The legacy canonical-coordinate momentum residual, convective
contraction and sampled weighting are preserved. `kinematic_weight=0` disables
the new loss, and all tuning flags off preserve the original training path.
Runs, complete resumable checkpoints, metrics, diagnostics and GT/render pairs
are isolated in `runs/pidg_kinematic_tuning` and `experiments/kinematic_tuning`.

## Time-disjoint validation / test retraining (separate fork)

`variants/kinematic_resplit/code` copies the frozen kinematic source. Its optional
`runtime.py::Tuning` retains the V3/V4 physics and uses only validation cameras
for checkpoint selection. `data.py` prepares 120 training / 30 validation / the
unchanged 50 official test acquisitions with a common frame-index clock. The
training loader's legacy test slot holds validation; final-test RGB is absent
from that loader's data directory. A separate view is used for final evaluation.
No old weights, released point cloud, flow or depth are reused. A new train-only
50k parent initializes V3/V4, and `run.py::branch` copies an identical complete
1000-step refinement state before activating V4 consistency. Selections are
locked before final test rendering. Historical use of the official test set
remains disclosed; the split does not retroactively remove tuning exposure.
New logs, checkpoint state, metrics and pairs live in
`runs/pidg_kinematic_resplit` and `experiments/kinematic_resplit`.

## V3 plus Gaussian isotropy (separate fork)

`variants/v3_isotropic/code` copies the completed resplit source. Its optional
`runtime.py::Tuning.regularization` reuses the tested normalized principal-axis
penalty from `variants/isotropic_tuning`, while inheriting the validation-only
evaluation and shared clock from `variants/kinematic_resplit/runtime.py`.
Two additional50k continuations start from the same completed V3 weights: iso
weight zero versus 0.001, with a byte-identical full-state branch at extra1000.
The only new trainer operation is the optional isotropic loss hook; kinematic
consistency stays off. Original resplit data and the frozen V3 parent stay
unchanged. New logs/results/charts live in `experiments/v3_isotropic`, and weights,
final renders and pixel-verified GT/render pairs in `runs/pidg_v3_isotropic`.

## Fresh V3 / V3+iso initialization (separate fork)

`variants/v3_iso_scratch/code` copies the optional-iso source and adds a flagged
step-zero capture hook before the training loop. `parent_run` is empty; new
Gaussian and neural parameters are initialized from random geometry and seed0.
The capture contains zero optimizer updates, verified by empty optimizer state
dictionaries. Both arms copy identical complete checkpoint bytes before step1;
only iso weight differs. They each train50k total, with original initial-training
warm-up, densification and learning rates, V3 image objectives/shared time, and
iso weight0 or0.001 from step1. Existing validation-only selection and final-test
locking are reused. The prior continuation is saved and stopped at the user's
request; no trained weights from it are used. New outputs are isolated in
`experiments/v3_iso_scratch` and `runs/pidg_v3_iso_scratch`.

## Ordered foreground coverage ablation (separate fork)

`variants/coverage_tuning/code` copies the frozen scratch source. Only optional
loss and post-optimizer density hooks are added to its trainer; an environment
flag enables diagnostic alpha PNG export in its renderer. The original RGB
compositor and all existing source trees remain unchanged. `data.py` audits and
reads original RGBA alpha from the120 training frames only. `model.py` defines
foreground alpha L1 and a weak local affine neighborhood fit allowing stretch
and shear. `density.py::CoverageAccumulator` collects unique-training-time
coverage evidence. `runtime.py::Tuning` integrates the losses, conservative late
point additions, and checksum-bound coverage checkpoint state. Render inspection
confirmed multiple ducks and obstacles; the earlier single-object assumption was
incorrect. Current real-scene affine neighborhoods have distance gating but no
object-ID filtering. Explicit known IDs are required for object-isolated contact
constraints. `run.py` orders mask, mask+affine,
and mask+affine+density50k scratch runs from the same random step-zero checkpoint,
reuses the completed V3 control, selects on validation only and locks selections
before final testing. Iso stays off. New diagnostics include foreground PSNR,
mask IoU and eroded-interior low-alpha rate. New outputs live exclusively in
`experiments/coverage_tuning` and `runs/pidg_coverage_tuning`.

## Standalone component effects (separate follow-up fork)

`variants/coverage_isolated/code` copies the frozen coverage trainer with only
the Tuning import changed. `runtime.py` retains the exact affine formula and
inherited density module, sets mask loss to zero, removes separate alpha targets
from the affine-only path, and replaces the disproved single-object declaration
with explicit acknowledgment of unsegmented neighbors. C4 enables affine only;
C5 enables density only and still uses training alpha to select additions.
`run.py` waits for the existing GPU queue, runs GPU preflight, then trains both
arms50k from the identical random zero state. Reports compare C4-C0 and C5-C0
alongside all existing cumulative arms. New files remain in
`experiments/coverage_isolated` and `runs/pidg_coverage_isolated`.

## Whole-scene foreground centroid (separate fork)

`variants/centroid_tuning/code` copies the frozen coverage hook fork and changes
only its Tuning import. `model.py` computes differentiable normalized 2D alpha
centroids; `runtime.py` applies an optional train-only centroid-distance loss.
The default original-PIDG recipe has no perceptual training loss, shared physics
clock, mask L1, affine prior, isotropic term, velocity consistency, or late
coverage additions. It uses the established clean120/30/50 data protocol and
its metadata rendering clock; original conventional densification remains.
This is not a physical3D center-of-mass estimator and has no real object-ID bank.

`run.py` captures a random full step0 state, trains a common3k prefix, then
branches identical tensors/optimizers/RNG/camera stacks into G0 and G1. Only
centroid_weight changes, with a2k ramp after3k and50k total steps per arm.
Validation-only selection is locked before final-test rendering. Completion
exports four image metrics, alpha/centroid diagnostics and pixel-verified GT
pairs. New outputs remain in `experiments/centroid_tuning` and
`runs/pidg_centroid_tuning`; historical controls are preserved.

## Coverage color repair (separate fork)

`variants/color_repair/prepare.py` copies the frozen coverage runtime, changes
the Tuning import, and inserts one opt-in renderer color branch. `color.py`
evaluates SH in deformed viewing directions and `model.py` maps to smooth
positive radiance. Flags-off retains the original CUDA SH path. `runtime.py`
adds delayed/ramped mask and train-foreground RGB losses, weaker spatial-affine
regularization and conservative training-color-aware coverage children;
`evidence.py` stores nonlearned color moments in checkpoint-bound sidecars.
No object-ID isolation, learned segmentation or physical mass claim is added.

`diagnose.py` only reads original final checkpoints and predetermined training
frames. `run.py` trains repaired C1/C2/C3 to50k from new random initialization,
with a shared3k prefix and exact shared15k C2/C3 pre-density state. Validation
selects snapshots before locked final testing. Original models, data split and
completed reports remain preserved. New outputs live in `experiments/color_repair`
and `runs/pidg_color_repair`.
# C1 shared effective-stiffness ablation

`variants/c1_material/prepare.py` copies frozen C1 repair code and changes only
the optional Tuning import. `model.py::SharedYoung` adds one bounded material-
shared E with fixed Poisson ratio. `runtime.py` reuses `pidg_ext` F estimation,
Neo-Hookean stress and robust consistency while retaining PIDG predicted stress.
Its canonical basis is detached; trajectory, stress and shared E receive gradients.
The new flag-off path returns the C1 runtime loss without new RNG/optimizer work.

All three runs branch the same complete pre-material C1 random-init3k state and
finish at total50k. Without known material IDs the scope is explicitly a global
effective prior, not real material identification. E is in normalized units, not
Pa. E/optimizer/canonical graph are checksum-bound to full checkpoints. Conditional
profiles cannot certify joint identifiability. New code/tests, reports and outputs
are isolated in `variants/c1_material`, `experiments/c1_material`, and
`runs/pidg_c1_material`. Old source, weights and test-selection files are preserved.

## Object-center candidate review (offline, no training integration)

`variants/object_centers/audit.py` reads frozen C1 repair final50000 and only
12 predetermined training frames. `grouping.py` compares fixed canonical,
trajectory/appearance and motion-dominant K-means candidates. IDs are fixed per
kernel; unknown kernels remain -1. `model.py` provides separately tested center,
zero-mean residual, piecewise-linear trajectory, topology inheritance and soft
overlap primitives. These are not inserted into C1 training or inference.

`experiments/object_centers/grouping_review.json` and the local gallery under
`runs/pidg_object_centers/grouping_review/c1_final50000` record the source hashes,
candidate IDs, GT/C1/group panels and measured center curves. Visual review
finds merged/split objects; clusters are not accepted as contact labels.
There is no new center-model long training or material segmentation.

`variants/object_centers/motion_grouping.py` additionally compares pure
displacement, displacement plus time-correct trajectory finite-difference
velocity, and motion-group refinement using persistent spatial connectivity.
`motion_review.py` reuses the original train-only evidence with SHA checks;
rendered C1 RGB and deformed poses are verified against that evidence before
exporting a separate motion-method gallery. Both baseK=5 and6 are inspected.
Spatial refinement can yield more components than baseK; these remain
unverified candidates, not accepted object/contact labels or new training.

`variants/object_centers/adaptive_grouping.py` removes the requested count
entirely: displacement HDBSCAN, displacement/velocity HDBSCAN, and a direct
motion/spatial graph with no base clustering. Minimum cluster sizes128/256,
min_samples and spatial/motion thresholds remain explicit parameters.
`adaptive_review.py` exports these no-K candidates to their own gallery using
the same train-only cache and immutable C1 snapshot. Shared export helpers now
accept an output directory and comparison field; previous default baseK paths
are retained. No center/contact loss is integrated into training.

`variants/object_centers/color_features.py` supplies an opt-in cached predicted
mean-RGB block for offline grouping. Density color_weight=0 and graph
color_radius=None retain the no-color algorithm paths. `color_review.py`
compares the three no-K methods with color enabled against their saved
motion-only candidates. Features use existing C1 predictions at training
views, not GT/test colors. A new gallery includes per-method before/after GT/C1
panels; C1 model weights, architecture and training losses remain unchanged.

## Trainable object identity / motion bottleneck (separate C1 fork)

`variants/object_id/prepare.py` copies frozen repaired C1 and inserts opt-in
deformation/state hooks. `model.py::ObjectMotion` maps canonical XYZ to soft
candidate membership and normalized time to group-shared SE3 center motion.
The original nonrigid trajectory is retained with a ramped25% grouped blend;
covariance rotation/scale, PIDG velocity/stress and C1 image/alpha losses remain.
The16-candidate ceiling is not a target object count; occupancy/pruning are
explicit diagnostics, with no GT instance or offline cluster label targets.

`runtime.py` adds separate optimization, RGB-gradient audits, auxiliary motion/
assignment losses and SHA-bound full-checkpoint sidecars. Selected deformation
snapshots contain matching object state. Disabled hooks return the original
warp and allocate no object optimizer. `run.py` compares C1, frozen identity
and learned identity from the identical retained full3k random-init checkpoint,
total50k per arm, validation-only selection and locked final testing. Reports
and new runs are isolated in `experiments/object_id` and `runs/pidg_object_id`.
This is a custom paper-inspired reconstruction ablation, not physical instance
ground truth, material segmentation or guaranteed nonpenetration.

## Frozen trajectory / velocity-only diagnostic (standalone, 2026-11-01)

`variants/frozen_velocity_probe/run.py::{export,train}` reads the H3 final50k
PLY/material/deformation snapshot, caches detached derivatives of the full
rendered center using `variants/kinematic_tuning/model.py`, and fits diagnostic
copies of the existing PIDG VelocityNet. No original training hook is changed.
Canonical coordinates and deformation/grid states stay frozen; the only fit
objective is velocity MSE, without momentum, force, image or constitutive loss.
`core.py` partitions cloned particle-ID families before selecting held-out
Gaussian rows, freezes the unused stress-head parameters, and reports amplitude,
direction and RMS alignment. Shared-feature updates can alter unused stress
outputs, so these velocity-only copies are not drop-in production stress models.
`report.py` and `collisions.py` inspect fitting/generalization and scalar-time
encoding aliases; the latter is a post-hoc frozen-teacher mechanism check.
`clock.py::apply_clock` defaults to a legacy no-op; its opt-in fixed monotonic
clock is tested in `time_control.py` with the same cached labels and fit budget.
Config/protocol/results live in `experiments/frozen_velocity_probe`; cached
targets and independent velocity checkpoints in `runs/pidg_frozen_velocity_probe`.
Disabled execution exits without model imports, checkpoint access or optimization.

## Optional transfer-integrity reconstruction experiment

`variants/transfer_integrity/prepare.py` forks repaired C1 by changing only the
Tuning import. `runtime.py::Tuning` delegates when its new flag is disabled;
the enabled path retains original image/physics training and adds real-view
coverage plus partial static and short-track geometry constraints.
`static_audit.py` tests a blue-region static hypothesis with independent image
tracks and held-observation triangulation. `data.py` binds conservative rows
using complete native-resolution contributions and frozen training-only flow.
`model.py` contains differentiable reconstruction objectives, not a new physics
module. `run.py` preserves the full C1 optimizer/RNG/sampler state for matched
continuations and verifies flags-off parity and enabled checkpoint recovery.
`experiments/transfer_integrity/evaluate.py` evaluates final full states and
reuses the frozen-camera renderer without changing historical artifacts.

## Full monotonic-clock / kinematic training (independent H3 fork)

`variants/monotonic_training/prepare.py` copies H3 and changes only the Tuning
import, the VelocityNet internal clock call, and the Gaussian constructor's
configuration hook. The renderer, deformation field, data loader and camera
sampler remain byte-identical to H3. `model.py::time_coordinate` preserves the
original Fourier/projected path by default; opt-in `velocity_clock_mode` selects
`tanh(2*t-1)`, retaining every original parameter and optimizer group while
freezing unused learned clock parameters. Both velocity and predicted stress
heads use the same existing features. Full resume retains the mode through the
checksum-bound run config; snapshots require that config when loading physics.
`runtime.py` reuses H3 and its existing joint full-center kinematic loss.
`run.py` performs GPU smoke/resume/flags-off checks, copies the identical full3k
state, trains to total50k, locks validation selection, then renders final test.
`export.py` checks pixel-identity GT|Render concatenations and independently
recomputes PNG PSNR before producing videos/gallery/archive. Experiments are in
`experiments/monotonic_training`; runs in `runs/pidg_monotonic_training`; user
deliverables in `output/monotonic_training`. No external force is introduced.

## R0 partial static pool (2026-11-03)

`variants/static_pool/prepare.py` forks repaired C1 without editing original
PIDG or historical experiments. The only trainer changes are its Tuning import
and optional exclusion of accepted static rows from dynamic CMR sampling.
The renderer calls `render.apply_warp` before computing centers/covariances;
flag-off returns the original objects. `render.StaticPool` binds immutable,
training-only reference warps to a specific source row/particle map and rejects
topology/checksum changes. Static geometry and SH remain trainable; no new
particles, segmentation or material/force parameters are introduced.
`runtime.Tuning` binds full-state resume and logs matched camera samples.
`bind`, `probe`, `native_check`, `run` implement source-bound references, native
GPU gradient/parity/resume gates and matched15k→50k arms. `pipeline` serializes
training, native fixed-view evaluation, decoded exports and numeric plan status;
visual review is separate from structural invariance. Plans/evidence live in
`experiments/static_pool`, outputs in `output/static_pool`, states in
`runs/pidg_static_pool`. Adaptive decisions are recorded in
`experiments/adaptive_reconstruction_plan.md`; the original physics ladder is
not renumbered or bypassed.

## S1–S6 staged reconstruction (2026-11-03)

`variants/staged_reconstruction` is opt-in and leaves every existing trainer and
loader unchanged. `camera_audit` checks train-only track hypotheses;
`prepare_data` builds conditional static masks and multitime silhouette hull in
metadata CV coordinates. `static_gaussian.StaticGaussian` is an independent
parameter bank consumed by the unmodified C1 renderer. `train_static` trains
only static geometry/SH/opacity, saves full optimizer/RNG/camera state, and
evaluates candidate static regions at native resolution. No deformation,
velocity, constitutive or contact parameter is introduced in S1. `freeze`
protects S1 sources/inputs and inherited baseline files. Flags-off never enters
the new renderer path. This camera contract is specific to the new branch;
historical legacy-camera weights/metrics are not a matched control for S1.
Stages and dependencies remain in `experiments/staged_reconstruction_plan.md`.

## Train-only adaptive static geometry (2026-11-03)

`variants/staged_reconstruction/adaptive_static` is a separate opt-in branch.
`geometry` implements camera-derived depth bounds, normalized-patch/chromatic
plane sweep, source-depth/round-trip checks and independently checked planar
priors. `prepare` fuses verified observations into an unrestricted surface
reservoir and common coarse initialization. `model.AdaptiveGaussian` and
`topology` append/prune evidence-supported rows while retaining existing Adam
moments and counters. `train` compares geometry-only and geometric-birth arms,
reusing the frozen confidence/objective and unchanged native PIDG renderer.
`audit` binds source/input hashes, checks full state and replays sampler and
topology. All new entry points require `--enable-adaptive-static`. Inputs have
no known static color, instance count or material segmentation. No baseline
trainer, field, renderer, loader, checkpoint or prior experiment is modified.

## Conditional visibility quarantine (2026-11-03)

`variants/staged_reconstruction/clean_static` is a separate opt-in branch.
`evidence` distinguishes front-of-surface contradictions from occlusion and
missing evidence. `prepare` queries inherited train-only depths without colors,
classes or object counts. `probe` renders a reversible initial-row quarantine.
`train` continues full G1 Adam/RNG/sampler state, comparing unchanged and
quarantined static rows with the same native objective and camera sequence.
`audit` checks parent/subset exactness, Adam moments/counters and absolute
iteration/sampler recovery. `evaluate` uses common frozen proxy masks, native
PNGs/PLY and decoded fixed-camera/original-trajectory videos. Baseline entry
points, renderer, physics fields and historical sources remain unchanged.

## Independent RGB admission audit (2026-11-03)

`variants/staged_reconstruction/synthetic_audit` ray traces a controlled static,
transient and glossy scene with independently varied camera/time. `input.load`
accepts only RGB/calibration/time; generated static RGB/depth/normal/visibility
are isolated under evaluation_only. `infer` reuses frozen native-branch MVS
functions and conditional ray quarantine without GT; `evaluate` independently
scores surface precision and visible/thin surface coverage. This is a point
admission audit, not a full Gaussian training or PIDG physics reproduction.
Data/input/source checksums and sparse-v1/dense-v2 sampling reasons are retained.

## Native per-image nuisance decomposition (2026-11-03)

`variants/staged_reconstruction/nuisance_static/core.py` defines opt-in
premultiplied composition and camera-plane Gaussian appearance explanations.
`runtime` adapts calibrated cameras and calls the unchanged PIDG rasterizer.
`prepare` builds common initialization/support only from train-RGB inference.
`train` uses full-RGB and restricted support loss, one independent Adam per
image, complete static/nuisance optimizer/RNG/sampler checkpoints and user-stop
saving. `native_check` checks disabled parity and native RGB/alpha/position
gradients. `audit` freezes code/inputs and replays global/per-view state.
`evaluate` alone reads independent synthetic static RGB/depth/normal GT; it
uses expected depthD/alpha and treats view-specific nuisance as appearance,
not physical identity. Real training is gated on both full synthetic cases.

## Native per-image nuisance decomposition (2026-11-03)

`variants/staged_reconstruction/nuisance_static/core.py` defines opt-in
premultiplied composition and camera-plane Gaussian appearance explanations.
`runtime` adapts calibrated cameras and calls the unchanged PIDG rasterizer.
`prepare` builds common initialization/support only from train-RGB inference.
`train` uses full-RGB and restricted support loss, one independent Adam per
image, complete static/nuisance optimizer/RNG/sampler checkpoints and user-stop
saving. `native_check` checks disabled parity and native RGB/alpha/position
gradients. `audit` freezes code/inputs and replays global/per-view state.
`evaluate` alone reads independent synthetic static RGB/depth/normal GT; it
uses expected depthD/alpha and treats view-specific nuisance as appearance,
not physical identity. Real training is gated on both full synthetic cases.


## Optional RGB depth evidence refinement (2026-11-03)

`variants/staged_reconstruction/depth_refine/core.py` implements a default-off
bounded inverse-depth proposal and separate held-source acceptance; disabled
calls return the original depth tensor unchanged. `infer.py` preserves the
inherited candidate set and reads only RGB/calibration/time plus saved RGB-MVS
depths, with per-view save/stop and input/source locks. `evaluate.py` alone reads
independent static/depth/normal GT after inference, compares fixed candidates
and fixed original normal-valid pixels. No Gaussian optimizer, field, renderer,
loader or historical checkpoint is altered. `tests/test_depth_refine.py` checks
the vertex, bounds, invalid/zero steps, held-source refusal and flags-off path.


## Optional RGB source-depth interval evidence (2026-11-03)

`variants/staged_reconstruction/depth_uncertainty/core.py` observes the original
six RGB-derived source depths, counts distinct poses and constructs empirical
depth envelopes with an inverse-depth quantization floor. Missing/closer-surface
samples retain unknown status. Disabled calls return the original depth object.
`infer` preserves all D1 candidates/depths with per-view stop/lock/source checks.
`evaluate` alone reads GT and measures reliability, width and visible/thin
retention; `prepare` binds tests, protocol, parent arrays and frozen source.
This is a geometry-evidence diagnostic, with no change to PIDG models, fields,
renderer, optimizer or physical identities. Tests: test_depth_uncertainty.py.


## Optional fixed pretrained RGB observed-depth evidence (2026-11-03)

`variants/staged_reconstruction/pretrained_depth/model.py` lazily loads the
hash-verified official Small model with weights_only and unchanged518 BGR
preprocessing; disabled calls retain object identity without model import.
`core.py` uses disjoint spatial train-anchor checks and Theil-Sen inverse-depth
calibration, retaining failed views as unknown. `infer/prepare/replay` freeze
RGB/anchors/upstream source and replay official weights in a separate process.
`evaluate` alone imports analytic GT ray tracing, checks static/thin/observed
depth, true empty background and same-pose time overlap. No original PIDG
renderer, Gaussian, optimizer, physics field or identity is changed. Tests:
test_pretrained_depth.py. Reports: S1_pretrained_geometry/v1; not Gaussian training.


## Optional RGB-validated relative-depth proposal (2026-11-03)

`variants/staged_reconstruction/rgb_prior_proposals/core.py` uses frozen raw
Small ordering and D2 interval anchors to nominate alternative192-grid depths;
the unchanged patch_features/source_cost and fixed4-fit/2-held checks accept
or retain old D1 depths. No learned static label or physical identity is made.
`compute/infer/io` retain full membership with atomic hash-bound per-view saves;
`preflight` verifies independent2+2 inference recovery and exact disabled identity.
`audit` replays every saved input/bound/acceptance and actual native RGB costs in
a separate process; `evaluate` alone reads static GT and visible-thin labels.
No PIDG renderer/model/optimizer/physics or Gaussian training integration.
Tests: test_rgb_prior_proposals.py; reports: S1_rgb_prior_proposals/v1.


## Opt-in RGB local-plane support warp (S1-D6, 2026-11-03)

`variants/staged_reconstruction/rgb_plane_warp/core.py` fits/checks original D1
inverse-depth slopes from disjoint local neighbors, constructs plane_support,
and samples raw RGB before patch normalization in warped_cost. propose keeps
all parent depths unless the fixed fit/held equation accepts; disabled calls
return the same parent object. compute.py retains full slope/cost/rejection
diagnostics. infer.py is locked/default-off; preflight.py checks both conditions
in three independent processes, full4 versus2+2 with identical saved arrays.
prepare.py freezes409 train-only input/source files. audit.py replays192 parent/
slope/acceptance states and six fixed full RGB views; evaluate.py is independent
GT-only assessment. No original PIDG trainer/renderer/weight or physics field
changes. D6 completed with limited small-error gains, failed geometry gates,
and is not admitted to Gaussian training. Reports under S1_rgb_plane_warp.
S1-D7 is implemented and completed; see the following map section.


## Opt-in cross-view source-plane transport (S1-D7, 2026-11-03)

`variants/staged_reconstruction/plane_transport/core.py` queries original first4
fit-source anchors, transforms original D1/D6-admitted planes and intersects
target rays inside source5x5 footprints. Target slope admission is unnecessary;
source checks and frozen raw-RGB warp/fit/held equations are unchanged. Disabled
calls return the same parent object. inputs/compute retain all queries, source
planes, rejected hypotheses and parent rows. infer is locked/default-off;
preflight checks independent full4 versus2+2 for both conditions. prepare binds
809 train-only/source files. audit replays192 states plus six full RGB views;
evaluate alone reads static/visible-thin GT. Independent export_review replays
aggregate counts/MAE/P90 and original GT hashes; no GT feedback. 40 tests pass.
D7 has local depth gains but fails both gates, not integrated into Gaussian
training. No PIDG trainer/renderer, frozen source, weight or physics changes.
Reports: S1_plane_transport/v1. Next J0 is preregistered no-update full-RGB
static-opacity/SH gradient audit; no joint-training implementation claimed.


## No-update native static-revocation audit (S1-J0, 2026-11-03)

`joint_revocation_audit/core.py` temporarily enables static opacity/SH autograd
and restores flags/values/leaf gradients/RNG, including exceptions. worker uses
the pinned historical S3-U scene and unchanged native renderer/full RGB loss;
no optimizer or sampler updates. Independent receipt wrapper handles virtual
Torch source paths without editing the frozen worker. preflight reloads in two
processes; export_review checks six bindings/hashes/sign denominators/54 images.
finalize records manual old-native-image review and clears pending state under
both locks. 9 core+2 repair tests pass, gradient path qualified; geometry and
cleanup unproven. This is not newly trained unknown-static reconstruction.
All original PIDG/weights/paused stages unchanged. Reports: S1_joint_revocation_audit/v1.


## Opt-in RGB-only joint Gaussian training (S1-J1 and accepted recovery)

`variants/staged_reconstruction/joint_rgb/runtime.py` uses the original PIDG
deformation/hash-grid and single depth-sorted union with a4000-row unknown-count
canonical dynamic bank. The only paired switch opens static opacity/DC/rest SH;
static geometry stays frozen. core is default-off; train saves both complete
Adam states, RNG, camera stack/visits/trace and immutable source/protocol binding.
Original40-step cross-run checks failed and stay archived; the fixed epsilon
control also failed. No source/protocol/failure was overwritten.

`S1_joint_rgb/accepted_recovery/launch.py` redirects paths only, leaving frozen
J1 training/model code unchanged. Human-approved acceptance checks exact full
serialization/bindings and same-native-gradient Adam update in two independent
processes for each of four case-arms, plus native RGB/alpha equality and raw
gradient tolerance. All four pass; it does not claim deterministic40-step native
training. Formal training uses original epsilon/rates and quality gates. A
serial runner independently evaluates/exports, then waits for manual image and
geometry review. No physics, original PIDG, renderer or old weight changes.


## Frozen joint opacity/SH attribution (S1-J2)

`experiments/staged_reconstruction/S1_joint_attribution/audit.py` is an explicit,
default-off independent evaluator of four fixed opacity/SH interventions. It
reuses the unchanged native static renderer on pinned J1-A checkpoint values;
`core.py` closes all/novel static and thin ray-transition denominators and
symmetric marginal RGB contributions. `independent_review.py` re-renders all64
cameras without importing that accounting or mixer. Native SH-only alpha/depth
invariance and two-process exact output replay pass; no optimizer/model update,
GT training assets or identity estimates. Tests: `tests/test_joint_attribution.py`.
J1-A appearance improves but remains unqualified for real training after ray
accuracy loss. This maps a read-only reconstruction diagnostic, not new physics.


## Default-off native parent depth self-reference (S1-J3)

`variants/staged_reconstruction/joint_depth_trust/core.py` adds one robust
expected-depth self-reference on fixed parent-render support; disabling returns
the original JointScene/loss objects. `runtime.py` keeps the immutable RGB-derived
teacher and deterministic per-camera cache outside model/Adam state; the new
term connects only static opacity. The parent is not independent geometry GT.
`experiments/staged_reconstruction/S1_joint_depth_trust/launch.py` adapts the
frozen original trainer in-process, qualifies native finite differences and exact
full-state/fixed-gradient recovery, trains two candidates, and reuses pinned
historical matched controls without rerunning them. Independent evaluation/
export retain original quality gates. Tests: `tests/test_joint_depth_trust.py`.
No original PIDG code, physics or historical weights change; no promotion yet.

## Default-off Rubber Duck paired RGB joint pilot (J4)

`variants/staged_reconstruction/rubber_joint/core.py` opens only static opacity/SH
when explicitly enabled; the original `UnionScene.render` and frozen PIDG
renderer remain untouched. It reuses the frozen J3 opacity-only parent-depth
self-reference. `runtime.py` loads the pinned complete S3-U parent plus the
exactly matching S1 static Adam, retains complete dynamic Adam/RNG/sampler and
physics graphs, freezes velocity/stress, and saves/restores both optimizers and
bindings. This is RGB reconstruction, not original physics continuation; inherited
color-mask training lineage is disclosed. `RubberDuck_joint_pilot/launch.py`
freezes train inputs, qualifies original native output/gradients and independent
exact full-state/fixed-gradient Adam recovery before serial matched2000 updates.
`evaluate.py` independently audits all initial/final states and30 validation
images, native fixed86/188 early/mid/late RGB/alpha/D, PLY and fully decoded
H264. Fixed-view cross-time has no independent GT; teacher is a self-reference.
Tests: `tests/test_rubber_joint.py`; default off has original object identity.
No topology/identity/semantic masks/GT or physics modules are added.

### J5 Rubber Duck read-only visibility audit

- `variants/staged_reconstruction/rubber_visibility/{core,worker}.py`: diagnostic
  subset/denominators, exact camera-time census and original native override-color
  visibility on three pinned states; no optimizer or baseline changes.
- `experiments/staged_reconstruction/RubberDuck_visibility_audit/`: frozen plan,
  inputs, preserved failed evaluation fixture, independent repair and metadata/
  fixed-source replay, actual image review and final assessment. Native first20
  IDs select a capacity-limited union; post-T exported weights are never treated
  as compositing contributions. Historical color-conditioned parent disclosed.
- `tests/test_rubber_visibility*.py`: six tests. Default off; full state unchanged.
  This maps a diagnostic only, not new Gaussian training or physics integration.

### J6 I10 dynamic plus accepted J4 static joint pilot

`variants/staged_reconstruction/i10_joint/{core,runtime}.py` composes only
static model/Adam from J4 with I10 non-static model/Adam/RNG/sampler/graphs.
`experiments/staged_reconstruction/RubberDuck_I10_joint/launch.py` is an opt-in
in-process adapter to frozen J4 full-RGB training; no source/history edits.
Eleven tests, native flags-off/gradient checks, two 20-update smoke arms and
two-process full-state/fixed-native-gradient Adam replay passed. Static geometry
and physics remain frozen; historical color-conditioned lineage is retained.

J6 evaluation: `RubberDuck_I10_joint/{evaluate,audit,finalize}.py` records
all30 native RGB, fixed86/188 decompositions, full Gaussian/XYZ PLY and
200-frame decoded H264 comparisons. CPU all-field PLY/state/sampler audit
passes; sampled dynamic cleanup gate fails. No frozen training-source edits.

## Mask-free I10 descendant persistent 3D tracking (T0-T2)

`variants/staged_reconstruction/i10_tracking/core.py` provides full-image
channel-symmetric texture proposals, bidirectional image gates, camera-motion
compensated stationary compatibility, persistent Gaussian-ID fitting/held
validation and separated-3D-alternative ambiguity. `extract.py` uses the pinned
local CoTracker3; no appearance or alpha foreground/static masks. `bind.py`
loads the immutable J6 I10-descendant state and checks native camera parity and
all 120 model-conditioned 3D trajectories. These are not measured 3D GT.
`i10_tracking_dense/run.py` isolates one denser proposal budget with unchanged
reliability/admission gates. T0 failed coverage; T1 passed with 123 tracks / 116
IDs. No physical object number or identity is inferred.

`i10_track_train/core.py` deduplicates persistent (frame,Gaussian ID) observations
and adds default-off robust 3D-to-image reprojection; `runtime.py` retains the
original UnionScene/deformation/native renderer and complete two Adam/RNG/
camera/physics state. Static background and physics are fully frozen; original
observed full RGB is retained. No old I10 target substitution, color-selected
static reference, foreground/union coverage term or new topology. Historical
color-conditioned parent lineage is disclosed. `RubberDuck_I10_tracking/
training/launch.py` tests native flags-off/gradient and independent exact-state/
fixed-native-gradient Adam recovery before matched 2000-step arms. Independent
native RGB/alpha/D, PLY and H264 evaluation is in task-local evaluate/audit files.
Tests: test_i10_tracking.py and test_i10_track_train.py. Original PIDG and all
frozen experiments remain untouched.

### Audited I10 binding and persistent local-neighborhood evidence (T3/T4)

- `variants/staged_reconstruction/i10_track_neighborhood/core.py`: reciprocal
  split/equivalent-surface checks, native patch visibility and short-trajectory
  local neighbors; model-conditioned, no physical instances or GT.
- `audit.py`: opt-in123-binding native visibility/no-mutation audit, independent replay.
- `visible_core.py` and `visible.py`: split-independent first20 native-ID nomination,
  actual one-hot compositing, persistent visible-first binding over222 original tracks.
- `experiments/staged_reconstruction/RubberDuck_I10_track_neighborhood/`: fixed
  protocols,337/343-file freezes, independent accounting, real source-RGB diagrams.
  T3/T4 fail temporal coverage; zero optimizer updates, no trained neighborhood prior.
- Tests: `tests/test_i10_track_neighborhood.py` (11), `test_i10_track_visible.py` (7).
  Original PIDG and all frozen historical modules/weights remain unchanged.

### L1 supported-window I10 local motion training

- `variants/staged_reconstruction/i10_local_motion/core.py::support`: current-frame
  visible persistent observations and undirected pairs within audited12-frame windows;
  no graph extrapolation or new labels. `add_local`: robust relative-correction prior,
  default identity; parent-derived local offsets are not ground-truth geometry.
- `runtime.py::{load,objective,snapshot,install}`: isolated native PIDG adapter,
  exact J6 full-state start; static and physics frozen, dynamic Adam retained.
- `experiments/staged_reconstruction/RubberDuck_I10_local_motion/launch.py`:
  default-off preparation/native gradients, matched smoke/recovery/full training.
  Global T3/T4 coverage rejection remains; this is a separately scoped local trial.
- `tests/test_i10_local_motion.py`: nine locality/occlusion/shape/gradient tests,
  plus five retained direct-track tests. No constitutive/material/force additions.

### L2 revocable RGB surface-patch bindings and connections

- `variants/staged_reconstruction/i10_revocable_surface/core.py`: fit-only patch
  memberships/weights, independently bound patch connections, detached evidence,
  suspend/revoke/recover transitions, rotation-invariant length reference loss.
- `runtime.py::{measure,refresh,objective,snapshot,restore,train}`: native full-scene
  patch contributions and raw RGB NCC; scheduled checks; exact controller plus
  model/twoAdam/RNG/camera/physics state. Static/physics frozen; default off.
- `RubberDuck_I10_revocable_surface/{launch,launch_trace}.py`: isolated matched
  qualification/training; separately frozen I/O adapter saves model-only refresh
  evidence for replay without altering frozen training decisions or losses.
- `tests/test_i10_revocable_surface.py`: selection, unknown/revocation/recovery,
  held exclusion, overlap/window checks, rotations, gradients and disabled path.
- `evaluate.py`, `audit.py`, `controller_audit.py`: independent native RGB/artifact,
  complete refresh replay and explicit CPU decision accounting. No geometry GT,
  physical identities, new color static references or new physics are introduced.
