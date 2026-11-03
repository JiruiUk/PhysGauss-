# Rubber Duck reconstruction plan — revised 2026-11-03

This is a reconstruction/observability branch. The PIDG baseline and the existing
M0–M10 physics milestone order remain unchanged. Every new branch defaults off.

The original S1–S6 training stages remain the execution plan; their explicit
status is in `staged_reconstruction_plan.md`. R0–R4 below are diagnostic work
items, not replacement stages. Completing R0 did not complete S1 or the overall
task. Negative results require corrective work, without cancelling S2–S6.

## Evidence and decisions

1. Transfer-integrity v1 (matched 2k continuation) improved local objectives but
   did not repair fixed-view fragmentation.
2. Scratch-integrity v2 (fresh common 15k, matched 35k continuations to 50k)
   reduced short-track EPE 1.8039→0.2800 and static anchor distance
   0.47937→0.16306. Test PSNR was 40.8475→40.8205 dB. Fixed-view fragmentation
   and negligible late early-bank support remained. More optimization of the
   same local constraints is not the next experiment.

## Execution ladder

| Step | Experiment | Admission / decision |
|---|---|---|
| R0 | Verified static subset with a time-independent rendering map | Completed; image guardrails passed, structure did not improve; H1 not promoted |
| R0a | Static selection/reference geometry audit | Completed: reference inconsistency already existed at15k; motivated the conditional S1 reconstruction, without certifying contact geometry |
| R1 | Persistent known object support / object-centric canonical representation | First audit coverage and independent identity evidence; unknown stays unknown |
| R2 | Reliable tracks, separated from static geometry | Retain confidence gates; do not infer identity through ambiguous occlusion |
| R3 | Kinematic compatibility, then CMR | Rendered trajectory, velocity and coordinate derivatives must be defined consistently |
| R4 | Known-surface nonpenetration, then fixed low-dimensional contact | Requires accepted object identities and surface geometry; no per-Gaussian free forces |

Rubber Duck remains the main development dataset. Small manufactured tests
supplement F/material/contact verification; they do not replace its benchmark.

## R0 scope

Two matched arms start from the completed scratch experiment's immutable common
15k checkpoint, retain the full optimizer/LR/RNG/camera state, and train to 50k.
No old final50k trained state is used for initialization. Only the R0 main flag
differs. No v1/v2 soft transfer/track/shape losses are enabled.

The accepted 14,624 static rows are a *partial* image-supported hypothesis,
covering about 60.68% of blue-core contributions on average, 47.86% in the middle
third. Other rows remain on the original path; no labels propagate by proximity.
Each accepted row receives a fixed deformation transform and covariance offsets
captured at its pre-existing training-only best-support reference frame. Its
underlying Gaussian parameters and SH appearance remain trainable. This makes
the selected geometry independent of query time without adding particles or
learnable parameters. It does not prove that the captured geometry is correct.
Known-static rows are excluded from the dynamic CMR sampling domain. This
exclusion is part of the declared treatment, not a separate claimed benefit.

## Evaluation and adaptation rules

- Fixed final50k states; validation30 and test50 remain outside gradients.
- Compare R0 arms directly; historical C1/S0/S1 are contextual references only.
- Predeclare native1600×900 rendering, fixed cameras2/86/188 × all200 times,
  moving-camera control, per-region image metrics, and static-subset invariance.
- Check independent held-observation static-track reprojection, where available.
  Structural invariance and low training loss are not independent geometry GT.
- View-completeness measures are descriptive proxies, never true completeness.
- Image guardrail: validation PSNR loss ≤0.2dB and standard VGG LPIPS relative
  worsening ≤5% against the matched control. A visual/geometry benefit must
  additionally be present; meeting guardrails alone is not success.
- Do not promote a model based only on all-scene alpha: inspect static and
  dynamic regions and fixed-view native crops separately.
- After evaluation, append evidence and revise the next step. If static-only
  structure improves but dynamic fragmentation persists, move to R1. If the
  static subset remains poorly supported or image quality degrades materially,
  audit selection/reference geometry before another long run. Do not increase
  weights or launch parameter sweeps automatically.
- Each implementation/result milestone is committed separately. Historical
  source, datasets, checkpoints and reports remain protected.

## Current status

R0: implementation and preflight passed. All 32 related tests, native gradient
and Adam checks, flags-off numerical parity, and full-state recovery passed.
Implementation milestone: `23f53c35`; launch record: `4738e2b9`. Both matched
arms finished50k with35000 actual updates each. Final metrics, native image
review,16 fully decoded videos and independent evidence audit are complete.
H1 is not promoted; the next step is R0a, not a full R1 training run.
Immutable settings remain in `static_pool/plan.json`.

## R0 completed measurements

Final50k measurements and reviewed images: `static_pool/assessment.md`.
Both image guardrails passed: validation PSNR40.455640→40.437735dB and standard
VGG LPIPS0.01360374→0.01418889 (+4.30%). Test PSNR40.846448→40.797551dB.
Fixed86/188 early/middle/late native crops retained the fragmentation pattern,
while the observed camera trajectory remained visually good.

Selected-static validation blue contribution fell51.60%→12.12%; selected median
opacity fell0.999996→0.003908 (68.18% below0.01 in H1). Training-image track
reprojection worsened1.84→7.56px at400px width. This supports opacity suppression
of the fixed subset, without identifying its root cause. Static invariance is
structural; it does not establish geometry recovery. H1 is not promoted.

## R0a admission and audit scope

1. Use training-only observations to map suppressed rows to their reference
   frames, initial and final coordinates/scales/opacity, and visible support.
2. Audit whether independent per-row reference warps form consistent static
   surfaces. Compare initial/final reprojection and cross-time support, including
   occlusion; retain unknown labels where evidence is ambiguous.
3. Check feasibility of one consistent reference geometry or calibrated static
   fit before proposing a short controlled experiment. Do not fix the result by
   threshold changes chosen on validation/test images.
4. Only after this audit, write a separate short-run protocol and admission gate.
   No new full candidate, automatic weight sweep, R1 model, constitutive/contact
   or unknown-force module has been launched from the R0 result.

Rubber Duck remains the main benchmark. Missing simultaneous multi-view/physical
GT limits conclusions; the manufactured F/material tests remain supplementary
and the original M0–M10 order remains unchanged.

## R0a initial reference audit completed

`static_reference_audit/results.json` independently recomputes the same fixed
mapping in the immutable initial15k and final H1 states. The37 conditional
tracks/567 observations already had mean7.79751px reprojection error before R0
training, versus7.56417px afterward (400px width). Initial selected opacity
below0.01 was0%; final68.18%. The14624 rows use120 separate reference frames.
The initial static reference was not established as consistent geometry; this
is not solely a late optimization drift result. The audit is read-only and
does not count as S1 training. Continue S1 reference/mask preparation and then
independent static optimization; S2–S6 remain pending in the restored stage plan.

## S1--S6 actual continuation (supersedes the earlier pending status)

S1 completed12000 updates and conditional static admission. Three sequential
S2 candidates completed16000 each; the first two failed and remain preserved.
The third passes unchanged image gates (dynamic24.25343dB/alpha0.982654,
static35.53011dB) and permits conditional sequential reconstruction corrections.
Its fixed-view fragments and imperfect frame42 ownership remain unresolved; no
physical identity or complete geometry success has been established. See
staged_reconstruction/S2_geometry_recovery/assessment.md.

S3 real-parent recovery exposed a rank-deficient canonical-neighbor case. A new
independent state_recovery entry uses actual current-world same-bank neighbors,
retains the original validity thresholds, and saves derived state. Corrected
40→80 recovery passes136 Adam states and exact camera/Python-RNG replay. Its6000
formal updates, native evaluation and actual fixed-view review are complete;
validation dynamic25.392299dB/static35.941669dB. All57 related tests pass.

S4 binds91 train-only fragments/455observations to79 fixed rows and completes
4000 updates with full evaluation/review. Exact forward one-hot contributions
pass the original visibility-partition gate after preserving an earlier failed
backward-approximation attempt. Validation dynamic25.340218dB/static36.199356dB,
but track projection worsens0.536841→0.785017px@400: retain this negative result.

S5 completes4000 updates with validation dynamic25.603743dB/static35.983147dB
and actual fixed-view review. Track error0.664671px@400 remains worse than S3.
All800 conditional-interior calls have zero active violations; there was no
actual collision correction. Its limited conditional continuation admission
is not admission of the original-plan complete contact scope. S6 real-parent
40→80 recovery passes138 Adam states; its predeclared4000 updates, independent
native val/test, full sampler/state audits and actual image review are complete.
Validation dynamic28.327031dB/static36.169914dB, test dynamic28.563049dB;
455 qualified track observations average0.427107px@400 (20.44% below S3).
All800 conditional-interior calls are inactive. Static geometry is exactly S1;
only its appearance was unfrozen. Three final videos decode200frames each.
Observed-view contours improve, but novel-view sheets/holes and long late bank
streaks remain. Physical IDs after ambiguity remain unknown.

S5's manufactured300-step optimizer validation is separate from real-scene
training. Rubber Duck has no certified contact surface or acquisition seconds;
contact force/friction/restitution remain off. S6 jointly refines the admitted
image/kinematic/track/conditional-interior objectives and static appearance at
0.25 learning rates; static geometry remains S1. Full operational continuation
is in staged_reconstruction/runbook.md. These are conditional reconstruction
experiments, not material/contact identifiability claims.

## Measured completion and unresolved original scope

The revised conditional S1--S6 training lineage completed46000 formal updates;
two failed S2 candidates add32000 preserved updates. Standard native1600x900
quality is measured for every admitted S2--S6 final state. Final S6 validation
full PSNR30.58520dB, SSIM0.975011, VGG-LPIPS0.041803; test LPIPS0.041247.
See staged_reconstruction/final_summary.md and output/staged_reconstruction/index.html.
These are sequential image/qualified-track results, not causal ablations or
matched legacy-camera PIDG comparisons.

No further training is automatically queued. The original complete geometry,
persistent physical identity and contact goals remain unresolved; S5 implemented
only a conditional static-interior prior that never activated in the real scene.
Object-object nonpenetration, contact forces, friction and restitution remain
unvalidated/off, rather than silently removed from the original objective.

Next evidence priority is geometry/identity observability, supported by a small
calibrated RGB synthetic supplement with metric geometry, acquisition seconds,
material-point trajectories and persistent IDs. Keep asynchronous single-view
RGB as training input and synchronized novel views/3D truth as evaluation only.
First isolate known motion without collision; then admit genuine collision
tests only with certified surfaces/IDs/time and independently exported contact
truth. Manufactured sphere/plane optimization is not a replacement for this
RGB reconstruction benchmark. Freeze split/metrics/criteria before running;
do not increase weights, sweep candidates or add constitutive/unknown forces.
The concrete retained dependencies are in staged_reconstruction_plan.md.


## S3-U: user-authorized unified dynamic population (2026-11-03)

The user selected one unified dynamic Gaussian population without estimating
body number. This is a separate branch, not unknown-count instance discovery.
No instance labels, original four-group initializations, identity ledgers or
trained S2/S3 dynamic weights are inputs. Frozen S1 static support is shared.
One shared canonical population, deformation network and velocity/stress field
replace separate instance fields. The global15,507 point budget is unchanged;
capacity and initialization differences are disclosed rather than treated as
a matched single-switch ablation.

Frozen protocol: staged_reconstruction/S3_union/protocol.json. Execute16k
image-only warmup, actual native review and unchanged numeric gates, then6k
S3 kinematic/CMR continuation. Existing weights, time units, camera convention
and static freeze remain. Validate each phase's40-to80 full-state recovery
before formal training. Dynamic-only spatial stencils have no physical-boundary
exclusion. No contact/constitutive/unknown-force/instance-count inference is on.
Compare native val/test and fixed86/fixed188/original-camera videos against
original S3, preserve negative results and late identity uncertainty.

Implementation:49 related tests, native renderer parity(max3.5763e-7) and
warmup40-to80 Adam/RNG/sampler/static-freeze verification passed. Formal warmup
is ready; no historical paused trainer or automation is resumed. Test setup
failures involving the shared temp directory and a missing new parent directory
are retained in the test receipt; the isolated fresh test directory passes.


S3-U warmup milestone:16k actual updates and native validation completed.
Dynamic23.990448dB/alpha0.986168/static34.829215dB pass the unchanged gates;
static drop from S1 is0.814601dB. All16k sampler choices/final RNG and22 Adam
counters verify; S1 tensors remain exact. Actual observed/fixed-view review
retains fragmented sheets/streaks and blurred foreground, so admission is only
for the declared6k S3 representation comparison, not geometry/contact success.
The real-parent physics40-to80 recovery passes34 Adam states and all RNG/static
checks. Original sources/weights and loss thresholds are unchanged. See
S3_union/warmup_assessment.md and receipts. Independent preview failure and
source are retained; its CPU/CUDA comparison repair changed no training state.

### Completed S3-U evidence and next dependency (2026-11-03)

User-selected unified dynamic Gaussian pool, without body-count or identity
inference, completed16k warmup+6k S3 continuation.49 related tests, independent
40-to80 phase recovery and complete sampler/RNG/Adam/source/static-freeze audits
pass. Native final validation dynamic25.181221dB versus original S3's25.392299dB
(-0.211079); static34.999955dB, test dynamic25.393077dB. Standard validation
PSNR28.165765/SSIM0.969167/LPIPS0.045682. The4k dynamic26.156537dB is retained but
does not replace the preselected6k endpoint. Shared capacity/initialization/
coverage differ; do not interpret this as a one-switch ablation.

Actual13-image review retains fixed86 late sheets/streaks and fixed188 middle
fragmentation/stretching. Good observed-image fit/alpha and valid derivative
stencils do not certify geometry or physical boundaries. Six new H264 comparisons
and single-model videos decoded200frames each; historical checkpoints/videos
remain unchanged. See staged_reconstruction/S3_union/assessment.md and
completion_integrity.json. The main gallery links this count-free branch.

The authorized S3-U comparison is complete, with no automatic S4-U--S6-U or
physics expansion. Next evidence priority remains calibrated synthetic RGB and
independent novel-view geometry/trajectory evaluation, retaining unknown physical
identities. It must not reintroduce physical-body-count estimation into this
unified-pool branch. Original incomplete contact goals stay recorded; no weight
sweep or paused historical trainer/automation is resumed.

### Unknown static support requirement: S1-A (2026-11-03)

The user requires static discovery without preassigned blue/static labels.
Prior S1/S3-U remain historical prior-conditioned results, not evidence of this
ability. New opt-in S1-A uses training RGB, times and supplied calibrated cameras;
alpha only composites RGB and never gives motion labels. Generic tracks must
explain held observations by one fixed world point; low-parallax/short/ambiguous
tracks remain unknown. No original masks, hull, weights or selected tracks enter.

Initial discovery:97 fixed-point candidates/200 incompatible/515 unknown;
conservative initialization543 Gaussians. Actual inferred-mask inspection reveals
late support-face coverage with early sampled views unknown. Train the predeclared
12k conditional static pilot only after tests, native parity and40-to80 recovery.
Report coverage and limitations; complete static recovery is not established.
Do not substitute color priors for missing evidence or attach physics/force/contact.
See [S1-A plan](staged_reconstruction/S1_automatic/plan.md). The next full-scene
automatic version must also remove S3-U's yellow dynamic initialization/coverage;
replacing only static support would not meet the user's full no-label intent.

S1-A12k training/recovery audit completed, but actual images show only partial
late horizontal support and missing tall surfaces/early-view fragments.51/120
images have some inferred support; all-pixel fraction0.6507% is not static recall.
Training selected-region24.069166dB does not establish independent performance.
Do not replace S3-U static support or certify full automatic decomposition.
Next evidence priority is long-baseline correspondence and coverage validation;
no color-prior fallback, threshold sweep or physics integration follows.

### Cross-temporal confidence static pilot: S1-CT (2026-11-03)

The user authorized confidence-guided cross-time fusion without known static
colors. Completed matched fixed/progressive12k arms,53 deduplicated train-RGB
geometric anchors and a common1954-Gaussian initialization.29 tests, native
renderer parity, independent40-to80 confidence/Adam/RNG/sampler recovery and
formal state/source/history audits pass. No old static weights/hull or instance
count enters; this borrows confidence updates, not the paper's historical3DGS
or DUSt3R setup.

Fixed-to-progressive common-candidate training PSNR23.1274->29.0826dB;
held-out-camera inferred-region PSNR16.1345->19.6458dB, while frozen old-seed
region PSNR26.6069->26.0643dB. Positive-support frames41->81/120 and all-pixel
fraction0.6261%->1.0921% are selection diagnostics, not static recall. Total
optimizer steps and camera schedules match; image-supervised steps6000->10946.
Validation regions are inferred from geometry/RGB, not independent static GT.

Actual native86/188 and early/middle/late comparison retains missing tall
surfaces, fragments, blur and spill despite local support gains. Preserve the
mechanism and negative completeness finding; reject S3-U static replacement.
Prioritize candidate-geometry/occlusion visibility, long-baseline correspondence
and independent synthetic static/geometry-GT audits before adding missing
surfaces. No threshold/weight sweep, physics expansion or paused-work resumption.
Evaluation-only original-trajectory RGB repair preserves the first failed
export; latest video decodes200 frames, PLY matches checkpoint vertices.
See [S1-CT assessment](staged_reconstruction/S1_cross_temporal/assessment.md).

### S1-G: dense geometry and evidence-gated birth (completed 2026-11-03)

Completed geometry(G1) and birth(G2),12000 updates each, with common3273-point
initialization,120-camera pool/order and revisable confidence. No known static
colors/masks, object count or physics extension.45 tests, independent40-to80
resume across birth, native parity and full formal Adam/RNG/confidence/topology
and frozen-history audits passed. G2 tried1536 births, revoked1222 low-opacity
new points and ended3587 points; G1 retained3273.

Frozen CT validation inferred-region PSNR: oldCT19.6458, G125.4932, G225.4742dB.
New common inferred validation:14.6414/22.0033/22.0512dB. Frozen old-seed training
region regressed26.0643->25.2010/25.2414dB; frozen CT training29.0826->28.2980/
28.4432dB. Inferred regions are conditional proxies, not independent geometry or
static GT. G2's extra benefit is small and inconsistent; do not promote birth.

Actual native86/188 and early/middle/late panels show much clearer column and
pedestal surfaces, while yellow foreground-like smears, halos/spikes and floating
fragments persist. Initial coarse points are protected from pruning, so this
birth mechanism cannot retract contaminated original candidates. Keep G1 as a
simpler geometric-admission audit reference; do not replace S3-U static support.
Next priority is visibility-aware static-candidate/depth audit with independent
synthetic static/motion/specular GT and decoupled camera/time observations,
before sustained object attribution. Unseen remains unknown. No automatic third
training, threshold/weight search, body-count estimate, physics or paused-work
resumption follows. Three H264 videos each decode200 frames; PLY positions match
checkpoints. Static fixed-view time invariance is structural, not geometry proof.
See [S1-G measured assessment](staged_reconstruction/S1_geometry_growth/revision2/assessment.md).

### S1-C：文献驱动的可见性隔离（2026-11-03，负结果保留）

完成完整G1 12k→18k匹配对照/隔离续训、55测试、双进程短训恢复及正式Adam/RNG/
相机采样/源文件审计；各6000相机样本一致。3273→3031点的隔离未带来可确认清理：
CT验证代理PSNR对照25.7836、隔离25.7377dB，新验证22.0890/22.0206dB。
实际原生86/188及早中晚图仍有黄色拖影、蓝色尖刺/悬浮片。主结构保留不等于完整
正确；不升级隔离组、不替换S3-U。三视频各完整解码200帧，PLY对应检查点。

用户现已授权持续查文献并改进。下一步使用独立RGB合成审计（相机/时间解耦，
薄结构/运动/反光，GT仅评估），再选择一个DeSplat启发的干扰层/静态层分离试验。
未选择区域长期无监督、静态高斯已吸收瞬态外观是待证机制；不反复加步数/删点、
不自动加权/扫参、不启动材料/本构/接触/力扩展，不恢复旧暂停任务。
见[实测评估](staged_reconstruction/S1_clean_visibility/v1/assessment.md)与
[文献和后续顺序](staged_reconstruction/S1_clean_visibility/literature.md)。
