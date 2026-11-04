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


### S1-N：完整Gaussian合成验证未通过（2026-11-03）

完成漫反射/反光×全RGB对照/每图干扰层四臂，各6000更新；72测试、原生
flags-off/梯度、双进程恢复、完整Adam/RNG/同6000相机序列及冻结审计通过。
未见相机独立静态GT PSNR对照→干扰34.4082→33.6690、33.8767→33.2329dB，
RGB外溢增加6.05%/6.44%。可见点表面覆盖略升，薄板约99%，但0.05容差
大于0.04厚度，不能证明正确厚度。条件D/alpha略好不抵消静态图像退化。
实际静态/分解/深度-alpha图仍见方块残留、底座光晕和悬浮片；两视频各完整
解码160帧，四PLY精确对应检查点。合成相机时间网格不是物理运动轨迹。

拒绝本版升级，未启动真实18k→24k，不替换S3-U，不改冻结权重/门槛。
下一步先只读审计已有RGB推断深度/遮挡/时间/薄板证据，避免把错误深度作为
强表面约束；再选择一个独立开关的表面约束候选。暂不能断言数据极限。
不重复C1/S1-N，不扫权重/门槛，不恢复旧暂停任务，不引入颜色/静态标签、
数量、材质、本构、接触或力。见[S1-N实测评估](staged_reconstruction/S1_nuisance/v1/assessment.md)。


### S1-D0/D1：先修复深度证据再决定表面约束（2026-11-03）

独立只读审计固定原RGB推断后再读GT：原深度与同射线静态GT误差<=0.05仅
79.4%，<=0.02约52.6%；薄结构射线有深度56.8%，不能混同约99%空间点覆盖。
检查后深度导出法线平均误差30.8–30.9度；相同像素解析GT深度控制约0.04度，
GT理想量化到原192级网格已产生约16.2度，真实匹配MAE0.038仍高于理想量化
0.00814。暂不将当前深度/有限差分法线直接作为强监督或伪GT。

D0已提交be408aa4。D1固定候选协议在输出前预声明，7a411aa3提交实现及18测试；
默认关闭，保持全部候选/原盆地，连续逆深度半网格细化，4源拟合、另外2源
代价不退化才更新；其接受视角不是独立GT。单一两条件推断/独立评估及
原图复核已完成：深度MAE约0.0380→0.0368，法线30.8→26.7度，薄结构/P90略好；
约20.3%仍未在0.05内匹配静态GT。只保留局部数值改进，不重复D1、不宣称伪影
改善。下一步单候选审计源视角不确定性/深度区间，禁止GT选训练点；新的完整
Gaussian训练须另立协议及静态GT/原生恢复门槛。无全图强深度/法线或物理扩展。
见[深度证据评估](staged_reconstruction/S1_depth_evidence/assessment.md)与
[进度](staged_reconstruction/S1_depth_evidence/progress.md)。


### S1-D2：源视角不确定性有辨别力，但覆盖门槛未过（2026-11-03）

按3b69cf52预声明协议完成两例各96视图，21测试与冻结/父深度/flags-off审计
通过；全部423747/423100候选及深度保留。经验源包络不是概率置信区间。
严格支撑GT区间覆盖95.86%/95.96%，大错误0.69%/0.49%，但候选保留仅
2.03%/2.01%、薄结构支撑保留3.17%/3.15%，未达到预定25%覆盖门槛。
不放宽区间/机位/门槛，不启动当前区间损失Gaussian训练或旧真实续训。
实际c00_t1/c08_t0/c24_t0诊断图已看：可靠点集中在立柱少量面片，底座/
边界支撑仍不足。AUROC约0.76不是静态身份或正确几何证明，亦非伪影改善。

当前已测RGB-MVS证据链存在可靠性/覆盖瓶颈，不证明所有RGB方法都不可解。
用户已明确选择固定Depth Anything V2 Small预训练RGB几何先验；据此继续
新代码/权重冻结和两例独立GT审计，无静态标签/物体数，不再次等待确认。
先验仍非GT，不升级S3-U；新Gaussian训练须另立固定协议和原生完整恢复/
静态GT完整性门槛。见[D2评估](staged_reconstruction/S1_depth_uncertainty/v1/assessment.md)
及[已选下一方向](staged_reconstruction/S1_depth_uncertainty/next_choice.md)。


### S1-D3：固定Small已测，局部锚点无法标定全图（2026-11-03）

用户选定Depth Anything V2 Small后，按e47030ac预声明协议实际完成两例各96图。
官方代码/权重SHA冻结，27测试、192图输入/锚点/标定重放及独立进程模型重载
逐值通过。仅1/96图/例通过空间检查锚点标定；其可见静态深度MAE0.116/0.112，
且该图100%真空背景被赋有限深度。两例直接静态监督门槛均失败，无Gaussian/
真实训练、新86/188/PLY/视频，不放宽旧门槛或更换候选。实际固定0/24/72与
唯一成功c26_t2的RGB/原始先验/误差图均已查看，不能归因反光或宣称数据无解。
原始深度顺序与GT相关均值约0.896，但不是几何正确/静态身份的证明。
下一步固定顺序可靠性及评估专用理想尺度诊断，区分标定瓶颈与单目形状偏差；
GT仅诊断，不回灌系数/掩膜/训练点。不能以排序均值绕过Gaussian完整性门槛。
见[Small实测](staged_reconstruction/S1_pretrained_geometry/v1/assessment.md)。


### D3-O：单目形状/薄结构顺序也有偏差，转向RGB验证的候选（2026-11-03）

固定Small的192个输出已做预声明的独立形状/顺序诊断（3fac945b/5c311055，20测试）。
仅评估可用的GT仿射对齐后，留出可见静态MAE仍约0.0722；相同D1像素约0.0677，
D1约0.0357。特权对齐不是训练或部署结果，系数/深度/掩膜未导出。强顺序的
静态错误约2.2%、薄结构约8.7%；强候选仅7.42%两端均可见静态，91.24%涉及
真空背景（运动/背景类别可重叠）。不能把整体排序相关0.896当静态/细结构监督。
两例固定RGB/误差及明确交集补图已实际查看，原图/指标/门槛保留。
继续一个单独预声明的S1_rgb_prior_proposals候选：先验只提出RGB深度备选，原
候选全部保留，由固定fit/held源RGB验证；不直接增加metric/法线/全图排序损失。
不更换模型/扫阈值，不进入旧真实续训；Gaussian阶段仍需独立完整性门槛。
见[诊断评估](staged_reconstruction/S1_pretrained_geometry/oracle_diagnostic/assessment.md)。


### S1-D4：备选充分但RGB代价几乎全拒，未见收益（2026-11-03）

按59fb74b3单候选协议实现/冻结aabe584b并完成两例各96图。22测试、独立进程
2+2保存恢复、全部192图输入/界/接受方程与实际view82原RGB代价重放均通过。
全部423747/423100候选保留，有限备选38290/37475，但fit不恶化仅3/2、最终
接受1/1。唯一c26_t3(85,108)GT误差0.009416→0.039775，held代价下降亦不能
证明几何正确。全局MAE约0.036805/0.036826不改善，薄结构准确覆盖约41.8%/
41.6%不变；数值及Gaussian证据门槛均失败。固定图及唯一变化裁切实际已看。
不放宽阈值/容差、不更换模型，不做当前Gaussian或旧真实续训，无新86/188。
下一步仅评估专用RGB代价oracle审计：对比连续GT/理想同格GT/D1代价，并按
跨面窗口、源真实可见性分组，区分匹配测量/遮挡/搜索瓶颈。GT不得输出为
训练深度/法线/掩膜/系数/选点；未做该审计前不预设是某一个原因或宣称RGB无解。
见[D4评估](staged_reconstruction/S1_rgb_prior_proposals/v1/assessment.md)。


### S1-D5：真值也难通过原RGB验证，改进支撑匹配（2026-11-03）

按0d983836固定协议完成两例各96图；实现/冻结ccd7ff5b，测量/独立重放3347ad67。
28测试，192图输入/输出/分母及交叉分组核验、两例固定0/24/72原RGB重放通过；
六源/held成本统计逐值一致，独立接受/可见性分组计数一致。全部423747/423100
候选保留，421292/420623有参考可见静态GT，其余2455/2477保留未知。连续GT
通过原接受17268/17324（约4.1%），理想同192格GT为0/0；同面+六源真实可见+
误差>0.05的点仅13/27723、12/27604通过。分数像素解析平面真值4，原最低点
4.04947；整数控制3.99303。边界/量化/中心遮挡均有影响，但不能单独解释。
D1已优化原fit，存在选择偏差；不能由均值或oracle断言单一病因/数据无解。
实际查看六固定彩色图与三控制全部曲线，未输出逐像素GT成本或训练几何资产。
D5仅评估，无Gaussian/真实续训/新86/188/PLY/视频；final_review/status已完成。

下一步一个预声明S1-D6局部平面warp RGB支撑候选：原D1邻域分开拟合/检查
逆深度斜率，可撤销；投影25条支撑射线后采样原RGB再归一化，代替插值已
归一化patch。旧深度/备选用同一新成本比较，原4fit best3+2held接受方程和
1e-6容差保留。没有GT平面/可见性/静态颜色/物体数，Small与S3-U保留；未知
不叫动态/背景。先测试/冻结/独立完整推断恢复，再一次两条件96图推断及
静态GT/薄结构审计。D4数值和Gaussian证据门槛逐项复制，不以小幅改进绕过，
不启动旧真实18k→24k、不恢复暂停分支、不重跑D5或任何已完成负分支。
见[D5评估](staged_reconstruction/S1_photometric_oracle/v1/assessment.md)及
[D6协议](staged_reconstruction/S1_rgb_plane_warp/protocol.json)。


### S1-D6：warp只改善小误差，跨视角提供下一轮平面备选（2026-11-03）

按3c952319协议完成两例96图，实现/冻结5ca41f68，测量与独立审计aff05616。
29测试、409文件训练输入/代码冻结、两条件独立进程完整4对2+2保存恢复通过；
192图原候选/输入/斜率/接受方程及固定0/24/72完整原RGB推断重放逐值一致。
保留423747/423100候选，98662/98504有合格斜率，14731/14655更新。深度MAE
0.03680488→0.03659113、0.03682616→0.03661146（约0.58%），较大错误几乎不动，
薄结构P90略升；数值5%改善/薄结构非恶化及Gaussian严格门槛均未过，不升级。
解析分数位移平面最低点4.04947→3.99303（真值4），但一个斜面像素反而更差。
实际看两例三固定原图/误差/更新图及全部解析对比曲线；没有Gaussian/真实
训练或新86/188/PLY/视频。保存有限局部改进，不冒充伪影减少或结构完整。

事后评估诊断（未回灌或改门槛）：原大错误/无静态85912/85846点，只有974/1004
通过目标邻域检查，约98.8%无法进入提议；准入且原大错误的可见点法线无向角
误差约59度。D1局部自洽不能给这些错误提供可靠斜率，不放宽原0.02检查。
下一步一个S1-D7跨视角局部平面传递候选：固定D6深度作为当前值，从原4个
fit源的原D1合格局部平面产生射线交点与斜率，保持源准入/5x5支撑范围检查，
原始warp成本和4fit best3/2held接受方程/1e-6容差不变；目标无需先有好斜率。
最多4个源平面假设不是物体数，不优化法线/扫参，不换Small、使用GT标签或
补点；父候选和未知身份、S3-U保留。先测试/冻结/完整独立推断恢复，再一次
两条件推断及静态GT/薄结构/覆盖审计，D4两组门槛逐值保留。不得启动当前
Gaussian/旧真实18k→24k，不恢复暂停任务，不重跑D6或旧负分支。
见[D6评估](staged_reconstruction/S1_rgb_plane_warp/v1/assessment.md)和
[D7固定协议](staged_reconstruction/S1_plane_transport/protocol.json)。


### S1-D7：跨视角传递有局部收益，未准入；转查联合撤销路径（2026-11-03）

7680e19a固定协议完整执行，实现/40测试/809文件冻结及三独立进程完整4对2+2
恢复9b2d4ed4；两条件各96图推断、全192父值/源查询/平面/接受/hash、六固定
原RGB重放与独立GT/分母/MAE/P90/原始数据SHA核验eca285b4。全部423747/423100
候选保留，58279/58201更新，其中25119/25226没有自身合格斜率也获外视角更新。
静态MAE0.03659113→0.03590450、0.03661146→0.03591997（约1.88/1.89%）；大错误
净减少603/602，仍占20.13/20.15%。准确静态覆盖60.37/60.27%，薄结构42.66/42.47%，
薄结构MAE/P90均略改善，但原数值5%和Gaussian严格证据门槛均失败，不升级。
约94.9%的父大错误仍无可用源平面；不放宽原邻域、支撑范围或RGB容差，不重跑。
实际查看两例固定0/24/72原图/误差/更新及全部解析控制。final_review/status已
消解pending；无Gaussian/Rubber Duck新训练、原生86/188、PLY或视频结果。

用户提出动态阶段联合清理静态拖影。现S3-U全冻结静态，原S6只放开静态颜色；
动态拟合不能直接撤销静态opacity。并行static_dynamic_coupling审计9a9c8fb6已
检查初始化/梯度/遮挡，去掉静态后动态仍碎裂，不把所有缺失归于当前静态遮挡，
不重复该审计。下一单阶段J0仅检查全RGB无颜色掩膜下开放静态opacity/SH的
原生梯度与状态可恢复性，无优化器更新、不拼接新旧静态库、保留旧权重条件
颜色训练的来源说明。它不是联合训练成功或未知静态身份推断，不能绕过失败
深度/Gaussian门槛。今后学习候选仍须另立匹配协议、完整Adam/RNG/相机恢复及
两条件独立静态GT/结构审计后再考虑Rubber Duck。S3-U/Small/未知身份与暂停
任务保留，不新增本构/材料/力/接触、不扫参。
见[D7评估](staged_reconstruction/S1_plane_transport/v1/assessment.md)、
[后续方向](staged_reconstruction/S1_plane_transport/next_plan.md)和
[J0固定协议](staged_reconstruction/S1_joint_revocation_audit/protocol.json)。


### S1-J0：静态撤销梯度资格通过，尚无更新/清理结果（2026-11-03）

24ccb92d协议，b7e304ba实现与预检，c9ebd762六固定训练帧测量/独立复核完成。
全图RGB原生单次深度排序使静态opacity/SH和动态canonical/grid/deform六组
都有有限非零梯度；两独立进程800/1600 RGB/alpha与参数精确一致，原梯度
atol1e-6/rtol1e-5通过，flags/叶梯度/值/RNG不变。9+2测试、391文件冻结、
54图哈希解码与原checkpoint核验通过。两次失败及修复receipt单独保留；冻结
worker、门槛、权重不改。六图opacity正/负梯度不能当静动身份或删点证据。

实际看86/188早中晚分解及两张1600原图，仍见动态碎裂/拉长、静态边缘模糊。
这是旧S3-U及其自身4126点条件静态库，历史曾用颜色掩膜；本次全RGB诊断
没有重训未知静态，未拼接新S1-C。optimizer更新0，未验证完整Adam训练恢复，
未证明清理/几何正确；final_review/status已完成，无新训练PLY/视频。

下一步另立一个匹配RGB联合Gaussian实验：统一canonical动态池，比较静态
全冻结与开放opacity/SH，保持静态几何不动。先固定两条件合成协议及输入
来源，完成原生flags-off/梯度、独立完整Adam/RNG/相机恢复及独立静态GT/
结构/外溢门槛，才考虑Rubber Duck。失败D7深度/法线不进监督，原门槛不放宽，
不恢复暂停任务，不改并行完整性训练。见S1_joint_revocation_audit/v1/assessment.md。


### S1-J1：联合RGB训练与经用户选定的完整恢复验收（2026-11-03）

J0仅证明梯度可达，见c9ebd762/4fbe1832；无更新，不是拖影消除。J1固定
全RGB统一canonical动态池，与静态全冻结/开放opacity+SH两臂；静态几何不动，
无新颜色掩膜、静态标签、物体数或GT监督。父模型是两例RGB派生S1-N control6k，
继承条件几何支持来源，不能称独立几何GT。每条件每臂计划6000新更新。

原f035486d协议/51b7c6f1实现通过6测试、275文件冻结、原生flags-off与梯度，
但连续40对20+20逐参数轨迹一致失败；原initial/RNG/相机状态精确。独立
即时加载及原生反传诊断发现断点前就有微小非确定性，不能混同于状态丢失。
ac16bc95/31f37abf固定共同epsilon1e-8对照也未过；不扫epsilon、保留失败。

用户明确选择“采用精确状态恢复验收（推荐）”，12aae457另行固定接受协议：
原模型、两个Adam及参数绑定、全部RNG/相机精确加载；独立原生RGB/alpha精确、
raw梯度按原1e-6/1e-5，使用同一份native梯度后完整Adam/model/constraint更新
精确一致。四个条件×臂全部通过，已与原40步失败分别存档，不能声称后者通过。
恢复原J1的epsilon1e-15及所有训练数值，未升级失败epsilon对照。286文件新冻结，
2个launcher测试与原6测试通过。恢复验收修订不改变任何重建质量/完整性门槛。

当前已进入串行完整Gaussian合成训练：两条件diffuse/glossy × frozen/revisable。
独立评估静态GT PSNR、可见/薄结构覆盖、RGB外溢、D/alpha全分母及动态泛化，
并导出固定训练0/24/72、新视角24/各时间、匹配PLY和全解码视频，实际查看后
再判断；此处无清理改善结论。正式状态见runs/.../S1_joint_rgb/accepted_recovery，
报告见S1_joint_rgb/accepted_recovery/v1。Rubber Duck仍是目标，合成通过后还须
另立真实匹配协议；尚无新86/188结果、不启动旧18k→24k，不恢复PAUSED任务。
原PIDG/历史权重、S3-U与未知身份及并行完整性工作保持，不新增本构/材料/力。


### S1-J1-A完成：RGB清理改善，但准确深度覆盖退化，不升级（2026-11-03）

四臂各6000完整Gaussian训练、两Adam/RNG/6000相机采样、独立GT评估和原生
导出661d3158完成。原控制静态精确，候选仅opacity/SH变更，22个动态Adam
状态到6000；8PLY与检查点匹配、两H264各解码160帧。实际看两例静态对比、
训练0/24/72、novel24五时间、原生t2、D/alpha/alpha及解码图，pending已消解。

静态GT PSNR34.408→36.819、33.877→36.065dB，RGB外溢减少70.28/68.87%；
但以47782个GT静态像素为分母的准确射线覆盖88.996→84.618%、87.376→82.378%，
损失4.378/4.998百分点，原<=1门槛失败。薄结构中心覆盖大体保留不抵消它。
alpha缺失仅19→112、33→114，不能把全部覆盖回退称为大面积破洞。RGB悬浮块
变暗，而alpha/深度仍有悬浮支撑和底座外晕；动态池含静态柱/底座，部分动态
球仍是亮斑。不能从外观改善认定几何已清理，不准入Rubber Duck或旧真实续训。

下一步固定一个无训练分解审计：完整lost/gained射线闭合分母，区分alpha缺失
与可见深度偏移；原/candidate opacity×原/candidate SH固定2×2反事实，仅作
冻结权重诊断，不冒充四种新训练或完整训练因果。GT仅在独立评估器中，不输出
训练选点/掩膜/监督，不放宽门槛或扫损失权重。依据证据再定一个结构保护候选。
见S1_joint_rgb/accepted_recovery/v1/assessment.md；完整失败和用户恢复授权保留。


### S1-J2完成：联合清理主要损失是有效深度漂移，下一版需保护结构

18b323e0预声明、0a653f5a实现、882dac4f完整测量/独立复核。5测试、623文件
冻结，原32相机×两条件×opacity/SH四组合全量原生重放通过，c24两独立进程
四组RGB/alpha/D逐值一致，模型/几何无更新。实际看两例c0/c8/c24共12图。

novel8每例47782表面像素，lost3039/3458中2958/3386（97.33/97.92%）仍有效
但误差>0.05，只有81/72变缺失；gained947/1070，净−2092/−2388闭合原J1-A。
lost仍有效子集平均深度位移+0.01810/+0.02149；薄结构8417分母准确像素净增
123/171。不能把4–5百分点覆盖下降都叫破洞，也不能只用薄板局部增益升级。

SH单独改变使外溢下降38.67/36.83%，全部32相机alpha/D精确不变；固定参数
反事实的外溢对称边际份额opacity69.79/71.30%、SH30.21/28.70%，含交互，
不是训练因果或物理身份。RGB更干净而alpha仍有悬浮块/底座外晕，主体表面
深度漂移占主要损失。J2是0更新诊断；J1-A继续completed_not_promoted。

下一步先预声明一个保护opacity调整时有效深度的联合候选：可研究RGB父模型
原生渲染深度的自身参照，必须披露它不是GT、可能保护错误支撑，不用本次GT
图选训练像素，不使用失败D0–D7深度，不把SH变暗等同几何清理。先文献/原生
梯度可实现性与固定数值协议，再匹配两条件完整Gaussian及精确状态恢复验收；
原独立GT/结构门槛通过才另立Rubber Duck协议。不恢复旧暂停任务、不扫参。
报告：experiments/staged_reconstruction/S1_joint_attribution/v1/assessment.md。


### S1-J3进入完整Gaussian训练：固定父渲染深度自身参照（2026-11-04）

825d23a1预声明、c98f6381实施、2e3afbda资格完成：仅增加固定0.1的支持内
log1p(|D/alpha-parent_D/parent_alpha|)自身参照，teacher来自相同RGB父静态，
不是GT且可能保留错误支撑。仅该项连接static opacity梯度；联合全图RGB
照常更新静态SH和统一动态池，静态几何冻结。没有alpha复制/alpha-one/
新深度法线/DA-V2/GT训练输入、语义掩膜、物体数、删点、扫权重或新物理。

6新+6原测试、295训练输入/源码冻结通过。原生初始RGB/alpha/D精确，flags-off
返回原class/loss，初始RGB梯度原容差；隔离Lself只到opacity，固定h0.005差分
0.0002142100对autograd0.0002141777通过。两例各20步smoke及独立双进程
完整model/两个Adam/绑定/RNG/相机恢复与固定native梯度Adam更新精确通过。
用户授权验收沿用，旧40步CUDA轨迹和epsilon对照失败保留，没有重试或改判。

当前只训练diffuse/glossy两个新候选各6000步；匹配控制逐字节复用J1-A已完成
同起点/采样/完整Adam的frozen_static6000，不重复旧训练。原trainer/runtime/
renderer代码不变，进程内默认关适配器提供新增项。保留原epsilon1e-15/学习率/
初始化/采样和全部质量门槛，不凭smoke/自身参照当独立几何改善。

结束后独立评估全分母准确射线、GT静态RGB、表面/薄结构/动态泛化和外溢，
核验匹配6000采样/Adam/原生0/24/72及novel24所有时间、D/alpha/alpha、PLY
和完整解码视频并实际看图。当前没有新的改善结论或Rubber Duck86/188结果。
两条件全部原门槛通过人工复核后才另立真实协议。状态runs/.../S1_joint_depth_trust/v1，
报告experiments/staged_reconstruction/S1_joint_depth_trust；不重跑J1-A/J2或旧任务。

### J3完成与Rubber Duck联合试验J4（2026-11-04）

J3已5a6a61a8完成两新候选各6000步，匹配历史控制未重复。原数值门槛全部通过：
静态RGB外溢下降41.13/40.10%，准确射线覆盖净+49/+27；但原生alpha/D人工
复核仍有悬浮与底座晕。独立事后支撑审计outside-GT高alpha仅减2.39/2.59%，
不能将变暗当几何清理，最终completed_not_promoted/pending:false。8PLY与
两H264各160完整解码已核验；这是合成Gaussian，不是Rubber Duck结果。

用户明确要求回Rubber Duck检测并联合动态物体提升。另立J4有界真实探索，
experiments/staged_reconstruction/RubberDuck_joint_pilot/plan.md及protocol.json。
同一S3-U final6k完整父与其自身静态库：冻结静态对照/开放opacity-SH并加
固定0.1父D/alpha自身参照候选，各2000步；不嫁接S1-C、不改J3失败结论。
完整继承动态Adam/RNG/相机与S1静态Adam（静态权重已逐值相同），静态几何
冻结。physics参数/Adam/图保留且冻结，纯RGB重建试验不能称原物理续训。
父曾用颜色条件掩膜须披露，新训练无新语义掩膜/GT/数量。先独立完整恢复
资格再训练，原生86/188早中晚及全序列RGB/alpha/D和全部val比较，缺少独立
固定视角GT须明说，几何/完整性人工审计决定结果，不按PSNR自动升级。

### J4 Rubber Duck联合对照完成（2026-11-04）

9f5d54dd预声明，3f1d8974实施，88ccffb9资格通过，两臂各2000新更新完整完成。
原生1600的30张val平均PSNR父28.1663→冻结静态30.1266→联合31.3365dB，
联合较匹配续训多1.2099dB；SSIM0.974798→0.978989，属于RGB改善。
沿用S3-U自身4126静态/15507统一动态池，无数量/身份/新语义掩膜。历史父曾用
颜色条件掩膜必须保留披露；不是从零未知静态重建，也不是物理采集数据声明。
完整继承静态12k Adam（候选opacity/SH至14k）及动态22k→24k Adam、RNG/
同2000相机采样。静态几何、velocity/stress/其600步Adam及图缓存精确不变；
本轮纯RGB重建不能称原physics续训。10测试/198输入源码冻结/原生flags-off
梯度及两独立进程完整双Adam/RNG/相机/图与固定native梯度Adam更新精确通过。

实际复核86/188早中晚分解与原生1600RGB、alpha/D和六训练轨迹观测：静态边缘
和轨迹图像更清楚，但离开训练相机-时间组合的鸭子仍碎裂，动态暗RGB区域
仍有大块alpha/D拖尾，未通过干净支撑/完整性人工门槛。c86/c188父静态高alpha
支持135441/148933→134111/147546，父支持内D/alpha变化0.002774/0.002762，
只是自身代理，不是独立GT几何准确率。保留有限RGB收益，completed_not_promoted。
原生同相机全200时间视频/PLY与检查点核验见RubberDuck_joint_pilot/v1报告；
不重跑J4、不自动延长或升权，不改J3失败结论。Rubber Duck仍是下一步目标：
先审计实际相机-时间覆盖和动态暗色高alpha支撑的RGB可见性，复用已有耦合/
完整性研究，不能从暗色/残差直接删点或伪造同时间多视角GT。下一候选独立固定
协议后再实施，不回到无止境辅助深度循环。

### J5 Rubber Duck可见性审计完成（2026-11-04）

c434fc1c预声明、72e9dd22实现/资格通过。三个既有检查点各120源原生1600只读
审计完成，0训练更新；完整模型/双Adam/RNG/相机/physics图缓存前后逐值不变。
实际train120张=120精确时间=120相机变换，同时间多相机时间数0，各相机仅1时刻。
固定86/188早中晚18目标和源0/24/72原RGB/原生贡献、全120聚合分母/图像SHA
独立重放通过。6测试，原默认关/打包与单独渲染/原生RGB-alpha-D一致，两进程
资格逐值一致。单高斯确认导出weight是post-T的alpha*(1-alpha)，不用它量贡献。
新夹具主点错误保留日志和冻结代码，独立文件修复，未改协议/门槛或PIDG。

J4联合六组在目标暗高alpha区域的first20 ID并集解释99.83%—99.99%原生alpha，
但同组沿训练轨迹可见贡献约75.88%—88.48%落在暗观测，约11.52%—24.12%仍落
在较亮观测。组级非零不等于每点可见，暗观测不等于背景/可删，亮观测不保证
正确表面/物理身份。86/200暗高alpha像素父17,788、冻结39,830、联合37,689，
全像素分母1,440,000；各模型集合不同，不能叫GT伪影面积。实际看6张三模型
固定对照、9张训练源贡献图、2张原生1600 RGB/alpha，宽大暗支撑/碎裂仍在。
J5 completed_read_only_not_promoted/pending:false，没有新训练/视频/PLY或独立GT。
报告：RubberDuck_visibility_audit/v1/assessment.md、independent_audit.json和manual_review.json。

下一机制采用时间变可靠锚点约束弱可见动态运动，复用uncertainty_motion的2195
RGB支持锚点证据；它不是深度GT。并行“动态物体保持完整度”已在reliable_motion
实际实现该机制，本线不重复/不停止它，不自动拼接未通过候选或重启J4。
先跟随其完整资格、匹配训练和原6×9/6×200完整度结果，通过后另立静态
opacity/SH可撤销联合的完整状态匹配协议；原结构/外观门槛保持，不按暗色删点。
父历史颜色条件掩膜/统一动态池未知数量继续披露，无新物体数/静态标签/GT输入。
这不证明RGB无解，整体清理与完整度目标仍未完成。
