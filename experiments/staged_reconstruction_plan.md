# S1–S6 执行清单（2026-11-03 校正）

依据用户原始方案第 9 节 Stage 1–6。用户已授权按计划执行并根据结果修订；
不需要再次申请常规实现、测试、训练及里程碑提交的许可。

此前 R0 是辅助诊断实验，不是 S1，也不代表六阶段任务完成。原先的 R0–R4
是诊断工作列表，不能替代以下阶段或把后续工作从计划中删除。R0 负结果应触发
S1 修正；S2–S6 保留，但只有其依赖的实现和验收具备后才能训练。
历史 `S1_scratch_transfer` 是旧实验名称，与此处 Stage 1 无关。

本轮条件重建训练链现已完成：12k/16k/6k/4k/4k/4k，共46k次正式更新，
另保留两次失败S2的32k次更新。原生评估、恢复/采样审计、固定视角图像复核
和视频解码均已完成，汇总见 [实测报告](staged_reconstruction/final_summary.md)。
这是修订范围内的实验执行完成；完整几何、跨遮挡物理身份和原始接触目标尚未
完成。下表及末尾依赖保留这些未解决目标；不据此自动开启下一轮训练。

| 阶段 | 原计划内容 | 当前实际状态 | 下一项可执行工作 / 依赖 |
|---|---|---|---|
| S1 | 独立静态 Gaussian 重建、统一世界坐标与参考几何 | 已完成12000步训练、原生评估和图像复核；作为条件静态初始化进入S2 | 测试静态候选PSNR35.79dB；尚不能认证精确接触表面，见S1_assessment.md |
| S2 | 动态物体 canonical 初始化、持久 object ID、逐物体形变 | 第三次16k和原生评估/固定视角复核完成；数值门槛通过，只准许作为后续重建修正的条件初始化 | 动态24.253dB/alpha0.98265，静态35.530dB（下降0.114）；固定视角仍碎裂，帧42有组混合，物理身份与完整几何未验证 |
| S3 | PIDG velocity/stress/CMR，静态几何保持固定 | 6000步、原生评估、完整状态与实际图像复核完成 | 动态25.392dB/alpha0.98550，静态35.942dB；固定视角仍碎裂。当前世界同组邻域有效，时间不是物理秒，不能宣称应力真值恢复 |
| S4 | 可靠 sparse tracks 与轨迹重投影 | 91条/455次合格观测绑定79个固定行；4000步与原生/固定视角复核完成，但轨迹精度未改善 | 动态25.340dB、静态36.199dB；轨迹0.53684→0.78502px@400为负结果。保留结果，不自动增权/扫参；仅继续预定联合微调，不能称轨迹改进成功 |
| S5 | 碰撞与非穿透，再接触力、恢复与摩擦 | 受控算子300步、真实父状态恢复、正式4000步、原生评估和图像复核完成；范围限条件静态体先验 | 动态25.604dB/alpha0.98482，静态35.983dB；800次先验调用全部未激活，未证明碰撞修正。物体间非穿透与接触力/恢复/摩擦因身份、表面与秒时钟不足未验证/关闭 |
| S6 | 静态/动态/物理/对应/接触/外观联合微调 | 4000步、原生评估、138组Adam/采样审计、图像复核与3段200帧视频解码完成 | 动态28.327dB、静态36.170dB，轨迹0.427107px@400，比S3低20.44%；新视角仍碎裂，晚期ID未知。800次条件先验全部未激活。静态几何保持S1，完整接触未验证 |

## S1 已完成的补充审计

见 `static_reference_audit/results.json`，审计脚本未执行任何优化器更新，未修改
R0 训练代码、检查点或原始数据。

- 14,624 个静态候选来自 120 个不同参考帧。
- 同一固定映射的初始 15k 状态，37 条条件静态轨迹、567 次训练图像观测的
  重投影均值为 7.79751 px；H1 终态为 7.56417 px，均在 400 像素宽坐标系。
- 初始候选没有 opacity <0.01 的点，H1 终态该比例为 68.18%。
- 因此初始化本身就存在跨观测误差；长期训练主要降低了大量候选的不透明度。
  这支持先修参考几何，尚不能证明每个候选标签错误或整个静态表示无效。
- 这些轨迹来自模型原始训练图像，属于条件静态证据，不是独立 3D GT。

## S1 实施前准备（本轮已执行，保留方法记录）

实现见 `../variants/staged_reconstruction/`，冻结协议见
`staged_reconstruction/S1_protocol.json`，进度见 `staged_reconstruction/progress.md`。
S1 新分支采用数据元信息相机约定，原 PIDG 加载器保持不变。历史 legacy 相机
检查点不用于初始化这一分支；两种相机约定的数值不作为匹配对照。早期错误地
沿用 canonical 范围的准备结果保留，正式训练使用训练图像重建的统一世界几何。

1. 审计静态像素支持与动态遮挡。已验证静态、已知动态、未知三种区域分开；
   不能仅凭蓝色就把全部区域声明为真实静态标签。未知像素不作静态负监督。
2. 建立同一世界坐标下的一致静态参考几何，避免把不同时间逐行 warp 捕获
   直接当作已验证表面。训练用静态多视角观测；稀疏轨迹只约束有证据的点。
3. 独立静态表示与损失：训练图像只在有支持的静态区域监督；屏蔽动态和
   不确定遮挡。动态表示在该阶段不更新。不得用黑色目标惩罚未知静态表面。
4. 新实现默认关闭；先测试坐标/投影、掩码、梯度归属、恢复和 flags-off 行为，
   再运行短训练。根据短训结果决定完整 S1 预算，并保存单独协议和提交。
5. 验收同时检查跨观测重投影、静态可见贡献、不透明度退化和原生多视角图像。
   零时变误差、低 RGB loss 或静态点被透明化均不构成几何验收。无深度/法向
   真值时明确诊断的证据范围，不把模型预测互相一致当作外部验证。

## 持续执行规则

每阶段分开保存实现、短训练证据、正式结果和计划修订。负结果先修复该阶段，
后续阶段的状态仍是待执行，而不是取消。阶段完成或诊断队列完成不代表用户
整体任务完成；报告必须明确当前真正运行的任务和仍待实现的阶段。

## 原始目标中仍未完成的部分

本轮 S1–S6 是条件重建课程的训练与复核；阶段编号不代表原始几何/身份/接触
验收全部成功。S1 的静态接触表面未认证，S2 跨遮挡物理身份与完整新视角几何
未通过，S4 轨迹精度为负结果，S5 只具备未激活的条件静态体先验。原方案的
物体间非穿透、接触力、摩擦与恢复系数仍为未完成目标，不能在总结中删除。

后续先补足几何与身份的可观测证据，再推进接触；不自动开启另一轮权重搜索。

1. 保留 Rubber Duck 作为单时刻单视角重建压力测试。补充一个可导出校准相机、
   米/秒单位、网格/深度/法向、持久材料点与物体 ID 的最小 RGB 合成验证集。
   先使用静态支撑和两个运动固体；无碰撞的已知运动用于隔离相机、形变与身份
   问题。它与 S5 球/平面优化算子夹具是两种不同验证，后者不能代替 RGB 训练。
2. 训练仍只输入异步单视角 RGB；同步新视角和三维轨迹只在独立评估中使用。
   固定训练/验证/测试相机与时间清单，防止把评估几何或身份真值送入训练。
   同一相机约定、容量、起点和预算下，先复核默认关闭的重建基线，再逐项验证。
3. 除 RGB 指标外，测独立新视角深度/法向/表面误差、材料点轨迹和遮挡前后 ID。
   对低可见性报告覆盖率与未知比例；不能仅在已筛出的容易点上声称整体成功。
   在查看候选结果前冻结指标定义与容许误差，禁止根据结果反向放宽门槛。
4. 只有表面/身份/时间标度获证据支持，才另行建立含已知碰撞的验证集并检查
   整段穿透量和真实激活区间。只有模拟器确实提供独立应力/接触力真值时才做
   相应物理评估；人工设定轨迹不产生可验证的应力或接触力真值。

这些是未解决原目标的后续依赖；本轮不会据此自动增加损失权重、并行扫参或
启用本构/未知力/接触力模块。


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

### S3-U final result and plan revision (2026-11-03)

Executed the declared16k warmup+6k S3 continuation and completed native val/test,
standard image metrics, fixed86/fixed188/original-trajectory comparisons and
full-state/lineage audits. Unified dynamic population only: no input/output
physical body count or instance-ID estimation.49 related tests, both40-to80
recovery checks and exact frozen-S1/source/RNG/Adam/sampler checks pass.

Use the preselected6k endpoint: validation dynamic25.181221dB, compared with
original S3's25.392299dB(-0.211079). Static34.999955dB; test dynamic25.393077dB.
Full-frame validation28.165765dB/SSIM0.969167/LPIPS0.045682. Preserve4k's higher
26.156537dB as an intermediate observation; no checkpoint selection, weight
increase or retraining follows. Same15,507-point budget is not matched capacity:
actual parameters3,645,217 versus5,831,273, with different initialization/coverage.

Sampled native/fixed-view inspection still shows yellow sheets, stretched
fragments and incomplete foreground fit. Numeric gates/valid derivative support
do not certify complete geometry, boundary-safe neighborhoods, physical identity
or calibrated stress. The unified pool satisfies the requested count-free
representation; it has not established an overall improvement over original S3.
Evidence and six decoded H264 videos are in staged_reconstruction/S3_union/
assessment.md and output/staged_reconstruction/S3_union/physics/index.html.

This additional branch is complete; do not automatically extend to S4-U--S6-U
or transplant original S4's per-group row bindings. Next research dependency
remains the calibrated synthetic RGB geometry audit described above. Training
must preserve this user's choice of a unified pool without estimating count;
any independent truth remains evaluation-only. Original contact/constitutive/
force goals remain unresolved and disabled; no historical paused work resumes.

### Unknown static regions: S1-A (user requirement, 2026-11-03)

The user requires discovering static support from different times/cameras,
without knowing that blue is static. Historical S1 and S3-U do not satisfy this:
S1 uses blue masks and S3-U also retains yellow dynamic appearance objectives.
Keep those results, with their prior dependence explicit.

The separate opt-in S1-A branch uses supplied calibrated cameras and training
RGB (dataset alpha only composites RGB), unmasked tracks, disjoint fit/check
fixed-world-point reprojection, parallax and temporal evidence. Generic image
regions receive geometric seeds; unsupported/occluded/mixed regions stay unknown.
No S1 weights/hull/old color-selected tracks or blue/yellow thresholds are inputs.
This is static-motion discovery, not material segmentation or object counting.

The first discovery finds97 fixed-point candidate tracklets,200 incompatible
with a fixed point and515 unknown. A conservative543-point shell passes the
declared geometric initialization gate. Actual mask inspection shows mostly
late support faces and no accepted support in sampled early frames; this is
partial recovery, not complete automatic scene decomposition. Execute only the
declared12k static pilot after tests/native parity/full-state recovery; report
unknown coverage rather than filling it from color priors. No automatic physics
or S3-U static-bank replacement is admitted by these partial observations.
Detailed protocol/results: [S1_automatic/plan.md](staged_reconstruction/S1_automatic/plan.md).

S1-A final12k result:17 tests, independent recovery and full-state audit pass.
51/120 images have any positive support(41 trainable); all-pixel support0.6507%
is not static recall. Training inferred-region PSNR24.069166dB is not comparable
to S1 validation. Actual native images retain missing tall surfaces and early
fragments/spikes. Reject S3-U static replacement; complete automatic reconstruction
remains unresolved. Preserve this negative result, improve geometric correspondence
and coverage evidence before another integration, without a color-prior fallback.

### S1-CT: revisable cross-temporal confidence (completed 2026-11-03)

Completed opt-in fixed/progressive12k static arms using the same1954-point
initialization and camera sequence, without static-color labels or body counts.
Only the progressive arm refreshes geometry-capped confidence every1k steps;
both admission and revocation occur.29 tests and full confidence/Adam/RNG/sampler
resume plus source/history/native-renderer audits pass. This is a cold-start
adaptation of the confidence idea, not a replication with a historical3DGS prior.

Common-candidate training PSNR23.1274->29.0826dB and held-out-camera inferred
region16.1345->19.6458dB improve, but old-seed region26.6069->26.0643dB regresses.
Positive support41->81/120 images; all-pixel0.6261%->1.0921% is not static recall.
Total12k updates match;6000 versus10946 have nonzero image supervision. Inferred
validation masks are not independent static or geometric truth.

Actual native fixed86/188 and early/middle/late images still show absent tall
surfaces, fragmented early geometry, blur and spill. Full static reconstruction
remains unresolved; do not replace S3-U or claim contact/identity correctness.
Next evidence priority: long-baseline candidate coverage, depth/occlusion
visibility and independent synthetic geometry/static-GT admission audits. No
loss sweep, constitutive/contact/force or old automation resumption follows.
Frozen evaluation and weights are unchanged; a separate video-RGB path repair
retains failed outputs and exports a fully decoded200-frame H264 comparison.
See [S1-CT measured report](staged_reconstruction/S1_cross_temporal/assessment.md).

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

### J5后续依赖未准入，自动跟进暂停（2026-11-04）

只读核对reliable_motion预训练图：8592/12882活跃父点被覆盖（66.6977%），
低于冻结80%门槛；协议/graph数组SHA一致。尚无该候选的新训练模型或完整度
结果，不能接入下一静动联合训练。最大共同时间距离筛选是主要覆盖损失环节，
仅扩展当前可靠锚点的事后诊断不构成新图准入，不放宽门槛。
用户已在依赖任务“动态物体保持完整度”要求“这轮后结束”；本线不接续启动、
重启或改动该任务。pidg自动跟进已PAUSED，保留本线全部状态/记录，当前无本线
GPU训练。J5仍completed_read_only_not_promoted，整体结构清理目标未达到。
详见RubberDuck_visibility_audit/v1/dependency_review.json及progress.md。
仅用户明确恢复后先读依赖最终报告，再决定独立预声明候选；不推断RGB无解。

### J6: user-authorized I10 + accepted J4 static (2026-11-05)

User accepts the displayed J4 static starting quality and requests I10 joint
training. Fixed protocol d02b4976 uses actual I10 final2k non-static model,
full Adam/RNG/sampler/physics cache plus accepted J4 static model/full Adam.
No old I10 target replacement: matched frozen-static and opacity/SH-open arms
use full RGB, unchanged 0.1 parent self-depth, 2000 updates each. Eleven tests,
native flags-off/gradients, 20-update smoke and exact two-process state/Adam
replay passed. This is a new bounded authorization; paused jobs stay paused.
See RubberDuck_I10_joint; no new quality result or geometry certification yet.

### J6 completed: I10 + accepted J4 static (2026-11-05)

Both matched arms completed 2000 full-RGB updates; initial/control/joint all30
native val PSNR 29.068226/31.401124/31.994673 dB, joint +0.593549 versus
control. Complete two-source/dual-Adam state and all4000 camera samples audited;
static geometry/physics exact, dynamic Adam26k, joint static opacity/SH16k.
180 image hashes, 6 full Gaussian all-field PLY matches +9 world XYZ exports,
two H264 videos each200 decoded frames verified. Thirty image files actually
reviewed: sampled static non-regression retained, but fixed86/t200 and188/t100
dynamic fragments/dark high-alpha trails persist. completed_not_promoted;
pending:false. This fulfills the requested bounded joint training, not the
overall geometric-cleanup objective. No new physics updates, automatic
extension, old-task resume or historical result rewrite. Full evidence in
RubberDuck_I10_joint/v1/assessment.md and final_review.json.
