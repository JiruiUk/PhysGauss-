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
