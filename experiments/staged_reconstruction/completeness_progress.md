# 跨视角物体完整度：持续实验记录

用户目标：在其他相机位置渲染时，物体保持完整。当前目标仍未完成。
物理分支按根 AGENTS.md 独立推进；此处属于可关闭的重建与运动先验实验。

共同基线为用户指定的 S3-U final6k，保持原 PIDG 场、渲染器、背景
及优化器/RNG 状态。数据在每个时刻只有一个真实相机观测，未提供
固定新视角的同步 GT。不能将 PSNR、alpha 或代理形状指标当成完整性证明。

| 轮次 | 新增条件 | 验证集动态 PSNR | 结论 |
|---|---|---:|---|
| S3-U | 固定基线 |25.1812|新视角仍有缺失和薄片|
| I0 | 原模型继续2k步，对照 |24.8071|未改善完整性|
| I1 | 冻结轨迹邻接图、局部长度约束 |26.5556|有图像指标收益，完整性失败|
| I2 | 在 I1 条件上改为256控制点的共享运动 |25.3992|部分区域更平滑，仍有缺失和薄片|
| I3 | 原稠密运动、I1先验、训练帧伪深度顺序 |25.1853|顺序诊断改善，但完整性失败|
| I4 | 原S3-U第0步即加入相同伪深度排序 |23.2489（仅16k预热）|静态保留护栏失败，未进入物理阶段；仍有长条/碎片|
| I5 | 原第0步加入旋转感知局部刚性/旋转相似先验 |21.9241（仅16k预热）|静态保留护栏失败；仍有明显缺失和碎片|
| I6 | 原第0步使用16个连续控制点的共享运动，原image/physics目标 |25.5403（仅16k预热）|静态保留护栏失败；新视角仍有大片缺失/拉散|
| I7 | 原第0步使用时间条件可逆位置映射，保留原协方差头与image/physics目标 |25.4532（仅16k预热）|静态保留护栏失败；可逆性未解决新视角碎片/拉长|
| I8 | 原第0步使用同一可逆映射共同输运位置和协方差 |23.8760（仅16k预热）|静态PSNR下降2.8281dB，护栏失败；实际图像仍有缺失、薄片和长桥|
| I9 | I8同架构增加逐点相对Jacobian的鲁棒对数伸长先验 |22.9463（仅16k预热）|静态PSNR下降3.7489dB，护栏失败；相对形变减小但实际缺面/薄片未修复，正式物理0步|
| I10 | S3-U final6k继续2k，在可见静态区域中和冻结背景RGB残差 |26.1679|静态区域动态alpha减小，实际缺面/碎片仍在，完整度失败；本次同预算对照26.4786，单次指标差异受数值重复性限制|

I0–I3正式实验从相同 S3-U final6k 开始，没有挑选或替换基线中间帧。
I3 使用新的训练深度先验，属于不同监督条件，不能只归因于网络结构。
I4–I9从原S3-U第0步状态开始，原16k+6k预算及采样/宽度计划不变。
I7新增33,420个共享位置参数及独立Adam组；原参数/初态及Adam组保留，
开启时停用原位置输出头。不能把扩展后的全部初态称为与原模型完全相同。
I4/I5/I6/I7/I8/I9在16k原准入门槛失败，不能把其预热指标当成22k匹配预算结果。
I8开启时原位置及协方差头均停用，仍保留原张量和Adam组；其有效容量减少。
I10保持原S3-U表示/初始化和物理项，只改变继续训练中的部分RGB目标。
历史I0与本次I10对照的完整初态/Adam/RNG/2k采样相同，但微小数值差异
随训练放大，动态PSNR相差1.6715dB。不能把单次候选/对照差异当成稳定因果效应；
I10的完整度失败有实际图像支持，未晋级、未宣称6×200验收通过。

CoTracker3另做训练图像对应审计：28个短窗口的1064个候选仅31条通过严格
检查（早24/中0/晚7），即使全绑定也达不到原R1门槛，未接入模型训练。
固定支持、跨遮挡身份和完整3D几何仍未建立。

验收预先固定为6个相机位置和9个筛查时刻。筛查通过的候选还必须完成
6×200时刻扫描与实际图像复核，同时满足动态 PSNR 相对 S3-U 下降不超过
0.3dB、动态 alpha 至少0.95。明显缺失、薄片、碎片、消失或误合并均失败。
尚无候选完成该验收。迭代选择只使用 train/val，未用 test RGB 挑选模型。

- I1结果：`motion_integrity/assessment.md`
- I2结果：`sparse_motion/assessment.md`
- I2图像：`../../output/staged_reconstruction/sparse_motion/screen/index.html`
- I3研究、准入与完整状态记录：`ordinal_depth/`
- I4失败准入与16k完整状态审计：`early_depth/assessment.md`
- CoTracker3证据不足：`learned_tracks/assessment.md`
- I5研究与独立预检：`local_rigidity/research.md`
- I5失败结果：`local_rigidity/assessment.md`
- I6研究与独立预检：`coarse_motion/research.md`
- I6失败结果：`coarse_motion/assessment.md`
- I2完整空间映射只读诊断：`sparse_motion/blend_assessment.md`
- I7文献依据与独立预检：`invertible_motion/research.md`
- I7失败结果：`invertible_motion/assessment.md`
- 视频深度独立资格检查：`video_depth/assessment.md`（未准入训练）
- I7实际位置映射应变诊断：`invertible_motion/strain_assessment.md`（固定点/时刻未发现极端局部塌缩）
- I7实际协方差与原生推理干预：`invertible_motion/covariance_assessment.md`（仅归一化不能恢复完整物体，未训练新模型）
- I8最终16k准入、完整状态审计、实际图像和形变诊断：`covariance_transport/assessment.md`
- 静态初始化/训练梯度/最终遮挡的独立诊断：`static_dynamic_coupling/assessment.md`（当次为只读诊断）
- 冻结背景残差的同起点2k继续训练消融、原生图像及对照数值重复性审计：`background_residual/assessment.md`、`background_residual/control_repeat_audit.json`（不涵盖初始化或早期训练损伤）
- I9论文改编、预注册和正式训练前资格检查：`relative_strain/research.md`、`relative_strain/preflight_assessment.md`
- I9与I8的7k同预算实际图像/完整初态和采样/形变对照：`relative_strain/preview07000_assessment.md`
- I9最终16k失败准入、完整状态、同预算I8对照与实际原生图像：`relative_strain/assessment.md`
- MonST3R官方模型12个相邻训练图像对实际推理与资格失败：`monst3r_geometry/assessment.md`（未生成PIDG训练目标）
- MonST3R新的12个较大时间间隔训练图像对：静态尺度对照与已知相机尺度两组均0/12通过；10项合成几何测试通过，实际点图置信度/投影仍不合格，未生成训练目标：`calibrated_pointmaps/assessment.md`
- 训练视角覆盖与固定点集的只读审计：`canonical_support/assessment.md`（S3-U/I8训练前景alpha很高，仍不能保证新视角完整）

相关原始论文：
[Dynamic3DGaussians](https://dynamic3dgaussians.github.io/paper.pdf)、
[SC-GS](https://arxiv.org/html/2312.14937v2)、
[MoSca](https://arxiv.org/html/2405.17421v2)、
[MoDGS](https://arxiv.org/html/2406.00434v4)。这些实验是针对 PIDG 的改编，
不声称复现完整论文系统。
