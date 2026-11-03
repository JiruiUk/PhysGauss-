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
| I7 | 原第0步使用时间条件可逆位置映射，保留原协方差头与image/physics目标 |待训练|8单测、原生关闭对照、20步物理短测、完整恢复审计通过|

I0–I3正式实验从相同 S3-U final6k 开始，没有挑选或替换基线中间帧。
I3 使用新的训练深度先验，属于不同监督条件，不能只归因于网络结构。
I4–I7从原S3-U第0步状态开始，原16k+6k预算及采样/宽度计划不变。
I7新增33,420个共享位置参数及独立Adam组；原参数/初态及Adam组保留，
开启时停用原位置输出头。不能把扩展后的全部初态称为与原模型完全相同。
I4/I5/I6在16k原准入门槛失败，不能把其预热指标当成22k匹配预算结果。

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

相关原始论文：
[Dynamic3DGaussians](https://dynamic3dgaussians.github.io/paper.pdf)、
[SC-GS](https://arxiv.org/html/2312.14937v2)、
[MoSca](https://arxiv.org/html/2405.17421v2)、
[MoDGS](https://arxiv.org/html/2406.00434v4)。这些实验是针对 PIDG 的改编，
不声称复现完整论文系统。
