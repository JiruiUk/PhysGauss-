# PIDG/C1：无实例标签的 Object ID 学习可行性

检索与本地核对日期：2026-10-30。
这里将 objectiveID 理解为 **Object ID：物体实例归属及跨时间身份**。
本文件是研究与实现建议，没有修改模型或启动训练。

## 判断

**可以在当前 C1 上研究自监督物体分组，但需要让分组影响重建和运动解释。**
仅在训练完成的轨迹／RGB 上聚类，或增加一个不影响渲染的 ID 输出头，不足以产生可靠物体身份。
最适合借鉴的是 DynaVol 的组合重建、MovingParts 的分组运动解释，以及 AdaSlot/QASA
的有效槽位选择。它们都不是现成的 PIDG 插件，当前 Rubber Duck 上的有效性尚未验证。

按本轮需求，Object ID 分支不使用人工实例 ID、实例 mask、跟踪框或固定聚类伪 ID 作交叉熵标签。
RGB 重建、训练帧间几何／光度一致性提供训练信号。这仍有优化目标，“自监督”不是没有损失。

## 相关工作与约束

| 工作 | 核实的方法 | 对当前项目的判断 |
|---|---|---|
| [DynaVol，ICLR 2024](https://arxiv.org/pdf/2305.00393) | 无实例标签的动态场景分解；运动/颜色图初始化后，slot 与组合 NeRF 通过渲染目标联合优化；支持初始单视角设置 | 最值得借鉴训练结构。原版为 voxel/NeRF，且预设 slot 容量，不能直接替换 PIDG |
| [MovingParts，ICLR 2024](https://openreview.net/forum?id=QQ6RgKYiQq)；[作者代码](https://github.com/SilenKZYoung/MovingParts) | 渲染与双向映射一致性驱动分组运动；使用分段刚性假设和组后合并 | 借鉴“分组参与运动解码”。Duck 会拉伸，必须保留受约束的非刚性残差，不能照搬全刚性 |
| [AdaSlot，CVPR 2024](https://arxiv.org/html/2406.09196v1)；[作者代码](https://github.com/amazon-science/AdaSlot) | 从候选 slots 中学习启用/丢弃；常用 DINO 特征重建，也有像素重建实验 | 借鉴数量门控；仍有 K_max，原方法不是动态 3D 身份模型 |
| [QASA，ECCV 2026，核对 v2](https://arxiv.org/html/2601.12936v2)；[作者代码](https://github.com/ouyangtianran/QASA-tianran) | 用无标签的 slot quality 选择有效 slots，区分选择与重建；使用 DINO 特征 | 借鉴自适应数量与质量筛选。仍有 K_max，不能声称无容量上限，也未直接解决 3D 软体时间身份 |
| [ParticleSplat，2026 预印本](https://arxiv.org/html/2609.19463v1) | 以多视角重建学习 object-centric latent particles，再解码为 Gaussians，无实例 mask 标签 | 证明 GS 与无实例标签组合重建可以结合；但固定 particle budget、前馈 VAE、多视角静态状态设置与当前逐场景动态 C1 不同 |

DynaVol 也使用运动/颜色图作为初始化。与现有离线检查的关键区别是：它随后让物体表示参与
组合渲染并继续联合学习。只复制初始化聚类，不能把论文后续优化带来的能力归给当前聚类。
原文还会将初始连通分量按 slot 数进行整理，因此原版也不完全满足此前“不指定数量”的要求。
[依据：DynaVol，训练流程及 Appendix B](https://arxiv.org/pdf/2305.00393)。

MovingParts 发现的首先是运动刚性部件，未保证一整只软鸭只有一个组；原方法也有候选组上限。
这里推荐的是它的可学习运动分解思路，而不是直接把部件标签当物体或材质标签。
[依据：MovingParts，方法与分组合并](https://arxiv.org/pdf/2303.05703)。

## 不作为本轮严格无实例标签训练的首选

- [Gaussian Grouping，ECCV 2024](https://github.com/lkeab/gaussian-grouping)：用 SAM 生成的 2D mask 监督 Gaussian identity encoding。
- [SADG](https://arxiv.org/abs/2411.19290)：不依赖 object tracker/IDs，但仍借助 SAM masks 做特征对比学习。
- [SplatFlow，CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Sun_SplatFlow_Self-Supervised_Dynamic_Gaussian_Splatting_in_Neural_Motion_Flow_Field_CVPR_2025_paper.pdf)：无需人工 3D box，但使用 LiDAR 运动预训练及基础模型特征。
- [Scene-Agnostic Object-Centric Representation Learning for 3DGS，2026](https://arxiv.org/html/2604.09045v2)：上游 OCL 不用人工实例标签，下游 GS 仍用其 masks/codebook 作为目标监督身份特征；不是单个 C1 内直接发现身份。

这些工作不能简单叫作“有人工标签”，但“无需人工标注”和“没有实例伪标签／外部教师”也不是同一个条件。
采用 DINO 等自监督预训练特征本身不等于使用人工实例标签；不过它依然引入外部数据和预训练先验，应单独声明。

## 当前模型核对

| 当前组件 | 实际位置 | 含义 |
|---|---|---|
| 形变特征/解码器 | `variants/color_repair/code/scene_PIDG/network.py::Grid4D,DeformNetwork` | 空间/时间特征融合后输出逐 kernel quaternion+translation、尺度和旋转偏移；没有物体归属分支 |
| attention_score | 同文件 `DeformNetwork.forward` | 空间通道门控缓存，不是 slot 间竞争的物体 attention，不能直接当 Object ID |
| 时间查询 | `variants/color_repair/code/scene_PIDG/deform_model.py::DeformModel.step` | 可以在训练时间上查询轨迹，为共享物体运动提供输入 |
| 3D 中心与渲染 | `variants/color_repair/code/gaussian_renderer/__init__.py::render` | kernel 中心是 `R(q_i)X_i+b_i`；支持 override_color，可渲染软归属作检查 |
| 已有聚类 | `variants/object_centers/*_review.py` | 都是离线候选，没有把实例归属接入 C1 训练 |
| 中心基础函数 | `variants/object_centers/model.py` | 已有加权中心、局部残差、轨迹和拓扑继承原型，尚未集成训练；硬 ID 代码不能直接冒充软归属实现 |

实际训练数据有 120 个不同时间、120 个不同相机姿态；不是每个时刻都有同步多视角。
因此不能把不同时刻的图像当作 ParticleSplat 的同一静态状态多视角输入。
DynaVol 明确包含 V=1 设置，时间输入方式更接近本项目，但这不是当前场景成功的保证。

训练目录中的 `transforms_test.json` 实际是 30 张 validation，最终 50 张 test 另行加载。
未来训练损失、邻接帧图像、特征提取只能读取 120 张 train；validation 只用于选择，test 留到选择锁定后。
相邻帧也必须在 train 内重新连线，不能沿用可能指向 validation/test 图像的索引。

## 推荐的定制结构：研究提案，不是已验证方法

保留 PIDG 的原有应力、物理路径与 renderer，在独立开关下加入：

1. **Canonical 身份分支。** 用 canonical 特征预测 kernel 对物体的软归属 `p_ik`。
   归属不以时间为输入；时间驱动物体运动。数值 ID 只在同一次训练内有意义，跨运行可整体置换。
2. **物体共享运动。** 各候选物体学习中心运动和可选整体旋转；其 kernel 共同使用它们。
3. **局部非刚性残差。** 保留 Duck 的拉伸、弯曲能力，但限制残差无限吸收全部运动。
4. **可视的软归属。** 以原本相同的 alpha 合成权重渲染各组贡献；未知点保留原路径。

例如，可研究如下分解：

`x_i(t) = sum_k p_ik [c_k(t) + R_k(t)(X_i-mu_k)] + r_i(t)`。

`mu_k` 是 canonical 组中心，`c_k(t)` 是世界坐标中的中心趋势；均为几何量，不是已知质量中心。
`r_i(t)` 是局部残差，避免强制软体全刚性。这只是设计示意，具体初始化、软归属规范和优化仍需测试。

**分组必须有作用。** 若只增加 ID 头，RGB 仍完全由旧 C1 决定，RGB 损失对该头没有梯度。
即使把组 mask 渲染出来，只要重新合成后与分组无关，也不会凭空学出物体。
同理，只优化“同一 kernel 的 ID 随时间相同”，并不能保证不同 kernel 属于正确的同一物体。

**防止绕过与坍缩。** 仅加时间/邻域一致性允许所有点同组；仅加熵约束也不能阻止此解。
另一方面，过强的少组惩罚会把同色鸭子合并，过强的均匀占用会强制凑数。
需要组合重建、受约束局部残差、可见性处理、槽位竞争及数量/质量选择共同起作用。
对软归属的零均值残差约束，需要一致的软权重投影，不能直接套用硬分组减均值代码。

建议先研究 DynaVol 式组合重建与运动解释，再加入 AdaSlot/QASA 式数量选择。
这是结合文献和本地失败案例的实现判断，不是文献已经证明适用于 PIDG 的结论。

## “不设置组数”的边界

不能把固定实际 K 偷换成“自动发现”。上述 slot 工作仍需要候选容量或 `K_max`。
自适应启用数量不同于强制输出 5/6 个物体，但依然存在容量超参数。

如果继续要求完全不设置 slot 容量，应研究可学习亲和图配合分裂/合并或其他非参数机制，
让连通组动态产生；它仍需要距离、复杂度或停止条件。这是额外定制研究，不能宣称直接复用上述论文就已满足。
本轮筛到的工作没有同时作为“现成 PIDG 插件、软体、无实例教师、无容量上限”的可用成品。

## 监督边界与消融

当前 C1 的 `metric_config.json` 中 mask_weight=0.03，已经有总前景 alpha 监督。
所以应区分：

- **Object ID 自监督版本：** 保持 C1 对照配方，但 ID 不使用实例标签；总前景 mask 不能告诉模型每只鸭子的身份。
- **完整 RGB-only 版本：** 关闭已有 GT alpha/mask 以及依赖它的前景加权监督，另做匹配对照；不能继续声称复用的 C1 已完全不含 mask 监督。

严格 RGB-only 应从相同随机初始化，或能证明只用 RGB 训练的检查点开始。
仅在已用 alpha 训练过的 C1 上关闭新损失，不能消除其已有监督历史。

Object ID 与 material_id 分开管理。本提案不做材料分割、不新增逐 kernel 自由刚度或外力，也不替换 PIDG 预测应力。
在身份未可靠之前，不叠加跨物体排斥，以免把同一只鸭子的部分互相推开。

建议验证顺序：

1. 合成两个/三个可变形对象：实例真值仅用于评估，不进入训练或初始化。
2. 检查新开关关闭时基线输出一致、RGB 梯度能到达归属分支、残差不会完全绕过该分支。
3. C1 对照、C1+可学习组合分组、再加数量自适应；同初始化/数据/预算，先短验证。
4. 同时报图像指标和分组质量。无实例真值时只报告稳定性/连通性等诊断，不能把它们称为真实实例准确率。
5. 数值 ID 跨种子重编号不算跟踪失败；同一次训练中物体跨时间交换身份才是需检查的问题。

尚未安装论文代码、修改训练网络或安排长训练。论文效果不能直接作为本地 PSNR 增益承诺。
