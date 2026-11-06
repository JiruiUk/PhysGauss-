# Rubber Duck 动态表面追踪、可撤销连接与结构完整性技术报告

**报告日期：2026-11-06。对象：PIDG 扩展中的 J6 工作基线及已完成的 L5 匹配实验。**

本文回答四个问题：当前模型如何利用跨时间信息；连接、厚度与完整性约束究竟约束了什么；实测哪些方面改善、哪些方面失败；下一步应解决什么。所有结果来自已保存并闭环审计的实验。本次报告整理只读取记录、重算统计与制作图表，**没有新增训练、改变权重或修改实验门槛**。

## 1. 核心结论

**L5 实现了可靠局部区域的连接与观测支撑改善，但尚未实现全时段、整只物体的表面完整重建。新权重已保存，未替换 J6 工作基线。**

两臂均从同一完整 J6 状态开始，均启用时序记忆，各训练 2000 步；候选额外加入局部结构约束。全 30 张验证图的平均 PSNR 从匹配时序对照的 **31.2234 dB** 提高到 **31.8597 dB**，增加 **0.6363 dB**，但仍比 J6 的 **31.9947 dB** 低 **0.1350 dB**。两项预定 RGB 不退化门槛通过。

固定初始局部片及帧分母上，连接惩罚较对照下降 **58.92%**，面积支撑越界从 **19 降至 13** 个片帧，观测贡献低于父参照 0.8 倍的片帧从 **96 降至 67**。这些说明约束在局部产生作用。然而，厚度支撑越界从 **2 增至 3**；固定 86/188 视角的动态碎裂、缺口、拉丝及暗色高 alpha 支撑没有一致消除，晚期部分代理指标明显恶化。**人工整体完整性与清理门槛失败**，最终状态为 `completed_not_promoted`，无待处理复核。

最明确的覆盖瓶颈是：初始 60 个结构片仅涉及 **275/15507＝1.77% 的不同动态高斯行**，这不是表面积覆盖率；按片起始帧分为早、中、晚三段是 **59/1/0**。固定 360 个片帧参照的分布为 **354/6/0**。候选实际只有 **166/2000＝8.3%** 的更新使用结构项，晚段没有结构应用。有限的早期关联尚不能把不同时间看到的表面统一成可靠的完整模型。

![已保存结果的汇总图：RGB、记忆可靠性、参照时间分布与暗高alpha代理](F:/A_Edinburgh/PhysGauss/experiments/staged_reconstruction/RubberDuck_surface_support/technical_report/results_overview.png)

图 1：本报告根据已保存统计绘制。A 为全部 30 张验证图；B 最后一次刷新在 1750 步，不能当成 2000 步的新测量；C 的单位是片帧记录；D 为固定 86 视角的渲染代理，不是几何真值。

## 2. 任务、数据与可辨识性边界

### 2.1 目标数据仍是 Rubber Duck

最终目标是 PIDG Rubber Duck 场景中的动态物体：减少拖影和悬浮支撑，同时保持结构、连接和完整性。早先辅助合成深度研究用于隔离机制；本报告的 J6/L5 指标、图片和训练均来自 Rubber Duck。Rubber Duck 是 PIDG 的目标物理基准数据；历史目录中的 `real` 标签不能作为真实物理相机采集的证明。

既有 J5 元数据审计核验：训练集有 **120 张 RGB、120 个不同精确时间键、120 个不同相机变换**；同一精确时刻存在多台不同相机观测的时间数为 **0**，每个精确相机位姿只对应一个训练时刻。验证指标另使用协议固定的全部 30 张 val 图。本轮未读取 test 图像。

这意味着模型不是在每个时刻获得同步多视角几何。不同时间图像确实可能补充不同表面，但需要先解决“这些观测是否属于同一个持续表面、其间如何运动、遮挡时是否仍保持对应”。**所有表面可能曾进入画面，并不自动等于已获得可相互融合的同一物质表面的对应关系。** 当前也没有证据证明全表面均被可靠观测。

### 2.2 三类证据不能混用

| 证据 | 可以支持的结论 | 不能支持的结论 |
|---|---|---|
| 30 张 val 的 RGB/SSIM/L1 | 指定相机与时间组合上的图像拟合能力 | 全物体三维几何正确、厚度正确 |
| 训练 RGB 轨迹及邻域检查 | 原图对应与父条件绑定的局部一致性 | 独立三维轨迹真值、真实物体身份 |
| 父模型谱、渲染贡献、alpha、D/alpha | 相对于父模型的支撑与深度变化 | 独立 GT、正确面积或真实体厚 |

固定相机 86/188 在多个时间下的图片是模型诊断输出。它们有助于暴露训练轨迹外的碎裂，但**没有对应的固定跨时间视角独立 GT**。时间标签 2/100/200 是既有评估编号，不解释为秒。黑色低 alpha 区域表示缺失或低支撑，不能当作零误差；D/alpha 是渲染深度代理。

来源：[J5 数据审计](F:/A_Edinburgh/PhysGauss/experiments/staged_reconstruction/RubberDuck_visibility_audit/v1/assessment.md)、[L5 原始评估](F:/A_Edinburgh/PhysGauss/experiments/staged_reconstruction/RubberDuck_surface_support/v1/evaluation.json)。

## 3. 模型表示、历史来源与本轮优化范围

### 3.1 统一动态池与局部关联

模型保留 PIDG 的 canonical Gaussian、时变形变与原生高斯渲染器。可概括为：

\[
G_i^0=(X_i,s_i,q_i,\alpha_i,\mathrm{SH}_i),\qquad
G_i(t)=\mathcal D_\theta(G_i^0,t),\qquad
\hat I_f=\mathcal R(\{G_i(t_f)\},\{G_j^{static}\};K_f,T_f).
\]

此式表示实际调用关系，不要求形变场是简单的刚体变换。当前统一动态池有 **15507 个高斯**，静态库有 **4126 个高斯**。池容量不是物体数量；模型不预设鸭子数量，也没有为每只鸭子建立已知身份的 canonical 实例。

局部表面片是对若干动态高斯的**临时、可撤销关联假设**，并非真实物体分割。统一池可以继续使用，关键是池内是否具有足够可靠的跨时关联与连接；单纯把池拆成多个实例，并不会自动得到正确身份或补全缺失表面。

### 3.2 J6 工作基线的来源

J6 将 I10 的非静态模型状态与用户接受的 J4 静态库及其 Adam 状态组合后，进行独立匹配联合训练。它不是未经历史条件训练的全新未知静态发现模型。**父系存在历史颜色条件训练/选择来源**，本轮没有新增颜色阈值或静动标签，不能因此抹去继承来源。

J6 当时的全 30 张 val：组合起点 29.0682 dB，静态冻结续训 31.4011 dB，静态 opacity/SH 与动态联合 31.9947 dB。RGB 收益存在，但其整体清理门槛也未通过。此处“保留 J6”指继续作为当前比较与工作基线，**不表示 J6 已通过几何完整性验收**。

### 3.3 L5 中哪些量真正更新

| 分支/状态 | L5 两臂的处理 |
|---|---|
| 动态 canonical、grid、deformation 及可训练高斯外观/尺度等 | 按既有动态优化器更新；动态 Adam 26k→28k |
| 静态 XYZ、scale、rotation、opacity、SH | 全部冻结 |
| 静态 Adam | 完整保留且冻结 |
| 速度、应力等 physics 参数、相应 Adam、图缓存 | 完整保留且冻结 |
| 高斯数量与拓扑 | 不增长、不删点、不改变统一池拓扑 |
| 时序及关联控制器 | 两臂刷新时序记忆；候选另刷新局部图资格 |

因此，**L5 是 RGB、时序与局部结构训练，不是新增物理续训**。本轮没有据未知物体身份估计新的刚度、材料、接触或外力。结构正则也不能等同于经过识别的本构规律。

来源：[J6 说明](F:/A_Edinburgh/PhysGauss/experiments/staged_reconstruction/RubberDuck_I10_joint/v1/assessment.md)、[L5 加载与训练实现](F:/A_Edinburgh/PhysGauss/variants/staged_reconstruction/i10_surface_support/runtime.py)。

## 4. 从 RGB 追踪到时序记忆

### 4.1 RGB 片段的拼接与 canonical 绑定

起点是保存的 **730 条双向 RGB tracklet**，每段窗口长 12 帧，相邻窗口移动 4 帧。只使用偶数 fit 观测挑选跨窗连接：重叠观测互为最近，RMS 不超过 0.75 px、最大不超过 1.5 px；存在竞争对象时，替代候选差距至少 0.25 px。冲突连接保留为拒绝或未知，不插值遮挡期间的新目标。

既有拼接得到 75 条长度至少 16 帧的长记忆，其中 61 条长 16 帧、14 条长 20 帧。29 条获得根节点 canonical 提议，最终 **22 条通过绑定和原生资格检查，共 372 个观测坐标**。这些观测按早中晚三段为 **356/16/0**。

canonical 行由父模型条件下的候选提议与分割观测检查决定，不是从 RGB 直接测得的三维物质点。L3 邻域位置还继承“根 RGB 轨迹＋父模型邻点相对投影”的提议方式；**邻点的代理位置不是每个邻点独立追踪出来的真值**。原生贡献、纹理和投影检查降低风险，却不能将父条件假设变为真实表面身份。

奇数 held 帧不参与辅助损失与候选选择，但仍参与普通全图 RGB 训练。因此 held 是对应资格的分离检查，**不是独立未训练图像，更不是三维 GT**。

### 4.2 时序损失与相机位移

对记忆绑定的高斯行 \(i\)，每个观测帧都用自己的相机投影：

\[
p_f=\pi(K_f,T_f,x_i(t_f)),\qquad e_f=p_f-u_f^{RGB},
\qquad\rho(e)=\sqrt{1+\|e\|_2^2}-1.
\]

当前 fit 帧 \(f\) 每步至多检索 4 个实际存在的 fit 端点 \(g\)，按距离当前帧的时间索引差从远到近稳定选择。单对项为：

\[
\ell_{f,g}=\frac{\rho(e_f)+\rho(e_g)+\rho(e_g-e_f)}{3}.
\]

每条记忆按已选端点数量归一化，乘 detached 置信度；再以当前固定适用的提议记忆分母归一化，权重为 **0.01**。暂停或即时检查失败的对应不产生该项梯度；全部提议仍保留。多时间 `dynamic_at` 调用共享 canonical 参数和形变场，因此误差能共同回传。

**已知相机运动通过每帧的 \(K_f,T_f\) 进入投影，不是直接把原始光流当成物体三维位移。** 但这也不是从图像单独恢复出相机无关的三维运动真值；深度、形变和绑定仍可能存在歧义。该机制是显式对应记忆与多时间优化，没有新增 RNN 或 Transformer，也没有完成全时段 canonical 表面融合。

实现：[时序记忆与投影损失](F:/A_Edinburgh/PhysGauss/variants/staged_reconstruction/i10_temporal_memory/core.py)。

## 5. 可靠性、暂停与撤销

可靠性同时检查投影、可见贡献和原始 RGB 局部外观，包括投影 RMS≤2 px、最大≤4 px、正深度>0.01、纹理标准差至少 2（0–255 尺度）、NCC≥0.6，并使用原生完整深度排序贡献。完整要求见冻结协议，不能用单个阈值替代全部准入条件。

结构片需要至少 4 个不同的初始活跃高斯行，父中心分布非共线：第二个协方差特征值至少 \(6.25\times10^{-6}\)。边仍通过原 L3 的距离及长度变化资格检查；父模型近邻本身不等于可靠连接。

每 **250 个完成更新**重新测量资格：

1. 证据缺失或未知时暂停约束，保留点和归属假设。
2. 连续两次明确矛盾撤销；连续两次通过可恢复。
3. 根节点不可靠时，所属邻域约束暂停。
4. 两次刷新之间使用保存的逐帧资格，同时检查当前正深度及投影误差。

**并非每一步都重新做完整原生可见性测量。** 置信度是停止梯度的启发式证据分数，不是在线学习并校准的概率。遮挡、未知、未观测也不直接改叫背景或动态。

候选刷新记录如下。“图节点”是原 L3 假设状态数，不是 L5 的 60 个结构片数，也不等于不同物体或表面积。

| 完成更新 | 活跃记忆 | 活跃观测 | 暂停记忆 | 撤销记忆 | 活跃图节点假设 |
|---:|---:|---:|---:|---:|---:|
| 0 | 22 | 372 | 0 | 0 | 448 |
| 250 | 19 | 320 | 3 | 0 | 375 |
| 500 | 18 | 304 | 4 | 0 | 370 |
| 750 | 17 | 288 | 5 | 0 | 354 |
| 1000 | 16 | 272 | 6 | 0 | 364 |
| 1250 | 17 | 288 | 5 | 0 | 369 |
| 1500 | 14 | 240 | 8 | 0 | 355 |
| 1750 | 12 | 204 | 10 | 0 | 330 |

最后一次刷新是 1750 步，不能声称 2000 步末又测得 12 条；未发生撤销也不证明绑定身份正确。

来源：[控制器审计](F:/A_Edinburgh/PhysGauss/experiments/staged_reconstruction/RubberDuck_surface_support/v1/controller_audit.json)、[状态转移实现](F:/A_Edinburgh/PhysGauss/variants/staged_reconstruction/i10_revocable_surface/core.py)。

## 6. 连接、厚度与完整性约束的准确含义

### 6.1 高斯协方差与局部支撑谱

在同一时刻，给定高斯中心 \(x_i\)、尺度 \(s_i\) 和旋转 \(q_i\)：

\[
\Sigma_i=R(q_i)\operatorname{diag}(s_i^2)R(q_i)^T,\quad
c=\frac1n\sum_i x_i,\quad
C=\frac1n\sum_i[\Sigma_i+(x_i-c)(x_i-c)^T].
\]

令 \(C\) 的特征值平方根按升序为 \(h_1\le h_2\le h_3\)。定义局部支撑厚度代理为 \(h_1\)，切向面积代理为 \(A=h_2h_3\)。它们包含高斯本身和中心分布的尺度。

这些谱量及边长度对共同刚体旋转、平移不变，不受选择哪台相机直接影响。但片成员和资格仍来自观测与父模型。**\(h_1\) 不是鸭子身体厚度，也不是壁厚；\(A\) 不是已重建的真实表面积。** 场景尺度未在此核验为米，不能附加物理单位。

### 6.2 连接：减少椭球支撑间隙，同时防止塌缩

对准入边 \((i,j)\)，令 \(l=\|x_j-x_i\|\)、\(n=(x_j-x_i)/\max(l,10^{-8})\)，父模型同时间长度为 \(l^0\)。沿边方向的双端二倍标准差支撑之和：

\[
r=2\left(\sqrt{n^T\Sigma_i n}+\sqrt{n^T\Sigma_j n}\right).
\]

\[
L_{conn}=\operatorname{mean}_{(i,j)}\left[
\operatorname{ReLU}\!\left(\frac{l-r}{\max(l^0,10^{-8})}\right)^2+
\operatorname{ReLU}\!\left(0.5-\frac{l}{\max(l^0,10^{-8})}\right)^2\right].
\]

第一项抑制支撑间隙，第二项抑制中心距离塌缩至父长度的一半以下。**它不是强制保持每条旧边的精确长度**，也不构造网格流形或保证全局连通。

### 6.3 厚度、面积与单高斯尺度

采用固定软区间：

\[
J(r)=\operatorname{ReLU}(0.8-r)^2+\operatorname{ReLU}(r-1.25)^2.
\]

\[
L_{thick}=J(h_1/h_1^0),\qquad
L_{area\_scale}=J(A/A^0)+\operatorname{mean}_i
\operatorname{ReLU}(s_{i,max}/s_{i,max}^0-1.25)^2.
\]

上标 0 全部指 **J6 同时间自身参照**。尺度上限避免仅靠无限放大椭球“连起来”。然而父模型已有的碎片、错误厚支撑也可能被保留；软约束允许越界，不是硬保证。

### 6.4 观测完整性：保持已观测局部贡献

将某一组高斯赋予 one-hot 覆盖颜色，其余点使用对应覆盖色，在保留全部遮挡者的原生**单次联合深度排序**中渲染组贡献。对实际根 RGB 轨迹位置的 5×5 双线性足迹求均值，得到 \(a_p\)，与父同时间贡献 \(a_p^0\) 比较：

\[
L_{cov}=\operatorname{ReLU}(0.8-a_p/a_p^0)^2.
\]

贡献测量宽度为 400；主 RGB 训练宽度为 1600。每步最多循环选择 3 个有效组。父足迹越界、缺失或贡献过小则记录未知、该项不施加，不把缺失当作零误差。

这里没有真实语义轮廓、GT alpha，也没有全图 alpha-one 先验。**它保护已观测局部的父渲染支撑，不创造新表面，不保证封闭、无洞或全物体完整。**

### 6.5 实际目标与分母

两臂共同使用：

\[
L_{base}=0.8L_1+0.2L_{DSSIM}+L_{memory}.
\]

候选额外使用：

\[
L_{struct}=\frac{0.01}{4}\left[
\frac{\sum_{p\in\mathcal A_f}c_p(L_{conn,p}+L_{thick,p}+L_{area\_scale,p})}{\max(N_f^0,1)}+
\frac{\sum_{p\in\mathcal S_f}c_pL_{cov,p}}{\max(|\mathcal S_f|,1)}\right].
\]

\(N_f^0\) 是当前帧全部初始适用提议片数，不因暂停而缩小；\(\mathcal A_f\) 为当前有效片；\(\mathcal S_f\) 为最多 3 个贡献组，未知项可以返回零但仍保留选择分母。两部分分母不同。\(c_p\) 为片成员资格置信度的最小值，停止梯度。对照没有结构项，但**有相同的时序损失与记忆刷新**。

实现：[几何与足迹公式](F:/A_Edinburgh/PhysGauss/variants/staged_reconstruction/i10_surface_support/core.py)、[资格门控与实际归一化](F:/A_Edinburgh/PhysGauss/variants/staged_reconstruction/i10_surface_support/structure.py)。

## 7. 匹配协议、覆盖与真实使用量

### 7.1 预声明与两臂匹配

| 项目 | 固定设置 |
|---|---|
| 起点 | 同一完整 J6 模型、两个 Adam、参数绑定、RNG、相机剩余栈与 physics 图 |
| 对照 | `temporal_control`，全图 RGB＋时序记忆 |
| 候选 | `surface_support`，相同目标＋上述结构项 |
| 正式训练 | 每臂 2000 步，共 4000 次正式更新 |
| 资格短训 | 每臂另 20 步，不混入正式统计 |
| 主图分辨率 | 1600×900，原生渲染 |
| 可见性/追踪尺度 | 宽 400，阈值像素均在相应尺度解释 |
| 刷新周期 / Adam epsilon | 250 步 / 原值 1e-15 |
| RGB 门槛 | 候选较匹配对照、较 J6，平均 PSNR 最大下降均≤0.3 dB |
| 整体门槛 | 实际固定视角清理与完整性须通过，不能只看 PSNR |
| 调参 | 单候选；无扫参、无门槛放宽、无自动延长 |

预声明提交 `3ca738e1`，实现与资格 `024e8a3a`，完整训练结果 `abf24db7`，提交字节及最终闭环 `9a91d537`。

**协议文字局限：** 冻结协议的 `objective` 字段继承了“control additional loss exactly zero and no refresh”的旧描述，与同文件 `matching` 字段不一致。实际代码、采样使用量及独立审计均确认：本轮两臂都使用时序项及刷新，候选单独新增结构项。本文按实际运行报告，此处透明披露该文档矛盾；没有改写冻结协议或历史结果。

### 7.2 覆盖分母

| 覆盖量 | 已保存或只读核算结果 | 正确解释 |
|---|---:|---|
| 初始局部结构片 | 60 | 关联提议，不是 60 个物体 |
| 片内边应用数 | 337 | 按片计数，不保证是全局不同物理边 |
| 不同动态高斯行 | 275/15507＝1.7734% | 行覆盖，不是面积覆盖 |
| 片起始帧早/中/晚 | 59/1/0 | 每段为 40 个训练列表索引 |
| 固定参照片帧 | 360 | 每片 6 个 fit 帧 |
| 参照片帧早/中/晚 | 354/6/0 | 与片起始时刻分布不同 |
| 参照训练列表索引 | 最小 0、最大 54；20 个不同索引 | 不是时间 200 的结构覆盖 |
| 初始长记忆观测早/中/晚 | 356/16/0 | 372 个坐标，不等于独立表面积 |
| 实际结构应用步早/中/晚 | 158/8/0 | 共 166 步，晚段为零 |

实际有结构应用的采样帧索引为 `0,2,4,6,8,10,12,14,16,18,20,22,44,46,48,50`。这里训练列表索引、图像文件帧号和固定评估时间号是不同索引体系，不能直接混用。

### 7.3 训练是否真正使用了新增约束

| 全 2000 步累计 | 时序对照 | 结构候选 |
|---|---:|---:|
| 时序项活跃步数 | 217 | 211 |
| 端点对应用 | 4768 | 4733 |
| 结构项活跃步数 | 0 | 166 |
| 片应用 | 0 | 1174 |
| 边应用 | 0 | 5527 |
| 原生贡献组应用 | 0 | 397 |
| 平均加权时序损失 | 1.25401944e-4 | 1.21113047e-4 |
| 平均加权结构损失 | 0 | 6.59933936e-7 |

不是“没有更新模型”或“结构项未接通”：已完成实际训练，原生梯度及应用记录均存在。损失数值的单位、分母和门控不同，**不能只凭两个损失标量的比例判断梯度强弱或断定应加大权重**。当前未进行各项分离梯度贡献的完整因果实验。

来源：[冻结协议](F:/A_Edinburgh/PhysGauss/experiments/staged_reconstruction/RubberDuck_surface_support/protocol.json)、[原始局部片库](F:/A_Edinburgh/PhysGauss/experiments/staged_reconstruction/RubberDuck_surface_support/v1/surface_bank.json)、[报告只读派生统计](F:/A_Edinburgh/PhysGauss/experiments/staged_reconstruction/RubberDuck_surface_support/technical_report/derived_metrics.json)。

## 8. 定量结果与通过/失败项

### 8.1 全部 30 张 val：图像质量

| 模型 | 平均 PSNR↑（dB） | 平均 SSIM↑ | 平均 L1↓ |
|---|---:|---:|---:|
| J6 父 | 31.994673 | 0.98042548 | 0.00329669 |
| 时序对照 | 31.223379 | 0.97909809 | 0.00360087 |
| 结构候选 | 31.859706 | 0.98015318 | 0.00336791 |

候选相对对照 **+0.636327 dB**，相对 J6 **−0.134967 dB**，两项 0.3 dB 不退化门槛通过。它显示本轮匹配条件下加入结构项比只有时序约束保留了更好的 RGB；并非超过父模型，也不证明所有视角或动态区域均改善。这里只有一对完整匹配 CUDA 轨迹，没有重复种子置信区间。

### 8.2 固定初始片与帧：局部结构代理

评估保留全部初始 360 个片帧，不按最终活跃片重新筛选。360 个几何记录均存在；父贡献只有 359 个可计算，另 1 个是未知。表中贡献比例类统计的有效分母是 **359**。

| 指标 | J6 | 时序对照 | 结构候选 |
|---|---:|---:|---:|
| 连接惩罚均值↓ | 0.02936779 | 0.04463774 | 0.01833533 |
| 厚度支撑比例超 [0.8,1.25]（/360）↓ | 0 | 2 | 3 |
| 面积比例超区间（/360）↓ | 0 | 19 | 13 |
| 父贡献已知（/360） | 359 | 359 | 359 |
| 已知父贡献对应的当前缺失（/359）↓ | 0 | 0 | 0 |
| 贡献比例<0.8（/359）↓ | 0 | 96 | 67 |
| 平均贡献比例 | 1.000000 | 0.891283 | 0.952521 |

连接、面积与观测支撑有有限改善，厚度代理没有改善。J6 的区间越界为零主要因为区间本来就以自身为参照，并不说明 J6 几何正确。这些是被直接优化的自身参照指标，不能当成独立几何准确率。

### 8.3 所有原始绑定记忆：重投影

| 模型 | fit 平均误差↓（px） | held 平均误差↓（px） | 缺失（/372） |
|---|---:|---:|---:|
| J6 | 0.834847 | 0.860543 | 0 |
| 时序对照 | 0.490510 | 0.534186 | 0 |
| 结构候选 | 0.349053 | 0.365452 | 0 |

对应坐标的一致性明显改善，但它仍只覆盖这 22 条绑定记忆。held 也参与 RGB 训练，因此不能称独立三维追踪正确率。二维投影可以通过多个三维深度/形变解释获得相似误差。

### 8.4 暗色、高 alpha 动态支撑：晚期没有稳定改善

以下定义仅用于 **L5 评估**：动态渲染 RGB 每个通道均<0.05，且动态 alpha>0.5。每张图共 **1,440,000 像素**。它不是训练颜色阈值、背景判别或删点规则，也不等于真实伪影面积；不能与使用其他暗色定义的历史实验直接拼接。

| 固定相机/时间号 | J6 | 时序对照 | 结构候选 |
|---|---:|---:|---:|
| 86/2 | 941 | 875 | 830 |
| 86/100 | 23333 | 22665 | 20876 |
| 86/200 | 36539 | 32488 | 47459 |
| 188/2 | 2647 | 3123 | 3026 |
| 188/100 | 15419 | 18322 | 16181 |
| 188/200 | 28619 | 29369 | 34545 |

候选较对照在 86/200 增加 **14971** 像素，在 188/200 增加 **5176** 像素。结合实际 alpha/深度图，这支持“未能一致清理晚期支撑”的结论；没有独立几何 GT，不能把增加量全部判为错误几何。

原始证据：[结构诊断](F:/A_Edinburgh/PhysGauss/experiments/staged_reconstruction/RubberDuck_surface_support/v1/structural_diagnostic.json)、[RGB/追踪/alpha 评估](F:/A_Edinburgh/PhysGauss/experiments/staged_reconstruction/RubberDuck_surface_support/v1/evaluation.json)。

## 9. 图像与视频证据

下列三模型比较图均依次为 **J6 父 / 时序对照 / 结构候选**；没有固定相机跨时间 GT 列。已有人工复核记录实际查看了 32 张图，本文复用这些结果。分解图的具体层和模型以图内标题为准；alpha 与 D/alpha 是支撑诊断，不能只看 RGB 是否更黑。

### 9.1 固定相机 86：早、中、晚

![相机86时间2：J6、时序对照、结构候选](F:/A_Edinburgh/PhysGauss/output/staged_reconstruction/RubberDuck_surface_support/v1/c086_t002_joint_comparison.png)

图 2：早期仍有松散片、缺口和碎片状表面。少数局部连接改善不足以形成整体连续物体。

![相机86时间100：J6、时序对照、结构候选](F:/A_Edinburgh/PhysGauss/output/staged_reconstruction/RubberDuck_surface_support/v1/c086_t100_joint_comparison.png)

图 3：中期物体形状较可辨，但后侧粒子串仍在。图像拟合和局部支撑改善可以同时存在于仍然碎裂的模型中。

![相机86时间200：晚期的拉伸碎片与缺失](F:/A_Edinburgh/PhysGauss/output/staged_reconstruction/RubberDuck_surface_support/v1/c086_t200_joint_comparison.png)

图 4：晚期仍有明显拉长碎片与缺失，候选没有完成整体清理。

![相机86时间200：静动态RGB、alpha与深度代理分解](F:/A_Edinburgh/PhysGauss/output/staged_reconstruction/RubberDuck_surface_support/v1/c086_t200_decomposition.png)

图 5：候选右上仍有宽大非零 alpha 支撑。RGB 暗化与支撑撤销不是同一件事。

### 9.2 固定相机 188：中、晚

![相机188时间100：破碎表面与横向拖尾](F:/A_Edinburgh/PhysGauss/output/staged_reconstruction/RubberDuck_surface_support/v1/c188_t100_joint_comparison.png)

图 6：中期两臂仍有碎裂与拉长拖尾，没有跨相机稳定修复。

![相机188时间200：三模型RGB比较](F:/A_Edinburgh/PhysGauss/output/staged_reconstruction/RubberDuck_surface_support/v1/c188_t200_joint_comparison.png)

![相机188时间200：RGB与alpha、D除以alpha分解](F:/A_Edinburgh/PhysGauss/output/staged_reconstruction/RubberDuck_surface_support/v1/c188_t200_decomposition.png)

图 7–8：晚期部分 RGB 看似整洁，动态 alpha 和 D/alpha 仍向左延伸。不能以变暗认定支撑已消失。静态各层在三模型之间精确相同。

### 9.3 原训练轨迹控制与视频

[训练列表 index24 原图与模型比较](F:/A_Edinburgh/PhysGauss/output/staged_reconstruction/RubberDuck_surface_support/v1/observed_train_index024.png) 中，前方柱顶物体仍不完整；[index72](F:/A_Edinburgh/PhysGauss/output/staged_reconstruction/RubberDuck_surface_support/v1/observed_train_index072.png) 的边缘仍松散。六张已观测相机/时间组合总体更贴近源 RGB，不能代替其他时间相机组合的真值。

- [固定相机 86 三模型视频](F:/A_Edinburgh/PhysGauss/output/staged_reconstruction/RubberDuck_surface_support/v1/fixed_c086_parent_control_joint.mp4)
- [固定相机 188 三模型视频](F:/A_Edinburgh/PhysGauss/output/staged_reconstruction/RubberDuck_surface_support/v1/fixed_c188_parent_control_joint.mp4)
- [86 解码接触图](F:/A_Edinburgh/PhysGauss/output/staged_reconstruction/RubberDuck_surface_support/v1/fixed_c086_parent_control_joint.decoded.png)、[188 解码接触图](F:/A_Edinburgh/PhysGauss/output/staged_reconstruction/RubberDuck_surface_support/v1/fixed_c188_parent_control_joint.decoded.png)

视频均为 H264 Main、yuv420p、faststart，各已程序完整解码 200 帧，接触图已实际查看。没有实测浏览器播放，也没有声称人工逐帧播放全部视频。

## 10. 当前瓶颈：事实、解释和未知

### 10.1 已被实测支持的事实

1. **关联覆盖少且偏早。** 不同动态行覆盖约 1.77%，晚段没有合格初始长记忆或结构应用；控制器活跃记忆也随训练下降。全局训练持续进行，不等于结构约束持续覆盖全场景。
2. **局部指标改善没有转化为全物体清理。** 连接惩罚、部分面积/贡献指标和投影误差改善，但晚期图像仍破碎，暗高 alpha 代理不一致，人工门槛失败。
3. **没有真实厚度与独立动态三维 GT。** 目前只能评价局部支撑谱、自参照与外观，不能量化真实体厚误差。
4. **本轮没有新增表面生成机制。** 高斯拓扑固定，目标不进行可靠新表面三角化、表面延拓或闭合重建。

### 10.2 与事实相容、但未完成因果隔离的机制解释

**父参照可能保护旧碎片。** 当前厚度、面积、非塌缩长度和贡献下界均以父模型为参照。它们可以抑制崩塌，也可能阻止错误支撑退出。未逐一消融这些项，不能给出每项的因果份额。

**局部高斯支撑不等于表面拓扑。** 椭球靠近、二阶矩稳定，不意味着它们构成无洞、无自交的连续表面；尺度或透明度改变也可以改善局部投影。当前没有对全物体流形、闭合体、可靠前后面厚度施加直接约束。

**相机—时间耦合与对应歧义仍限制动态几何。** 每时刻单视角可由不同三维形状解释。跨时融合需要可靠物质对应与运动补偿；外观相似、遮挡、反光和父模型错误都可能使对应不稳。当前证据不能将反光、某个网络结构或单一损失认定为唯一主因。

### 10.3 不能由本轮推出的结论

- 不能说“追踪完全不起效”：二维重投影已改善；问题是覆盖、身份与三维几何充分性。
- 不能说“所有表面已被可靠看到”：目前没有全表面可见性真值或全时段可信关联。
- 不能说“统一动态池必然失败”：本实验没有隔离池表示与绑定机制的作用。
- 不能说“只要提高别的方向权重或多训即可”：无 GT 新视角的更大权重可能强化模型自己的错误。
- 不能说“加入物理公式就会修好”：身份和邻接不可靠时，物理项同样可能连接错误表面；本轮也没有新增物理更新。
- 不能说“RGB 数据不可解”：失败的是当前固定机制与证据覆盖，尚无该不可能性证明。

## 11. 与此前实验的关系

| 实验 | 实际机制及结果 | 对本轮的意义 |
|---|---|---|
| J4 | 静态 opacity/SH 与动态 RGB 联合，较匹配冻结静态组 +1.2099 dB；整体清理失败 | 外观收益与正确几何可以分离；不同父系，不能当 L5 消融 |
| J6 | I10 非静态状态＋J4 静态的匹配联合，较冻结静态组 +0.5935 dB；整体清理失败 | 当前 L5 完整父状态 |
| L3 | 父时变标量长度的可撤销局部图；31.1956 dB，对照 31.7368 dB；两 RGB 门槛失败 | 提供邻域提议/资格；L5 不沿用旧精确长度损失 |
| L4 | 显式跨窗时序记忆；31.3193 dB，对照 31.6367 dB；两 RGB 门槛失败 | 提供共同的记忆机制，不证明该机制单独成功 |
| L5 | 两臂均时序记忆，候选加支撑连接、谱区间与贡献保持；31.8597 对 31.2234 dB | 局部改善、RGB 门槛通过，整体清理失败 |

**L3/L4/L5 是各自从 J6 重新开始的不同匹配实验，不是连续叠加失败权重。** 各实验的对照结果不同，不能跨运行相减来计算单模块因果贡献。L5 只检验“在共同记忆目标之上增加结构项”，不包含同轮独立 RGB-only 对照，不能完整估计时序与结构的交互效应。

历史来源：[J4](F:/A_Edinburgh/PhysGauss/experiments/staged_reconstruction/RubberDuck_joint_pilot/v1/assessment.md)、[L3](F:/A_Edinburgh/PhysGauss/experiments/staged_reconstruction/RubberDuck_I10_surface_graph/v1/assessment.md)、[L4](F:/A_Edinburgh/PhysGauss/experiments/staged_reconstruction/RubberDuck_temporal_memory/v1/assessment.md)。辅助深度/先验研究的结果不能当作本轮可靠 GT 或未实施的训练监督。

## 12. 工程验证与复现边界

本节汇总已完成、保存于冻结记录中的验证；**本次报告没有重复执行训练资格或 GPU 实验**。

| 验证类别 | 已完成检查 |
|---|---|
| 单测 | 13 个新增结构测试＋14 个保留时序测试，共 27 项通过 |
| 默认关闭 | 原生 RGB/alpha/D 精确一致，梯度在原容差内 |
| 原生反传 | canonical/grid/deform/scale 有限有效梯度；冻结分支不更新 |
| 只读与异常路径 | flags/RNG 恢复、无模型状态突变 |
| 短训练 | 两臂各 20 步 |
| 独立恢复 | 四个独立进程核验模型、两个 Adam、参数绑定、RNG、相机、physics 与控制器精确恢复 |
| Adam 更新验收 | 用户授权的同份固定原生梯度更新精确重放；epsilon 保持 1e-15 |
| 正式过程 | 14 次保存模型刷新原生重放，4000 条相机采样重放，应用量独立核对 |
| 导出 | 6 个完整 Gaussian PLY 字段匹配，另 9 个世界 XYZ 诊断 PLY |
| 评估/媒体 | 30 图均值重算、180 图核验、两视频各 200 帧完整解码；人工复核 32 图 |
| 最终闭环 | 17 个冻结任务 Git 字节检查、15 个最终证据清单及提交字节检查通过 |

原生 CUDA 反传存在历史轨迹非确定性，旧 40 步恢复轨迹失败记录保留。**精确状态恢复＋固定梯度 Adam 精确更新，不等于任意两次独立 CUDA 训练 40 步后逐参数完全一致。** 单对运行不提供重复种子稳定性或统计显著性。

首次准备因单个父足迹缺失中止，未进行优化更新；冻结前按未知处理修复并增加单测，原失败保留。另有报告收尾脚本语法错误，仅修复未冻结的报告写入代码，没有改训练、评估门槛或权重。日志继承 `L4_STEP` 等旧前缀，实际检查点 stage 为独立 L5。无结构步的旧 `raw_*` 字段可能残留，应用量分析采用已核验的加权损失和当步计数。

来源：[原生检查](F:/A_Edinburgh/PhysGauss/experiments/staged_reconstruction/RubberDuck_surface_support/v1/native_check.json)、[恢复验收](F:/A_Edinburgh/PhysGauss/experiments/staged_reconstruction/RubberDuck_surface_support/v1/recovery.json)、[独立审计](F:/A_Edinburgh/PhysGauss/experiments/staged_reconstruction/RubberDuck_surface_support/v1/independent_audit.json)、[最终字节闭环](F:/A_Edinburgh/PhysGauss/experiments/staged_reconstruction/RubberDuck_surface_support/v1/closure_validation.json)。

## 13. 下一步研究建议：尚未执行

下一步的优先级应由证据覆盖决定，而不是直接增加结构权重或训练步数。以下是研究建议，不是已完成实验或自动排队任务。

### 13.1 优先扩展中晚期、跨窗的可信表面关联

先针对缺乏关联的中晚段，建立每个节点自身的持续观测证据、双向与循环一致性及竞争归属检查；明确哪些邻点是独立 RGB 追踪，哪些仍依赖父相对投影。遮挡期间保留未知，重新出现时验证身份，不强行插值连接。报告时间覆盖、不同点覆盖和歧义率，而非只增加边数。

在覆盖门槛达到前，继续对少量早期片加权，无法直接监督晚期未关联区域。新增追踪或特征方法必须固定单候选协议，不能事后按 val 或 GT 选择成功轨迹。

### 13.2 分开研究连接保持与表面补全

连接保持解决已关联节点之间的断裂；表面补全需要从新增可靠观测建立新支撑，或引入明确的表面表示与延拓机制。应分别检验“保留已有正确面”和“增加缺失面”，避免仅靠放大高斯、增大 alpha 或保护旧碎片获得更低连接惩罚。

任何增长或表面表示调整应另立默认关闭的独立候选，在匹配起点验证；不能把本轮固定拓扑试验说成已经测试了补全机制。

### 13.3 真厚度需要前后表面证据

当前最小特征轴只能作为支撑退化/膨胀的代理。若要约束身体或壁厚，需要可靠的前后表面对应、独立尺度证据，或经过明确验证的几何先验。没有这些证据时，应将厚度保留为未知，而不是给所有局部片施加统一物理厚度。

### 13.4 新候选的验收条件

先固定输入、算法、损失、分母和失败条件；再做默认关闭、原生梯度、完整状态恢复及匹配训练。报告关联覆盖、局部支撑、RGB 与固定视角完整性，保留原不退化门槛。若研究损失平衡，应测量各项到目标参数的实际梯度和冲突，不能从平均损失标量直接推断。未见跨相机/时间一致改善前，不晋级、不自动延长。

若追加数据成为可行选项，同时间多相机或可重复轨迹能提供当前缺少的约束；这是增加证据的方向，不是宣告现有 RGB 无解，也不是本次报告已经执行的采集。

## 14. 权重、实现与证据索引

### 14.1 完整检查点

| 名称 | SHA-256 |
|---|---|
| [J6 完整父](F:/A_Edinburgh/PhysGauss/runs/pidg_staged_reconstruction/RubberDuck_I10_joint/v1/formal/joint_depth_trust/checkpoint.pt) | `89109bdb3990c95037924fdf0985e07302ed67065d98971d77209780b96e0502` |
| [L5 时序对照](F:/A_Edinburgh/PhysGauss/runs/pidg_staged_reconstruction/RubberDuck_surface_support/v1/formal/temporal_control/checkpoint.pt) | `75e4a06659d21347aa252e914886d1f45c4700c7be6a585b15b3361cf6475968` |
| [L5 结构候选](F:/A_Edinburgh/PhysGauss/runs/pidg_staged_reconstruction/RubberDuck_surface_support/v1/formal/surface_support/checkpoint.pt) | `57e07cdac6fbbbb2054c544464e6a7b1bf7dd079743890e20de0653386cd5829` |

检查点哈希来自已核验的冻结评估记录。本次报告没有重新加载或修改权重。

### 14.2 最终状态与报告复核

- [原始完整结论](F:/A_Edinburgh/PhysGauss/experiments/staged_reconstruction/RubberDuck_surface_support/v1/assessment.md)
- [最终复核：completed_not_promoted、pending=false](F:/A_Edinburgh/PhysGauss/experiments/staged_reconstruction/RubberDuck_surface_support/v1/final_review.json)
- [人工图像复核与文件哈希](F:/A_Edinburgh/PhysGauss/experiments/staged_reconstruction/RubberDuck_surface_support/v1/manual_review.json)
- [训练源码及输入冻结](F:/A_Edinburgh/PhysGauss/experiments/staged_reconstruction/RubberDuck_surface_support/v1/source_freeze.json)、[评估冻结](F:/A_Edinburgh/PhysGauss/experiments/staged_reconstruction/RubberDuck_surface_support/v1/evaluation_freeze.json)
- [本报告派生统计](F:/A_Edinburgh/PhysGauss/experiments/staged_reconstruction/RubberDuck_surface_support/technical_report/derived_metrics.json)、[资产生成与检查脚本](F:/A_Edinburgh/PhysGauss/experiments/staged_reconstruction/RubberDuck_surface_support/technical_report/build_report_assets.py)、[报告验证及输入 SHA 清单](F:/A_Edinburgh/PhysGauss/experiments/staged_reconstruction/RubberDuck_surface_support/technical_report/report_validation.json)

报告资产的复现命令（在仓库根目录，以保存结果为输入，不进行训练）：

```powershell
& 'F:\A_Edinburgh\PhysGauss\envs\pidg-cu128\python.exe' 'F:\A_Edinburgh\PhysGauss\experiments\staged_reconstruction\RubberDuck_surface_support\technical_report\build_report_assets.py'
```

本报告的判断边界是：**局部投影与支撑已取得有限实测改善，整体动态几何完整性尚未通过；优先缺口仍是可信跨时表面关联的覆盖及其三维充分性。** 原 PIDG、冻结协议、历史模型与结果均保留。
