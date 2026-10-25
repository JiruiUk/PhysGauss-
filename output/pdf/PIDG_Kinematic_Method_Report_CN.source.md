PIDG 方法动机、代码差异、完整长训练结果与后续研究启发


## 1. 时间尺度与速度-轨迹一致性
PIDG 方法动机、代码差异、完整长训练结果与后续研究启发

研究问题：渲染出的运动轨迹与物理网络预测的速度，是否描述同一个运动？本报告以本仓库实际实现和已保存实验产物为依据，复盘统一时间及新增运动学一致性约束的动机、实现、结果和证据边界。

> <b>结论：</b>V4 是本次运行中平均重建指标最好的候选，但现有实验尚不能把优势稳定归因于新约束，速度场也尚未与轨迹导数有效对齐。

已完成实验 | 训练与评估 | 可视化产物
V1-V4 共 4 组 | 每组追加 50k；固定 50 张测试图 | 5 个快照标签 × 50 帧 × 4 组
单场景 / seed 0 | 1600 × 900；最终 PNG 指标核验 | 共 1000 张 GT / Render 拼接图


### 读者应带走的四点

1. 原实现的图像形变与速度场存在耦合缺口，且物理时间输入多除了一次训练帧数。<br/>2. 共同时轴能显著改变评分，但这同时改变评测输入，必须与模型改进分开归因。<br/>3. V4 的平均 PSNR 优势集中在早期少数漏物体帧；大均值增益不等于全帧改善。<br/>4. 在新损失启用前，对照组已经出现明显差异；下一轮应从同一完整 checkpoint 分叉。


### 报告结构

第 2-5 页说明动机、原结构、时间与新目标；第 6-12 页给出实验设置、曲线、逐帧证据、物理诊断及归因检查；第 13 页提出后续验证路线；第 14-15 页列出绝对产物路径和可追溯来源。

把“图像中在运动”与“物理网络认为在运动”联系起来


## 2. 为什么需要这项约束？
把“图像中在运动”与“物理网络认为在运动”联系起来


### 原方法中有两条不同的学习路径

> 新增连接：从形变分支计算轨迹时间导数，与速度分支输出做一致性损失；梯度同时回传到两条分支。网络宽度、深度和输出头不变。

渲染分支通过图像误差优化 Gaussian 及其形变；速度/应力网络通过动量残差学习场量。本场景 use_flow=False，原有光流轨迹损失未启用。两条分支会通过共享坐标间接关联，但没有强制要求预测速度等于渲染轨迹的时间导数。


### 缺口为何会影响可解释性

如果速度近似常量、应力变化很小，动量残差可能很低；与此同时，形变网络仍可通过时间标签生成明显运动。低物理残差或高 PSNR 因而都不能单独证明预测速度对应图像运动。运动学一致性是一项必要联系，用于减少这种分支间不一致；它本身并不足以保证材料、应力或外力正确。


### 本次没有扩大网络

形变仍采用 W256_D0：空间特征经过 Linear(32,256) 和方向门控；时间平面特征经过 Linear(192,256)+ReLU，两者逐元素相乘，再进入 7/4/3 通道输出头。depth=0 表示共享隐藏堆栈为空，并不表示整个网络没有线性层或非线性。速度网络仍是 43→64→64 的共享 MLP，接 3 维速度头和 6 维应力头。

本轮不启用 ISO、残差适配器或新的本构分支。比较重点是时间输入和损失连接，非网络容量消融。源码依据：[S1]-[S3]。

共同物理变量不要求使用相同的内部编码坐标


## 3. 时间尺度：修正什么，假设什么
共同物理变量不要求使用相同的内部编码坐标

位置 | 原实现 | 本次开关启用后
形变 / 渲染 | 直接使用 fid | 使用共同 tau
动量约束速度网络 | fid / 150 | 使用同一 tau，移除额外 /150
训练 / 测试 metadata | 分别归一化到 [0,1] | 保留原协议，或单独使用帧编号协议

原数据的 150 张训练图时间为 j/149，50 张测试图时间为 k/49；图像文件编号则共同覆盖 0001-0200。可选 frame_index 协议令 tau=(frame_id-1)/199，不改变 RGB、相机或 150/50 划分，也不改写原 JSON。

图像编号 | 所属划分 | 原标签时间 | 共同帧时间
0001 | 测试 | 0 | 0
0002 | 训练 | 0 | 1/199 ≈ 0.005025
0005 | 测试 | 1/49 ≈ 0.020408 | 4/199 ≈ 0.020101
0200 | 训练 | 1 | 1

> 帧编号协议依赖“数字文件名代表均匀采样顺序”的假设。它不是已验证的真实时间戳。V1/V2 保留原标签；V3/V4 才使用该额外映射。


### 归一化时间不等于真实秒数

若真实时间跨度为 T，且 t=t0+T·tau，则 v_phys=(1/T)·dPhi/dtau。当前未知 T，速度单位只能表述为“场景距离 / 单位归一化时间”。内部 hash 坐标缩放与 Fourier 时间编码仍保留；轨迹求导经过完整形变调用，包含这些编码的时间变化。

原 compute_acceleration 用自动微分求输入时间导数，传入的 dt 并未参与该导数；仅改 dt 数值不能修复上述额外除法。源码依据：[S2]-[S4]。

对旋转和平移共同求导；保持原速度 / 应力预测分支


## 4. 新方法：让速度对应完整轨迹
对旋转和平移共同求导；保持原速度 / 应力预测分支


### 运动学定义

> Phi(X,tau) = R(q(X,tau))·X + b(X,tau)<br/>v_traj(X,tau) = dPhi(X,tau) / dtau<br/>L_kin = mean Huber_delta(v_pred(X,tau) - v_traj(X,tau))

这里 X 是可优化的 canonical 参考坐标，不保证就是第 0 帧或无应变材料构型。7 通道 d_xyz 中，前 4 个通道是 quaternion，后 3 个通道是平移；它并非一个 3 维位移。R(q) 使用与渲染器相同的归一化 quaternion 旋转。


### 为什么不能只对平移头求导

完整导数含有 dR/dtau·X + db/dtau。即便平移 b 不变，物体绕参考点旋转也会产生速度。只监督 db/dtau 会遗漏这部分运动。Gaussian 的协方差旋转/尺度输出是独立量，本项只对中心轨迹建立约束。


### 梯度如何融合

预测速度和轨迹导数两侧都保留参数梯度。损失能更新时间/空间形变网格、形变输出网络和速度网络。采样的 X 被 detach，使该项不直接通过移动 canonical 坐标来减小误差；图像损失仍沿原路径更新 canonical Gaussian。


### 它与原物理残差的关系

新项补充的是运动学联系，不替换 PIDG 的预测应力，也不引入新的自由刚度或逐 Gaussian 外力。原 canonical 坐标下的动量残差、convective contraction 和采样权重保持未改；因此“同一时间 + 速度一致性”仍不是完整物理正确性的证明。


### 与原实现的差异清单

项目 | 原始 PIDG | 本轮方法
网络 / 参数量 | 形变、速度、应力分支 | 结构不变；未新增可学习参数
训练目标 | 图像 + 特征正则 + 原物理项 | T3 感知项 + 可选速度一致性
推理路径 | R(q)X+b 渲染 | 渲染器不变；可选重映射时间
监督来源 | 训练 RGB 等现有输入 | 仍无 GT 速度、力或新物理标签

源码依据：[S1]-[S3]。新项是训练期正则，图像推理不需额外计算轨迹导数；全部新功能可关闭。

差分可反传；工程可用性与方法有效性分开验证


## 5. 数值实现与训练目标
差分可反传；工程可用性与方法有效性分开验证


### 为什么使用可微有限差分

为了把轨迹时间导数的损失反传进模型，直接使用导数图通常涉及高阶或混合导数。当前自定义 CUDA hash encoder 的高阶输入导数支持有限。本轮采用普通参数反传可处理的轨迹差分，避免依赖未覆盖的坐标二阶导数。

> 内部时刻：v_traj ≈ [Phi(tau+h)-Phi(tau-h)] / (2h)<br/>起点附近：[-3Phi(tau)+4Phi(tau+h)-Phi(tau+2h)] / (2h)<br/>终点附近：[3Phi(tau)-4Phi(tau-h)+Phi(tau-2h)] / (2h)<br/>h = 0.001；二阶精度差分；查询均位于 [0,1]。

配置 | 取值 / 行为
Huber delta / lambda | 0.1 / 0.001
采样 / 间隔 | 256 个粒子；每 10 步；确定性偏移采样
启用 / 渐增 | iter >1000；1000 步 ramp；首次可能施加于 1010
稀疏施加补偿 | 执行步的系数为 lambda × ramp × 10；其余步为 0
差分查询 | 暂关随机特征正则；清空 attention 缓存；查询后恢复

L_total = L_T3 + lambda·ramp·10·L_kin（仅采样迭代施加）。Huber 小误差为 0.5e²，大误差为 delta·(|e|-0.5delta)。间隔补偿表达的是平均目标权重，不等价于每步施加时完全相同的优化轨迹。


### 为何采用保守权重与 Huber

轨迹导数在快速运动或局部变形处可能出现大残差。Huber 在大误差区的梯度受 delta 限制，配合小权重和 ramp，意在减小新项对已有图像重建的冲击。代价是大误差样本的监督可能偏弱；两侧同时更新还允许通过改变轨迹来降低损失。本轮只测试一个权重，未证明该配置最优。

验证结果：38 项相关单元测试通过；真实 CUDA 网格/形变头/速度参数均有有限非零梯度；20→100 完整状态恢复与 PNG 指标核验通过。128 个冻结模型点上，h=0.001 与输入自动微分导数的相对 RMSE 为 0.815%，h=0.0005 时为 0.397%。这只验证局部导数近似，不是真实速度精度。

将时间协议、训练预算、损失改变和快照选择分别记录


## 6. 实验设计与公平比较
将时间协议、训练预算、损失改变和快照选择分别记录

变体 | 时间输入 | 一致性权重 | 比较目的
V0 / T3 | 原 metadata；物理保留旧缩放 | 0 | 复用已完成 T3
V1 | 原 metadata；物理共享时间 | 0 | 只修物理时间输入
V2 | 与 V1 相同 | 0.001 | 原协议下的损失对照
V3 | 共同帧编号时间 | 0 | 新时间协议控制组
V4 | 与 V3 相同 | 0.001 | 新协议下的损失对照

四组都从原始 PIDG W256_D0 最佳 50k 权重出发，新实验初始化新优化器，再追加 50k；继续运行时恢复完整模型、优化器、RNG 和相机采样状态。单场景 Rubber Duck，seed 0，训练 150 图、测试 50 图。全程关闭 densification 和 ISO。


### 共享的 T3 配方

图像目标为 0.8·L1 + 0.2·(1-SSIM)，加 VGG LPIPS 权重 0.02；感知项每 4 步在 800×450 图像计算并乘以 4。空间/时间特征正则权重各 0.5；原物理残差权重 0.01，每 100 步计算。图像渲染与正式评估仍用 1600×900。

优化设置 | 共同值
位置学习率 | 2e-5 → 1e-7；50k 调度
形变网格 / 网络倍率 | 25 / 5
速度网络学习率 | 1e-4
评价 / 保存 | 每 1k 测试；保留四项最佳及最终 50k 快照

本报告以最终 50k PNG 快照作主要比较。四指标最优快照按测试集选择，沿用历史流程；没有独立、未参与选择的验证集。原始 PIDG 只有 50k 训练，本轮总计约 100k；与原模型的差值还包含额外计算和 T3 损失影响。

正式 PNG 指标；每行同样覆盖固定的 50 张官方测试图


## 7. 最终重建指标：分协议解读
正式 PNG 指标；每行同样覆盖固定的 50 张官方测试图


### 原 metadata 时间协议

模型 | PSNR | SSIM | LPIPS | MS-SSIM
原 PIDG（仅 50k） | 27.2568 | 0.976551 | 0.029624 | 0.970483
T3（再训练 50k） | 28.2648 | 0.976566 | 0.027885 | 0.971197
V1 | 28.2515 | 0.976575 | 0.027881 | 0.971205
V2 | 28.2695 | 0.976564 | 0.027892 | 0.971231

V2 相对原始 PIDG：PSNR +1.0128 dB、LPIPS 降低 5.84%。但 T3 已达到 28.2648 dB；V2 相对 T3 仅 +0.0048 dB，相对 V1 为 +0.0180 dB，且 SSIM / LPIPS 略降。没有稳定的综合优势。


### 共同帧编号时间协议

模型 | PSNR | SSIM | LPIPS | MS-SSIM
V3 | 37.3459 | 0.993751 | 0.012408 | 0.995161
V4 | 38.7692 | 0.993914 | 0.011956 | 0.996027

V4 相对 V3：PSNR +1.4233 dB，SSIM +0.000163，LPIPS 降低 3.64%，MS-SSIM +0.000866。四项平均指标均改善。第 9-12 页检查这些改善是否广泛、是否能归因于一致性项。

> 同一原始权重，不训练而仅改测试时间：27.2568 → 33.2223 dB（+5.9655）。时间输入变化本身即可大幅改变评分。因此 V3/V4 与历史 27 dB 的跨协议差不能解释为结构或损失增益。

每 1k 的 PNG 等价评估；最终快照另经真实 PNG 核验


## 8. 训练曲线：收敛与收益时段
每 1k 的 PNG 等价评估；最终快照另经真实 PNG 核验

图 1. 上排为原时间协议，下排为共同帧时间；纵轴范围分别设置。虚线为第 1k 步，新增损失只会在此后启用。

V1/V2 的最终 PSNR 与 LPIPS 非常接近，SSIM / MS-SSIM 最优点落在 10k 左右；继续训练主要继续降低 LPIPS。V3/V4 的四项最优点都落在 50k，后期仍有小幅改善，说明它们的最终指标并非仅取自短训尖峰。

V4 在第 1000 步已经领先 V3：36.1283 vs 35.2620 dB；第 10k 为 37.9050 vs 36.5991 dB；第 50k 为 38.7692 vs 37.3459 dB。领先从损失启用前便已存在，这是比最终均值更重要的归因检查点。

> 这些是单场景、单 seed 的一次运行轨迹，没有重复试验置信区间。曲线平滑或长期领先，都不能单独排除执行非确定性与早期训练路径差异。

V4 相对 V3；按官方导出文件名 00000-00049 排序


## 9. 逐帧分析：均值没有说明全部
V4 相对 V3；按官方导出文件名 00000-00049 排序

图 2. 正值为 V4 的 PSNR 更高。前 8 张贡献明显，后续多数差值接近 0 或略为负值。

指标 | 改善图像数 / 50 | 逐帧差中位数
PSNR | 23 / 50 | -0.021545
SSIM | 18 / 50 | -0.000035
LPIPS | 33 / 50 | -0.000092
MS-SSIM | 28 / 50 | +0.000008

V4 仅 23/50 张图的 PSNR 优于 V3；逐帧 PSNR 差的中位数为 -0.0215 dB。前 8 张对 50 张平均差贡献 +1.4406 dB；剩下 42 张平均差为 -0.0207 dB。平均 +1.4233 dB 主要反映少数明显失败帧的恢复，而非全帧同幅改善。


### PSNR 平均值与像素 MSE 不是同一种平均

每张归一化 RGB 图有 MSE_i=10^(-PSNR_i/10)。从逐帧分数还原并平均 MSE，V4 相对 V3 下降约 51.21%；直接把平均 PSNR 差换算为误差比，得到的是几何平均意义。报告同时保留逐帧分布，避免混淆这两个统计量。

官方测试图 00004；三列使用完全相同的 GT 定义裁剪区域


## 10. 视觉证据：恢复一个漏掉的目标
官方测试图 00004；三列使用完全相同的 GT 定义裁剪区域

图 3. GT、V3、V4 的同相机、同帧时间对照。图中显示的是主体裁剪和排版缩放；原分辨率图像保存在 runs。

图 4. 对同一最上方黄色目标区域放大。V3 未显示该鸭子；V4 恢复了目标，但细节仍偏模糊。

测试帧 | V3 PSNR | V4 PSNR | 差值
00004.png | 28.9700 dB | 39.4435 dB | +10.4735 dB

该样例支持“重建结果存在有意义的局部改善”。它不证明恢复来自速度一致性项，更不能由此判断速度、应力或外力正确。第 12 页给出损失启用前的对照差异。

固定 256 个粒子、5 个训练时刻；不使用测试图 RGB 计算该诊断


## 11. 物理诊断：一致性尚未对齐
固定 256 个粒子、5 个训练时刻；不使用测试图 RGB 计算该诊断

图 5. 左：轨迹导数与预测速度幅度；右：每个时刻的相对误差。速度单位为场景距离 / 归一化时间。

最终诊断均值 | V3 | V4
速度与轨迹导数 RMSE | 1.3832 | 1.2185
轨迹导数 RMS | 1.3793 | 1.2155
预测速度 RMS | 0.0533 | 0.1102
逐时刻相对 RMSE 的均值 | 1.0057 | 0.9914

V4 的绝对误差下降 11.91%，同时轨迹导数幅度下降 11.87%。相对误差仍接近 100%，预测速度幅度明显偏低。结果与“部分通过降低轨迹导数减小损失”相容，但仅凭这些标量不能证明具体机制。

> 工程上梯度已打通，不等于优化上已学到正确物理。当前没有速度/力真值；这些只能评价自一致性，不能称作真实速度误差或材料识别精度。

新增损失之前已出现差异，必须调整下一轮对照设计


## 12. 归因审计：方法有效性尚待证明
新增损失之前已出现差异，必须调整下一轮对照设计

图 6. 左：早期 PSNR 轨迹；右：全程相对速度误差。第 1k 步对应的性能差不能来自尚未启用的一致性项。

检查点 | V3 | V4 | 差值 / 含义
初始 PSNR | 33.2223 | 33.2223 | 相同初始评分
第 1k PSNR | 35.2620 | 36.1283 | +0.8662，损失尚未启用
第 50k PSNR | 37.3459 | 38.7692 | +1.4233，归因存在混杂

代码在 iteration≤1000 时返回，首次可能施加新项是 1010；V3/V4 的运行配置除一致性权重外相同。然而相同 seed、相同初始权重和相同初始评分，不保证数值执行过程完全一致。CUDA 原子梯度等非确定性或其他执行差异是待排查因素，本次并未定位具体原因。

不能把 1.4233-0.8662 简单视为“纯约束增益”：早期参数与优化器状态的差异会非线性地影响后续训练。原协议下 V1/V2 在第 1k 也存在差异，同样需要更严格的控制。

> 可支持的结论：本次 V4 的平均重建指标最好。尚不支持的结论：该一致性项稳定造成 +1.42 dB 增益，或保证物理速度场正确。

先消除归因混杂，再优化物理耦合，最后扩大实验范围


## 13. 对后续研究与实现的启发
先消除归因混杂，再优化物理耦合，最后扩大实验范围

优先级 | 建议实验 / 修改 | 需要回答的问题
P0 | 从同一第 1k 完整 checkpoint 分叉；复制优化器、RNG、相机顺序，只改 lambda | 能否排除启用前的训练差异？
P0 | 先做无损失 / 无损失重复，再做至少 3 个 seed 的配对试验 | 无改动噪声多大？增益能否重复？
P0 | 核对模拟器/采集器真实时间戳；保留两套独立协议 | 共同帧时间假设是否正确？
P1 | 报告全图 + 前景 ROI + 分时段指标及失败帧 | 改善覆盖率如何？是否依赖少数漏物体帧？
P1 | 分支梯度范数、动态/静态采样与分阶段拟合对照 | 约束为何未对齐？是否出现运动幅度收缩？
P2 | 审计坐标变换及动量残差，再扩展物理解释 | 自一致性与守恒残差是否使用一致定义？


### 对损失与结构的具体启发

应在控制实验中比较：先冻结形变拟合速度，再以小权重联合优化；或考察带尺度记录的相对误差目标，防止绝对误差通过轨迹幅度下降而改善。相对目标的分母设计同样可能引入优化漏洞，需要梯度与退化解测试，不能直接当作已验证修复。

均匀采样可能让静态/低运动点主导。可利用已有训练前景或运动信息构建动态/静态分层采样对照，仍保留官方全图指标。这里不提出学习材料分割或增加逐 Gaussian 自由力。

本次没有改变 MLP 参数量，不能据此判断容量已足够或不足。时间语义、优化可复现性及两分支的耦合应优先审计；扩大网络前先确认当前网络确实收到并利用了正确监督。

以上均为后续建议，尚未运行。当前可在接受帧时间假设时保留 V4 用作重建展示；不将其作为物理识别成功的依据。

每组均覆盖 50 张官方测试图；以下列出最终 50k 快照


## 14. 渲染、GT 与拼接图的绝对地址
每组均覆盖 50 张官方测试图；以下列出最终 50k 快照

文件 00000.png-00049.png 按官方测试 JSON 顺序导出。网页可切换 best、metric_ssim、metric_lpips、metric_ms_ssim、50000 五个标签；相同迭代可能重复保存。每张拼接图左侧 GT、右侧 Render，原保存尺寸 3200×950（含 50 像素标题区），图像面板不缩放。


### V1_shared_time

Render / GT：

F:\A_Edinburgh\PhysGauss\runs\pidg_kinematic_tuning\V1_shared_time\rubber_duck\test\ours_50000\renders

F:\A_Edinburgh\PhysGauss\runs\pidg_kinematic_tuning\V1_shared_time\rubber_duck\test\ours_50000\gt

拼接图 / 网页：

F:\A_Edinburgh\PhysGauss\runs\pidg_kinematic_tuning\render_comparisons\V1_shared_time\pairs\50000

F:\A_Edinburgh\PhysGauss\runs\pidg_kinematic_tuning\render_comparisons\V1_shared_time\index.html


### V2_shared_time_kinematic

Render / GT：

F:\A_Edinburgh\PhysGauss\runs\pidg_kinematic_tuning\V2_shared_time_kinematic\rubber_duck\test\ours_50000\renders

F:\A_Edinburgh\PhysGauss\runs\pidg_kinematic_tuning\V2_shared_time_kinematic\rubber_duck\test\ours_50000\gt

拼接图 / 网页：

F:\A_Edinburgh\PhysGauss\runs\pidg_kinematic_tuning\render_comparisons\V2_shared_time_kinematic\pairs\50000

F:\A_Edinburgh\PhysGauss\runs\pidg_kinematic_tuning\render_comparisons\V2_shared_time_kinematic\index.html


### V3_frame_time

Render / GT：

F:\A_Edinburgh\PhysGauss\runs\pidg_kinematic_tuning\V3_frame_time\rubber_duck\test\ours_50000\renders

F:\A_Edinburgh\PhysGauss\runs\pidg_kinematic_tuning\V3_frame_time\rubber_duck\test\ours_50000\gt

拼接图 / 网页：

F:\A_Edinburgh\PhysGauss\runs\pidg_kinematic_tuning\render_comparisons\V3_frame_time\pairs\50000

F:\A_Edinburgh\PhysGauss\runs\pidg_kinematic_tuning\render_comparisons\V3_frame_time\index.html


### V4_frame_time_kinematic

Render / GT：

F:\A_Edinburgh\PhysGauss\runs\pidg_kinematic_tuning\V4_frame_time_kinematic\rubber_duck\test\ours_50000\renders

F:\A_Edinburgh\PhysGauss\runs\pidg_kinematic_tuning\V4_frame_time_kinematic\rubber_duck\test\ours_50000\gt

拼接图 / 网页：

F:\A_Edinburgh\PhysGauss\runs\pidg_kinematic_tuning\render_comparisons\V4_frame_time_kinematic\pairs\50000

F:\A_Edinburgh\PhysGauss\runs\pidg_kinematic_tuning\render_comparisons\V4_frame_time_kinematic\index.html

所有数值来自本地已完成运行；报告未触发训练或修改模型


## 15. 来源、复现与交付说明
所有数值来自本地已完成运行；报告未触发训练或修改模型

编号 | 主要证据 / 实现位置
S1 | PIDG_REPO_MAP.md；原始及独立 fork 的 network.py、renderer、motion_utils/pidg.py
S2 | variants/kinematic_tuning/{model,runtime,timeline,prepare,run}.py
S3 | experiments/kinematic_tuning/{plan,preflight,derivative_audit,source_freeze}.json
S4 | data/PIDG/rubber_duck/transforms_{train,test}.json
S5 | 各变体 rubber_duck/{results,per_view,metric_best}.json 和 metric_history.jsonl
S6 | 各变体 kinematic_diagnostics.jsonl、kinematic_loss.jsonl、metric_config.json
S7 | experiments/kinematic_tuning/assessment.{md,json}；initial_timing_assessment.json
S8 | runs/pidg_official/rubber_duck/results.json；T3_baseline_perceptual/rubber_duck/results.json


### 复现与完整来源清单

生成命令：envs/pidg-cu128/python.exe scripts/build_kinematic_report.py。构建时核验 assessment 中记录的来源 SHA-256、每组最终 50k、50 个逐帧键，以及逐帧均值与 PNG 指标的一致性。配套 .sources.json 保存完整来源哈希和图像裁剪记录；.source.md 保留正文与表格文本。

F:\A_Edinburgh\PhysGauss\output\pdf\PIDG_Kinematic_Method_Report_CN.sources.json

F:\A_Edinburgh\PhysGauss\experiments\kinematic_tuning\assessment.md


### 边界与适用范围

本报告中的“原方法”指本仓库已审计的 PIDG 实现和历史训练流程，未将未核验的论文细节当作事实。仅一场景、一个 seed、一个一致性权重；官方测试图被用于历史式快照选择，尚无独立未触碰测试集。没有真实速度、外力或材料参数标签。

> 最终建议：保留 V4 作为当前观测到的最佳重建候选；先完成共同 checkpoint 分叉、多 seed 和时间戳审计，再决定是否将速度一致性项设为默认，以及是否继续进行物理参数解释。
