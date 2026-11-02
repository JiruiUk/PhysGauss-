# Codex 工作交接：运动光流与轨迹导出速度对照

更新时间：2026-11-02 02:57 UTC（Europe/London）。
工作目录：`F:\A_Edinburgh\PhysGauss`；Windows / PowerShell。

> 给接手 Codex：先读本文件和根目录 `AGENTS.md`，检查队列是否仍在推进。训练已经启动，禁止重复启动第二个协调器。继续完成两组训练、渲染、运动诊断、结果解释和验证；不要从头重做已有工作。

## 1. 用户目标与授权

用户最新研究要求：先加入运动光流训练一组，再以“速度从形变轨迹导出”的结构做另一组比较。此前还要求完整训练后输出 Ground Truth 与 Render 的拼接结果。

用户现在要求生成本交接文件，方便另一位 Codex 接手。**没有要求停止训练**；原有队列继续运行。两组完整训练、评估、导出均已授权，无需再次询问是否继续。不要自行加入外力、本构或其他未知量联合训练。

用户中文沟通。不要把“已启动”写成“已完成”，不要用内部自一致性代替物理真值验证。未获授权不要发邮件/Slack 消息；不要主动创建新聊天或子代理。

## 2. 接手时的运行状态

**04:25 UTC 最新状态：K1/K2 全部训练、导出和三阶段后处理完成。测试 PSNR 38.301304 / 34.206497。进一步发现原辅助相机 FoV 与真实加载器不符，以及 CUDA mean-minus-pixel 偏移符号、metadata 写入后 T 权重约定有误，均用实际源码及 CUDA 单 Gaussian 测试核验。原训练源码/模型/缓存/V1诊断保持冻结，不能把本轮退化唯一归于结构。新增 `evaluate_motion_v2.py`、`gradient_balance_v2.py`、`summarize_results.py` 已实际执行；`motion_validation_v2.json`、`gradient_balance_v2.json`、`diagnostic_audit_v2.json` 和 assessment.md 为当前解释依据，旧 motion_validation.json / gradient_balance.json / motion_comparison.png 已被 V2 替代。真实相机下 K1/K2 覆盖率99%以上；速度推进 EPE 1.193206 / 0.846755 px，但几何 EPE 0.783709 / 0.846833，K2 仍不满足综合门槛。5项新增测试（含真实 loader CameraInfo 和 CUDA offset/权重）通过，日志 output/flow_comparison/geometry_v2_tests.log。已人工看原帧85拼接、解码视频0/21/49图及训练/运动图。下一轮应使用统一修正流程的分离应力对照 L0 和共享应力候选 L1，以隔离流程修复与共享分支。几何修正与相机契约代码已在 `variants/unified_flow/{flow,camera_contract}.py`；尚未整合完整训练器/启动下一轮。下一步继续实施下述候选，不能重复跑旧后处理。**

**04:08 UTC 接续记录：两组均已完成 50k，验证选择锁已写入。K1 验证 PSNR 38.026127，K2 33.848821；尚不能用内部零速度偏差替代独立运动诊断。原队列正在顺序渲染/导出。已启动唯一后处理等待器 `output/flow_comparison/postprocess_queue.py`（工具会话 87816）：等待 `progress.json` 为 completed 后，顺序执行 evaluate_motion、gradient_balance、finalize。先检查 `experiments/flow_comparison/postprocess_status.json` 和各阶段同名 `.log`；等待器/后处理仍活跃时禁止重复运行 GPU 诊断。若 failed，先修复该阶段并接续，不能冒称已通过。完成之后仍需人工查看输出、补充报告、结果提交和按用户条件做下一轮结构实验。**

最近核验：2026-11-02 02:57 UTC。用户已授权在比较后继续改进，详见 `experiments/flow_comparison/followup_plan.md`。

| 项目 | 状态 |
|---|---|
| K1_flow_independent | 已完成 50,000 / 50,000 步 |
| K1 最近/最佳验证 | 第 50,000 步，**验证 PSNR 38.026127 dB**，LPIPS 0.01371659 |
| K1 内部速度相对误差 | 五个训练时间探针平均约 **99.80%**，不是速度真值误差 |
| K2_flow_trajectory | 正在训练，约 4,100 / 50,000 步；4k 验证 PSNR 24.92583 dB |
| 整个队列 | `training`；最近检查没有数值错误 |
| 数据/单元/GPU/短训预检 | 全部通过，见第 5 节 |

上述进度会过时，以以下文件为准：

- `experiments/flow_comparison/progress.json`
- `experiments/flow_comparison/monitor.html`
- `experiments/flow_comparison/training_coordinator.log`
- `runs/pidg_flow_comparison/<模型>/rubber_duck/train.log`

**检查进度（不占用 GPU）：**

```powershell
Set-Location F:\A_Edinburgh\PhysGauss
envs/pidg-cu128/python.exe output/flow_comparison/status.py
Get-Content experiments/flow_comparison/training_coordinator.log -Tail 20
```

旧协调器会话 **14891 已退出**，最后未保存日志到 41,760 步。接手时已确认没有 Python 进程，核验 41,000 步完整检查点，归档并移出超过该边界的 7 条/文件未保存诊断记录，再于约 02:40 UTC 恢复。新协调器工具会话 **76990**，日志 `experiments/flow_comparison/resume_coordinator_20261102.log`，参数 `--hours 12`。K1 随后完成，K2 已启动。不要假设另一聊天可以访问该会话，优先根据日志时间戳和进程判断。

恢复凭据：`experiments/flow_comparison/resume_reconciliation.json`；完整原日志归档 `output/flow_comparison/resume_archives/20261102_024041/`。未改权重、配置或冻结源码；动量 KDTree 缓存未序列化、恢复时重建，因此不声称与不中断运行逐位一致。若再次中断，按新的检查点边界重新归档日志；`reconcile_resume.py` 是本次 K1/41k 的一次性脚本，不可原样重复用于 K2。

已建立当前聊天的自动接续检查（automation ID `automation`，每 10 分钟）。训练正常推进时保持安静；结束后继续诊断和用户授权的条件式下一轮。不要重复创建跟进任务，也不要重复启动协调器。此次接续记录/辅助脚本提交见 `git log`。

**只有确认旧协调器和训练进程已经退出后，才执行恢复命令：**

```powershell
Set-Location F:\A_Edinburgh\PhysGauss
. ./reproduction/scripts/enable_cuda.ps1
envs/pidg-cu128/python.exe -u -m variants.flow_comparison.run --hours 12
```

该命令会核验冻结源码、复用已通过预检、恢复已有完整检查点，并跳过已完成工作。不要直接裸跑上游 `train_pidg.py`，否则会遗漏变体配置/完整恢复约定。不要只因短时间没有新验证结果就判定进程已停：验证和写检查点会暂时停留在整千步。

## 3. 已固定的实验设计

完整协议：`experiments/flow_comparison/plan.json`。

- 场景 Rubber Duck；seed0；**120 train / 30 validation / 50 test**。
- 两组复用**同一个完整 3,000 步检查点**，含模型、优化器、相机栈与 RNG，继续到**总计 50,000 步**，不是额外再训 50,000 步。
- 起点：`runs/pidg_color_repair/shared_prefix/rubber_duck/checkpoints/resume.pt`。
- 起点 SHA256：`6da523ea247b4068ec9bd6a7c8f83dacb7506c698daaeecc9eb0ecd93ea1ebce`；已核对该目录 `resume.json`。
- K1：`velocity_source="independent"`，独立速度头 + 光流。
- K2：`velocity_source="trajectory"`，完整渲染中心轨迹的有限差分速度 + 相同光流。保留参数对象、state_dict 键和优化器组，冻结未使用的独立速度头；**应力头继续训练**。
- 两组配置只应相差 `velocity_source`；相同固定单调物理时钟 `tanh(2*t-1)`，相同图像、覆盖、邻边应变损失、相同光流和局部动量算子。
- 两组都不使用额外软运动学一致性损失：`kinematic_weight=0`。
- 光流权重 .01，每 100 步，3k 后启用、2k 渐增；RGB 1600×900，光流与对应关系 400×225。
- 动量权重 .01，每 100 步，8k 后启用、4k 渐增；64 anchors，12 neighbors，Huber delta=1，rho=1。
- 速度有限差分 h=.001；加速度 h=1/199；边界使用二阶单侧差分。
- 两组全部训练结束后，先锁定验证集 PSNR 最佳快照，再打开最终测试渲染。导出 `best` 和 `50000` 两种快照。

**关键解释限制：**为支持轨迹速度且避免 CUDA hash encoder 不支持的坐标 Hessian，两组共用了新的局部 Lagrangian 动量离散化。相对于历史 H3/J1，还改变了光流、时钟或动量算子/分阶段设置；**不能把历史模型到 K1 的差异全部归因于光流**。这是已向用户说明的设计。K1/K2 之间的主要受控差异是速度来源。

## 4. 物理与代码含义

共同帧时间轴已修复为 `t=(frame_id-1)/199`。当前问题不是再次发生 train/test 错帧，而是两个独立函数在同一 X、同一 t 给出的 `v_pred` 与 `∂t x` 不一致。

- 形变场：四个 3D hash grid（XYZ/XYT/YZT/XZT）与形变网络。
- 原物理场：另一套六个 2D planes + ID embedding + MLP，输出 velocity3 与 stress6；**不是两个分支共用同一套 grid**。
- 必须使用完整渲染中心 `x(X,t)=R(q(X,t))X+b(X,t)`，不能只对平移 b 求导。
- K2 速度从该中心轨迹得到。其内部速度误差为零是**结构定义**，不是物理准确性证据。
- 新局部动量项在固定材料标签 X 上求加速度，在形变后世界位置 x 上用局部加权拟合求速度梯度/应力散度；没有额外 Eulerian 对流项。
- 保留 PIDG 的预测应力和原速度梯度偏应力修正；没有加入 Neo-Hookean、外力、自由刚度或材料分割。
- 时间单位仍为归一化场景时间，不能直接称米/秒。
- 原训练器控制台 K2 的 `velocity mean` 仍打印未使用的旧速度头均值；以带来源标签的运动学/光流/动量日志为准。

主代码：`variants/flow_comparison/`。

| 文件 | 作用 |
|---|---|
| `run.py` | 顺序训练 K1→K2、完整恢复、选择锁定、测试渲染、导出 |
| `prepare.py`, `code/` | 独立冻结 fork；原 PIDG 与旧变体未改 |
| `model.py` | 时钟/速度来源开关，冻结闲置参数，保留组布局 |
| `temporal.py` | 完整中心轨迹、检查点化查询、速度/加速度差分 |
| `local_momentum.py` | 当前坐标局部梯度、应力与材料坐标动量残差 |
| `flow.py` | 仿射 Gaussian 光流、Torch 投影/协方差梯度 |
| `geometry.py` | 相机补偿、FB confidence、正确的 H,W,2 光流损失 |
| `runtime.py` | 损失接入、梯度审计、来源标签、诊断 |
| `checkpoint.py` | 仅允许有凭据的 use_flow False→True 与诊断目录迁移 |
| `prepare_data.py` | 训练专用光流/深度缓存；已生成并锁定，勿再编辑 |
| `export.py` | 拼接 PNG、MP4、ZIP、独立 PSNR 与像素核验 |
| `README.md` | 更完整的实现与限制说明 |

原发布光流代码发现 H,W,2 按行归一化/截断、协方差语义以及投影 metadata 梯度像素缩放等问题。新分支修正分量归一化、不裁剪位移，使用稳定 SPD 平方根及经过 CUDA forward 对齐和 gradcheck 的 Torch 投影。**不能称其为逐字复现论文 LPFM**。

## 5. 已完成的证据与提交

实现/协议/预检里程碑已提交：**`c73a97a`**，提交信息 `Add paired training-flow and structural velocity comparison with preflight`。

- `tests/test_flow_comparison_data.py`、`tests/test_flow_comparison_model.py` 与继承的时钟测试合计 **23 passed**（含显式 GPU 检查）。
- 两组各 100 步 GPU 短训，并从 70 步完整恢复到 100 步。
- 20 步 flags-off 短训，保留旧代码路径。
- 光流和动量损失到形变/应力/独立速度头的有限非零梯度检查通过；K2 闲置速度头无梯度。
- 原训练器最终步不执行 optimizer.step，因此另做了真正的优化器更新断言：两组 grid/deformation/stress 均改变，K1 velocity_head 改变、K2 velocity_head 完全不变。检查点未覆盖。

证据：

- `experiments/flow_comparison/preflight.json`
- `experiments/flow_comparison/tests.log`
- `experiments/flow_comparison/optimizer_update_checks.json`
- `output/flow_comparison/verify_optimizer_update.py`
- `experiments/flow_comparison/source_freeze.json`
- `variants/flow_comparison/source_manifest.json`

## 6. 光流数据与冻结约束

训练缓存：`data/derived/flow_comparison_train_only/`，120 张训练图、119 个真实相邻训练帧对；不能复用原数据旧 .flo，因为重划分后的帧对不同。

- UniMatch code commit：`943a99006950ab6c84ea72a72beb147fdd03c19a`。
- 权重：`reproduction/flow-assets/gmflow-scale2-mixdata-train320x576-9ff1c094.pth`。
- 权重 SHA256：`9ff1c094f496f5a209e910b2c84071e712c1f2a1f391f4a4546f1a467328876e`。
- 完整锁：`experiments/flow_comparison/flow_model_lock.json`。
- UniMatch 估计分辨率 800×448，结果缩放到 400×225。
- 相机补偿深度来自仅用训练图像拟合的冻结 H3 重建；**深度和光流均为估计，不是 GT**。
- 只读取 120 张训练 RGB 生成训练监督，没有 val/test RGB 混入；训练帧间隔最大 3 帧，使用真实 dt。
- 每对约 2,010–4,600 个可信像素，中位约 3,272；正反向一致性/不透明前景/深度/边界筛选。
- 目标文件、120 张输入图、H3 教师权重、生成代码等均有 SHA 锁，见缓存 `input_lock.json`、`manifest.json` 与 `experiments/flow_comparison/flow_data_manifest.json`。
- 输入可视检查：`output/flow_comparison/training_flow_inputs.png`。

**不要编辑已经冻结的 `variants/flow_comparison/*.py`、fork、plan 或父级助手文件。**队列在下一阶段/恢复时会检查哈希，修改会使其拒绝继续。发现实质 bug 时先保存状态、归档旧实验和锁文件，写清版本变化再修复；不要静默重写锁文件。尤其不要改 `geometry.py` / `prepare_data.py`，缓存也锁了其代码 SHA。

## 7. 队列会自动完成什么

1. K1 从共同 3k 完整训练到总 50k。
2. K2 从同一个 3k 完整状态独立训练到总 50k。
3. 写 `runs/pidg_flow_comparison/selection_lock.json`，两组按验证 PSNR 选择。
4. 对两组 `best` / `50000` 渲染 50 个最终测试视角并计算四项指标。
5. 每组导出 100 张 GT|Render 拼接 PNG、两段 50 帧视频（10fps 仅展示）、交互 HTML 和 ZIP。
6. 验证拼接左右图像区逐像素一致、独立重算 PSNR、视频帧数/时长、ZIP CRC。
7. 生成 `experiments/flow_comparison/results.json` / `assessment.md`，最终进度变 `completed`。

预计导出路径：

```text
output/flow_comparison/K1_flow_independent/
output/flow_comparison/K2_flow_trajectory/
  GT_Render_best.mp4
  GT_Render_50000.mp4
  pairs/best/00000.png ... 00049.png
  pairs/50000/00000.png ... 00049.png
  index.html
  manifest.json
output/flow_comparison/<模型>_GT_Render.zip
```

每张 PNG 为 3200×950：50px 标题 + 两张 1600×900 图，左 GT、右 Render。原帧 85 对应测试索引 21，即 `pairs/best/00021.png`。

## 8. 训练结束后还必须手工继续的工作

新增辅助脚本与本交接一并保存为独立提交，位于 `output/`，不会改变冻结训练代码。`status.py` 已实际使用；下面三项最终后处理只做过语法检查，**尚未执行最终模型诊断，不能宣称已通过**。

```powershell
# 仅在训练/渲染队列 completed 且 GPU 已释放后顺序运行。
. ./reproduction/scripts/enable_cuda.ps1
envs/pidg-cu128/python.exe -u output/flow_comparison/evaluate_motion.py
envs/pidg-cu128/python.exe -u output/flow_comparison/gradient_balance.py
envs/pidg-cu128/python.exe -u output/flow_comparison/finalize.py
```

### evaluate_motion.py

- 在选择锁定后，用 30 个“验证源帧→下一训练帧”新配对生成原始 UniMatch 光流作诊断，**不反馈训练/选择**。
- 直接比较 source→target 相机下的原始二维光流，**不使用 H3 估计深度补偿**。
- 使用固定观测 confidence mask；模型漏覆盖不能通过删除像素降低误差。
- 比较 geometric、velocity advection、仅相机运动/物体静止 baseline 的 pixel EPE 与 normalized L1。
- 对两组 best/final 共四快照评估；并测量速度/加速度差分步长减半/加倍的敏感性。
- 缓存：`data/derived/flow_comparison_validation_eval`；输出 `experiments/flow_comparison/motion_validation.json`。
- **重要加载修正已写入但待执行核验**：`load_ply()` 不恢复速度头/ID；本脚本显式加载 `material.pt` 中 `velocity_net`、`particle_indices`、`force_sigma`。不可删掉。
- 仍是估计的二维运动观测，不是三维速度 GT。

### gradient_balance.py

- 对验证选中的 best，在固定训练源帧 2/83/155 做 .01 光流/.01 动量加权梯度范数与 cosine 探针。
- 测 deformation、velocity_head、stress_head、physics_shared、rate_coefficient；不执行优化器更新。
- 追加日志重定向到 `output/flow_comparison/gradient_probes/`，避免混入真实训练日志。
- 输出 `experiments/flow_comparison/gradient_balance.json`。
- 只是局部固定探针，不是全部随机训练批次的审计；不能单凭范数认定唯一因果。

### finalize.py

- 核验源码/数据/共同起点哈希、两组仅 velocity_source 不同、最终 50k 完整检查点、闲置时钟/速度头未改变、参数有限、应力头已更新、训练光流配对合法、导出核验。
- 预期每组动量损失计算 420 次，其中原训练器最终步跳过优化器更新，实际参与更新为 419 次；还断言两组训练光流帧调度相同。若断言失败，先查日志与 RNG/调用路径，不能为了通过而删断言。
- 画 `training_comparison.png`、`motion_comparison.png`；输出 `experiments/flow_comparison/final_verification.json`。
- 可能还需根据实际日志结构修正辅助脚本。例如历史诊断文件是否存在；必须据证据处理，不能伪造通过。

### 最后交付

1. 看实际 best/final GT|Render PNG，至少检查早/中/晚时间与原帧 85；查看图表和视频。
2. 在 `assessment.md` 加入运动诊断、差分敏感性和梯度探针的真实结果与限制。
3. 解释图像 PSNR、观测二维运动、内部速度一致性三者是否同时改善。
4. 如结果差，列出有证据的原因与后续可区分原因的实验；不要直接断言一定是缺外力，也不要立刻加所有未知量联合训练。
5. 单独提交结果里程碑，显式 stage 本任务报告、辅助脚本/图表/验证收据；不要提交巨量 checkpoint/视频/PNG，也不要 stage 无关脏文件。
6. 中文给用户简短结果表、报告链接、两组 best 拼接视频/浏览页链接；明确还有哪些限制。

用户追加授权：根据比较继续一个受控改进。若综合效果良好，沿当前技术路径处理一个有证据的瓶颈；若不行，尝试形变和应力共享 Grid4D/latent、速度由完整轨迹导出的同一分支。事先记录的门槛、结构含义和测试要求见 `experiments/flow_comparison/followup_plan.md`。不要把本次两组交付当作取消后续授权；下一组必须独立变体、测试、短训、完整训练和 GT|Render 比较，保留 K1/K2。

## 9. 历史结果与研究背景（勿重跑）

- 历史 H3：`runs/pidg_integrity_tuning/H3_combined/rubber_duck`。
  最终 **测试** PSNR 40.878231、LPIPS .009441059；不是当前 K1 的验证 PSNR，不能直接拿混合 split 数字比较。
- J1：`runs/pidg_monotonic_training/J1_monotonic_kinematic/rubber_duck`。
  单调时钟 + .001 软一致性，已完整 50k；验证选中 49k，测试 best PSNR 40.080849，final 40.071724；最终内部相对速度误差约 99.73%。
  报告 `experiments/monotonic_training/assessment.md`，GT/Render 在 `output/monotonic_training/J1_monotonic_kinematic/`。
- 冻结轨迹拟合实验已完成，见 `experiments/frozen_velocity_probe` 与 `variants/frozen_velocity_probe`。
  原物理网络内部可学习标量时钟会折叠；固定单调时钟后全目标训练相对误差约 20.26%、held-time 21.97%，小任务可到 2.16%。不能说表示完全不能拟合。
- 冻结拟合约 49,152,000 采样对；J1 联训约 1,202,944，监督预算差约 40.86 倍，且目标移动、Huber/CMR 等不同；不能把拟合失败唯一归于结构。
- 最终测试在历史实验中已有调参暴露。本轮按验证选择仍不能称它为“全新未暴露测试集”。
- 原论文 LPFM 与本地 Rubber Duck 发布配置不同：本地该场景 `use_flow=False`；当前补光流是新的实验，不冒称之前已经使用完整论文监督。

## 10. 环境与工作区安全

- Python：`envs/pidg-cu128/python.exe`；Torch 2.8 + cu128。
- GPU/C++ 环境：先 dot-source `reproduction/scripts/enable_cuda.ps1`。
- hash source/build 由启动器指向 `third_party/PIDG/hashencoder` / `third_party/PIDG/tmp_build`，已有编译产物；不要随意重建环境。
- 启动器：`scripts/pidg_variant_entrypoint.py`；训练全状态：`reproduction/scripts/training_checkpoint.py`。
- 基线代码概念映射：`PIDG_REPO_MAP.md`；根 `AGENTS.md` 的模块开关、保留 baseline、测试与分里程碑提交规则继续适用。
- Git 当前 `main`，本任务实现提交 `c73a97a`；本交接和后处理准备为之后的独立提交，见最新 `git log`。仓库原有大量无关 untracked 文件和三个原有 tracked 改动：`experiments/integrity_feedback/proposals.json`、`experiments/results.json`、`experiments/results.md`。**不要覆盖、reset、clean 或一并提交。**
- 当前未提交的本任务运行文件主要为进度、监控页和协调器/数据生成日志。不是丢失的工作，不要删除。
- 沙箱 `.git` 写入可能需 `require_escalated` 自动审核；此前 scoped git add/commit 均通过。无需向用户重复索取已授权的常规提交确认。
- Windows 文件移动/删除使用单一 PowerShell 流程与 `-LiteralPath`，先验证目标在工作区；禁止宽泛递归删除。
- 网络资源已获取，不需要再次下载模型，更不能上传数据。不要输出 SSH 私钥/密码。

## 11. 可直接粘贴给下一位 Codex 的提示

> 请读取 `F:\A_Edinburgh\PhysGauss\CODEX_HANDOFF_FLOW_COMPARISON_20261102.md` 和根 `AGENTS.md`，先检查现有 K1/K2 光流对照队列进度，勿重复启动。接续已经授权的两组完整训练、选择锁定、GT/Render 导出，然后顺序执行并核验尚未运行的运动诊断/梯度平衡/最终验证脚本，解释两组结果并提交独立结果里程碑。保留原 PIDG、历史模型、冻结协议和无关改动。若队列仍活跃，继续监控，不要同时启动 GPU 后处理。
