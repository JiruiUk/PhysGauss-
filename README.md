# PhysGauss

## PIDG 官方复现入口

当前优先推进 PIDG 官方实现及其五个自建场景的复现。请先阅读 [复现说明](reproduction/README.md)，官方代码、CUDA 渲染、训练和评估独立运行于 reproduction/ 流程。下文记录的是前期 PhysGauss 参考架构。

当前执行 `pidg_codex_plan.zip` 中的 [新计划](CODEX_TASK.md)，遵循 [AGENTS.md](AGENTS.md)
逐阶段验证。原始五场景50k基线已核验复用，见 [baseline_metrics.json](baseline_metrics.json)。
实际PIDG模块映射见 [PIDG_REPO_MAP.md](PIDG_REPO_MAP.md)，独立扩展源码位于
`variants/constitutive_pidg/code` 与 `pidg_ext/`，配置位于 `experiments/configs/`。
旧Phase 0–15队列已按用户要求停止并移入回收站；新计划不依赖其目录。


**Uncertainty-Aware Physics Estimation from Limited Observations with 4D Gaussian Splatting**

根据聊天「物理变形高斯详解」建立的研究项目。当前版本是**可运行、可测试的基础架构**：从 canonical 高斯出发，同时学习动态形变、速度／应力场与观测不确定性，完成训练、断点续训、测试集评估和物理状态导出。

## 两种实验设置

| 设置 | Canonical 初始化 | 视频监督 | 约束 |
| --- | --- | --- | --- |
| Public / image-only | 图像重建点云 → 可优化 3DGS | RGB、可选 flow/depth | 禁止 CT 初始化；需保留官方 split 与输入协议 |
| Trachea / CT-assisted | 配准后的 CT 表面点云 → 3DGS | RGB、可选 depth/flow | CT 仅用于 canonical 初始化；默认固定 canonical 几何 |

当前支持读取统一的 `manifest.json + frames.npz + canonical.npz`。真实 PIDG、HyperNeRF、D-NeRF 和 CT 文件的自动转换尚未实现；格式见 [数据接口](docs/data_contract.md)。不能把这两种设置的结果混成同一个公平性比较。

## 已实现

- 完整高斯参数：中心、三轴尺度、四元数旋转、RGB、opacity。
- 时间锚定的形变场：在 `canonical_time` 精确回到 canonical 状态。
- 独立 Eulerian 材料场：预测速度与对称 Cauchy 应力。
- 有界 photometric variance、RGB Gaussian NLL 与 variance prior。
- RGB、参考 SSIM、可选深度和投影光流损失。
- 自动微分 Cauchy 动量残差、3D trajectory–velocity consistency。
- 物理约束 warmup/ramp、基于 EMA 残差的自适应 collocation 采样。
- 配置、严格数据校验、训练日志、checkpoint、可复现续训、测试集评估、状态导出。

## 快速启动（Windows PowerShell）

```powershell
Set-Location 'F:\A_Edinburgh\PhysGauss'
# 当前目录若已有 .venv，可直接运行以下两条。
.\.venv\Scripts\python.exe -m pytest -q
.\scripts\run_smoke.ps1 -RunName demo
```

全新机器需要 Python 3.10+，先执行：

```powershell
.\scripts\bootstrap.ps1 -PythonExecutable 'python' -Cpu
```

若 Python 不在 PATH，`-PythonExecutable` 传入 Python 可执行文件的绝对路径。`-Cpu` 安装 CPU PyTorch；GPU 环境需按 [PyTorch 官方安装说明](https://pytorch.org/get-started/locally/)选择与 CUDA 匹配的 wheel，再安装项目。无需额外 YAML 依赖，配置采用 JSON。

## 单独执行

```powershell
.\.venv\Scripts\python.exe -m physgauss make-synthetic --output data/synthetic
.\.venv\Scripts\python.exe -m physgauss validate-data --manifest data/synthetic/manifest.json
.\.venv\Scripts\python.exe -m physgauss train --manifest data/synthetic/manifest.json --config configs/smoke.json --output runs/experiment
.\.venv\Scripts\python.exe -m physgauss evaluate --manifest data/synthetic/manifest.json --checkpoint runs/experiment/checkpoint.pt --output runs/experiment/evaluation
.\.venv\Scripts\python.exe -m physgauss export-state --manifest data/synthetic/manifest.json --checkpoint runs/experiment/checkpoint.pt --time 0.5 --output runs/experiment/state_t050.npz
```

已有 synthetic 数据时跳过 `make-synthetic`。续训保持模型／损失／物理设置一致，`--steps` 表示目标总步数：

```powershell
.\.venv\Scripts\python.exe -m physgauss train --manifest data/synthetic/manifest.json --config configs/smoke.json --output runs/experiment --resume runs/experiment/checkpoint.pt --steps 60
```

## 目录

```text
PhysGauss/
├── configs/                 # smoke / public image-only / trachea CT-assisted
├── src/physgauss/
│   ├── data/                # manifest 校验、加载、合成数据
│   ├── models/              # canonical GS、形变、材料、不确定性
│   ├── rendering/           # Renderer 协议与 PyTorch 参考实现
│   ├── physics/             # 动量、轨迹、自适应采样、warmup
│   ├── engine/              # 训练、续训、评估、状态导出
│   ├── config.py
│   ├── losses.py
│   └── cli.py
├── tests/                   # 解析物理例、渲染梯度、协议与续训测试
├── scripts/                 # Windows 环境安装与端到端 smoke
├── docs/                    # 架构、数据约定、研究路线、验证记录
├── data/                    # 本地数据，Git 忽略
└── runs/                    # 实验与模型，Git 忽略
```

## 当前边界

渲染器是有透视协方差投影和深度排序 alpha 合成的 PyTorch 小场景参考实现，复杂度 `O(NHW)`，限制 `N*H*W <= 2,000,000`。它支持反向传播，但不是正式 CUDA 3DGS 实现。模型用 Fourier features + MLP，尚未接入 4D decomposed hash grids、SH 外观或 densification/pruning。两个真实场景配置是起始模板；参数尚未通过真实实验调优，默认 CUDA 也不会自动切换渲染后端。

当前估计的是**速度与应力场**。没有接入本构关系、外部载荷、边界条件和参数 GT，因此尚不能可靠宣称估计了 Young's modulus、压力或材料常数。`density` 是给定常数；气管模板的 1000 kg/m³ 只是需要核实的初始假设。variance 是光度误差模型，也不等于校准过的物理参数后验。

合成 smoke 由同一个参考渲染器生成，只能验证工程闭环。评估会标记 `benchmark_compatible=false`：PSNR 是测试帧结果，SSIM 使用简化 3×3 窗口，LPIPS 尚未接入。正式刷点前需要完成 [后续路线](docs/roadmap.md) 中的真实数据和 evaluator 工作。

## 依据

聊天参考：`chatgpt-conversation://6acdffb0-de6c-83eb-b0dd-e2cfdba87f87`，标题「物理变形高斯详解」。聊天里的 PDF／LaTeX 附件未取得，当前实现依据可读取的对话需求。

PIDG 的动态高斯、材料场与公开数据设置已对照 [官方仓库](https://github.com/SCAILab-USTC/Physics-Informed-Deformable-Gaussian-Splatting)与[论文](https://ojs.aaai.org/index.php/AAAI/article/view/42474)核实。该项目采用独立基础实现，没有复制 PIDG 源码，也不是 PIDG 官方复现。

