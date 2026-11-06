# PIDG 官方代码复现

目标论文：**Physics-Informed Deformable Gaussian Splatting: Towards Unified Constitutive Laws for Time-Evolving Material Field**。

本目录建立官方实现的独立运行路径。主项目 `src/physgauss` 的 Fourier MLP / CPU 参考渲染器不参与这些实验；PIDG 实验调用 `third_party/PIDG` 的 4D hash encoder、Gaussian densification、材料场、CMR 与 Lagrangian flow matching，以及官方 CUDA rasterizer 和 evaluator。

## 固定来源

| 项目 | 来源／版本 |
| --- | --- |
| 代码 | [官方 GitHub](https://github.com/SCAILab-USTC/Physics-Informed-Deformable-Gaussian-Splatting)，`b77fd2107a50c875d7219bb4f82388d5e5ea1319` |
| 光流 rasterizer | [作者仓库](https://github.com/HaoqinHong/flow-based-diff-gaussian-rasterization)，`2146a1c0dc3558644af420f20c21a0ad07ffa67a` |
| 数据 | [官方 Hugging Face](https://huggingface.co/datasets/HaoqinHong/Physics-Informed-Deformable-Gaussian-Splatting)，`46b23520acaaf5d58938fb19b50a3a62d12c55df` |
| 对照论文 | [arXiv 完整版本及补充材料](https://arxiv.org/html/2511.06299v1) |

下载的每个文件校验 LFS SHA256 或 Git blob SHA1；各场景保留 `download_manifest.json`。输入来自官方 train/test JSON，未改成项目 toy split。图像、深度、光流和 mask 校验记录在各场景 `audit.json`。

## 环境

```powershell
conda activate F:\A_Edinburgh\PhysGauss\envs\pidg-cu128
```

当前可用环境为 Windows / Python 3.11.14 / PyTorch 2.8.0+cu128 / torchvision 0.23.0+cu128，RTX 5070 Ti 16GB，CUDA Toolkit 12.8，MSVC 14.44。已通过 `pip check`。

该 Conda 环境是现有可用 CUDA 环境的独立副本，原环境未被修改。旧 torchvision 元数据已修正，移除了副本中与本任务无关的冲突包。最终 PIDG 依赖固定在 `requirements-windows.txt`，扩展 wheel 在 `wheels/`，可重复构建脚本为 `scripts/build_extensions.ps1`。

已验证三个真实 GPU 计算路径：hash 编码输入梯度、CMR 材料场二阶参数梯度、官方带光流输出的 rasterizer 反向传播。结果见 `gpu_preflight.json`。

## 运行

在 PowerShell 中：

```powershell
Set-Location 'F:\A_Edinburgh\PhysGauss'
# 120 步，400×225，提前启用物理／flow，仅检验工程闭环。
.\reproduction\scripts\run_reproduction.ps1 -Smoke -Tag newcheck
# 官方默认设置：50k、宽度1600、seed0、CMR weight0.01、densify至15k。
.\reproduction\scripts\run_reproduction.ps1 -Scenes mechanics_cloth -Iterations 50000 -Tag experiment2
# 依次跑所有五个自建 PIDG 场景。
.\reproduction\scripts\run_reproduction.ps1 -Scenes mechanics_cloth,balls_reaction,dry_ice,motion_kuro,rubber_duck -Iterations 50000 -Tag experiment3
```

已存在的 run 不会被覆盖，重试使用新 `-Tag`。每个场景顺序完成 train → best snapshot render → 官方 metrics。原训练器的 Gaussian/deformation/material snapshots 仅供推理。现已新增完整训练断点 checkpoints/resume.pt，每 1000 步保存；包含两组优化器、相机采样顺序和随机状态，可用 -Resume 恢复。暂停时使用 scripts/save_and_stop.ps1 保存后退出以释放显存。

`-Profile paper_hparams` 只对齐部分论文超参数（seed42、CMR weight0.1、densify至30k），**不等于全部结构及数据处理均与论文对齐**。默认正式首轮是 `official`，用于先建立公开代码基线。已启动的首轮无 Tag，输出位于 `runs/pidg_official/`。

## 输出

```text
runs/pidg_official/<scene>/
├── reproduction.json        # 版本、配置、设备、状态、完成后的指标
├── cfg_args
├── train.log
├── render.log
├── metrics.log
├── point_cloud/
│   ├── iteration_best/      # 最佳 Gaussian 与材料场 snapshot
│   └── iteration_50000/
├── deform/
├── test/ours_best/           # 预测、GT、depth
├── results.json             # 官方 SSIM、PSNR、VGG-LPIPS、MS-SSIM
└── per_view.json
```

实际 deformation 目录名称以官方输出为准。各场景训练日志中 `pts`、loss、物理/flow 信息及 test PSNR 可用于跟踪。正式运行完成前不能把 smoke 指标放入论文比较。

## 对官方发布版的运行修复

修复可由 `scripts/apply_compatibility.py` 重复应用；本地 Git diff 单独保存，便于审查。

1. 官方导入的 `motion_utils/time_evolving_material_field.py` 缺失，改用同一发布版 `motion_utils/pidg.py` 中可读取的 `VelocityNet`。不能验证缺失文件是否和这个类完全相同。
2. Hash encoder 改用 MSVC 编译标志；去掉冲突的 C++14 标志；允许 CUDA 12.8 调用已安装的 MSVC 14.44。
3. Loader 在 `resized_mask` 缺失时读取已发布的 `motion_mask`／`mask`。
4. 渲染函数默认 `enable_flow_grad=False`，补齐官方评估漏传该参数的调用。
5. 修复 `load2device` 中不存在的 mask 属性及 next/prev 相机张量设备问题，以按需加载适配 16GB GPU。
6. 增加材料场及真实 particle IDs 的 snapshot 保存／加载，替代官方 render 调用的不存在的 `VelocityNet.load_weights`。

当前 `official` 的网络宽度、损失、物理残差、时间处理和采样保持该公开版本原实现。上述运行修复会完整披露。

## 严格论文复现仍需核对的差异

| 项 | 论文陈述 | 当前公开源码／数据 |
| --- | --- | --- |
| Random seed | 42 | `safe_state` 默认0；runner 显式记录 |
| CMR loss weight | 0.1 | 默认0.01 |
| Densification stop | 补充材料30k | 默认15k |
| Material field | MLP width256、index64 | 可读取的 VelocityNet width64、index16 |
| Material time | 使用时空坐标中的 t | CMR 将已经归一化的 fid 再除以训练帧数，flow velocity 使用 fid |
| CMR convective contraction | `(v·∇)v` | 源码 `einsum('bi,bij->bj',v,grad_v)`，而 grad_v 按速度分量堆叠 |
| CMR sample average | 采样残差均值 | sampled 函数在均值后再乘 `M/N` |
| Optical-flow frame pair | 对应相邻时刻 | source next_cam 是下一张训练帧；发布 flow 是相邻原始帧 |
| Static/dynamic refinement | 用 mask 分区 refinement | 训练器没有调用 render.py 内的 dynamic-mask 初始化函数 |

这些是源码／论文审计发现，不是已完成数值消融。首轮结果应称为**带运行修复的公开代码复现**，不能仅凭代码跑通就宣称严格达到论文表格。应先比较实测值，再决定哪些差异需要逐项对齐。

论文 Table 1 的 PIDG Full 对照值：

| 场景 | PSNR | SSIM | LPIPS |
| --- | ---: | ---: | ---: |
| Mechanics Cloth | 39.96 | 0.991 | 0.034 |
| Balls Reaction | 33.31 | 0.988 | 0.031 |
| Dry Ice | 26.12 | 0.926 | 0.138 |
| Motion Kuro | 26.97 | 0.953 | 0.051 |
| Rubber Duck | 28.43 | 0.976 | 0.037 |

表中是论文数值；本机正式结果由各 run 的 `results.json` 产生，两者不可混淆。


## 训练停滞恢复

控制器现在监视训练子进程的日志。连续 15 分钟无日志更新时，先验证完整断点的 SHA256 和完整性标记，再结束该控制器创建的训练子进程，从该断点恢复；每个场景最多自动重试两次。失败会保留诊断，不无限重启。数学模型、损失和采样配置不变。

用户请求保存并停止时，不重启训练、不启动下一场景。若当前迭代已卡住而无法产生新断点，控制器保留最后一个校验通过的完整断点并记录实际步数。watchdog-events.jsonl 记录重试事件。
