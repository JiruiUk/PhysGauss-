# All remaining phases: selected-scene long-training queue

> 当前状态（2026-10-15）：用户要求暂停训练并删除所有Phase。
> 训练和监控已停止，Phase 0–15共16个目录及其中结果／断点已移入Windows回收站。
> 当前没有运行中的升级队列。以下为历史运行说明；需先恢复或重新创建变体才能运行。
> 原始PIDG代码和五场景基线结果校验未变。核验见 [deletion_receipt.json](deletion_receipt.json)。

用户已明确要求完成 Phase 1–15，并让每个 Phase 都进行长训练。
2026-10-15最新指示：只训练一个benchmark，选用当前的 **Mechanics Cloth**。
当前任务为 **15阶段×1场景＝15次50k训练**，仅与原始同场景结果比较。
当前实现独立变体、逐阶段验证和后台顺序队列。**代码实现、短测完成、长训练完成、
研究验收是不同状态**；只有真正完成的50k运行才会写入正式对比指标。

固定长训练协议：所选官方场景、每场景50,000步、宽度1600、seed0、官方split、
官方best-test-PSNR snapshot选择、原版PSNR/SSIM/VGG-LPIPS/MS-SSIM evaluator。
旧基线来自 `runs/pidg_official/`，不覆盖、不从其最终权重给新变体续训。
新变体从相同初始化协议训练，损失／网络变更记录在独立配置里。

| Phase | 独立目录 | 新增开启内容 |
|---|---|---|
| 1 | `phase01_continuous_time` | 统一连续时间，保留legacy模式 |
| 2 | `phase02_kinematic_closure` | 轨迹速度一致性；旧head保留 |
| 3 | `phase03_acceleration` | 轨迹二阶时间导数 |
| 4 | `phase04_deformation_gradient` | F/J、canonical锚定与空间频率稳定化 |
| 5 | `phase05_conservation` | rho0/J质量闭合，体积项另有独立开关 |
| 6 | `phase06_constitutive` | 线弹性→25k后Neo-Hookean |
| 7 | `phase07_material_field` | canonical材料参数场与平滑先验 |
| 8 | `phase08_closed_momentum` | 闭合强形式，正确空间散度 |
| 9 | `phase09_weak_form` | 局部Gaussian弱形式 |
| 10 | `phase10_material_moe` | 弹性／近刚性专家混合 |
| 11 | `phase11_neural_residual` | 小幅、对称、有相对上界的应力残差 |
| 12 | `phase12_contact` | 显式SDF接触，重力等力接口 |
| 13 | `phase13_uncertainty_flow` | 带mask的异方差光流NLL |
| 14 | `phase14_continuous_se3` | canonical MLP预测连续SE(3)样条控制量 |
| 15 | `phase15_symplectic_rollout` | 闭合粒子力与symplectic rollout |

每个目录的 `stage.json` 给出当前 `code_dir` 和修订版本；`config.json` 冻结
该实验的开关，`source_manifest.json` 冻结源码。修订不覆盖旧源码或旧run，
相关原因与先前manifest保存在 `reports/revision_*.json`。长训练开始后禁止修订。

## 运行与查看

每5秒刷新的独立训练监控：

```powershell
.\scripts\start_upgrade_monitor.ps1 -OpenBrowser
```

打开 <http://127.0.0.1:8765/>。页面显示当前阶段、步数、Loss、测试PSNR、
GPU显存、断点保存记录、日志与所选场景队列（当前15项）；后端每5秒直接读取训练输出，
不依赖总表的30秒刷新。实时JSON位于 `monitor/status.json`。
监控不会停止或重启训练。关闭浏览器后监控服务继续采样，重复启动会复用已有服务。
需要结束监控时可使用 `Stop-Process -Id <monitor_pid>`，PID见JSON；这只结束监控进程。
断点完整性标志来自保存清单，不会每5秒读取大型checkpoint计算校验和。

先完成单元测试和每阶段120步真实GPU训练→渲染→评估，再进入长训练。
当前共15次长训练。后台队列执行Phase1 Mechanics Cloth→Phase2 Mechanics Cloth→…→Phase15。
`benchmark_scope.json` 保存场景选择，重启默认沿用；单场景完成不代表完成全部五场景。

```powershell
# 后台启动／从已有完整断点继续；已有完成run会校验后跳过。
.\scripts\start_upgrade.ps1
# 曾明确保存停止后恢复：
.\scripts\start_upgrade.ps1 -Resume
# 显式选择场景（当前已持久化为此值）：
.\scripts\start_upgrade.ps1 -Resume -Scenes mechanics_cloth
# 刷新报告并查看当前子进程：
.\scripts\control_upgrade.ps1 -Status
# 请求完整断点保存并退出，停止后续队列：
.\scripts\control_upgrade.ps1 -Stop
```

前台选定阶段：

```powershell
.\scripts\run_upgrade.ps1 -Stages 1,2,3 -Mode all
```

`results.md`、`results.json` 仅汇总当前所选场景的长训练状态与同场景基线差值。
运行中每30秒刷新；`active.json` 标记当前run及子进程，`controller.json` 标记
队列状态。每个run都有 `train.log`、`render.log`、`metrics.log`、
`physics.jsonl`、`reproduction.json`、`checkpoints/resume.pt` 与SHA256 sidecar。
停滞15分钟时，只在校验完整断点后结束该子进程并重试，最多两次；用户的
保存停止请求不会触发自动重启。失败不伪装为完成，也不无限重启。

当前单场景15阶段粗估2–3天，保守预留3–4天；尚无后期阶段完整耗时，
这不是完成承诺。新增高阶物理计算、暂停和设备状态会影响时长。
队列需要机器保持运行，不能在电脑关机时继续；重启后从磁盘断点恢复。

数学实现、稳定性修订、独立开关、物理单位和当前边界见
[physics_design.md](../../docs/physics_design.md)。历史baseline没有逐文件源码
运行哈希，因此不会追溯宣称已证明它的运行时来源；新run开始记录这些信息。

速度head移除的10%门槛、弱形式梯度方差／质量提升、材料识别和外推精度属于
实测验收内容。长训练完成后仍需依据结果判断，不能由“代码能运行”直接勾选通过。
