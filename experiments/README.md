# 新 PIDG 计划的执行与比较

本目录对应压缩包中的 `CODEX_TASK.md`，按 M0–M10 顺序开发、验证并分别提交。
源码独立放在 `variants/constitutive_pidg/code` 与 `pidg_ext`，原始 PIDG 位于 `third_party/PIDG`。
旧 phase 队列已停止并移入回收站，不使用旧队列脚本。

## 当前范围

用户确认仅对 **Rubber Duck、50,000 步、宽度 1600、seed 0** 做长训练比较。
保留原始数据划分、配置和官方评估脚本；Rubber Duck 的配置文件拼写为 `rubber_dack.py`。

| 配置 | 本轮操作 |
|---|---|
| A0 | 复用已完成的原始 50k 基线，不重复训练 |
| A1 | F 诊断，不增加训练损失 |
| A2a | 固定共享材料参数的本构一致性 |
| A2b | 学习共享材料参数 |
| A3 | 合成已知双材料验证完成，真实多材料实验等待标签 |
| A4_single | 单材料共享参数与结构化外力，50k |
| A5_single | 对 A4 的最终完整检查点做诊断，复用其训练 |
| A6_single | 单材料配置加分阶段优化和诊断，50k |

`*_single` 是可用标签范围内的配置。完整双材料、界面及外力组合另在明确标注的合成场景中验证，不能将其称为真实 Rubber Duck 多材料结果。
用户确认记录见 `user_scope.json`。D-NeRF/HyperNeRF 数据未安装，且不属于本轮单场景范围。

## 运行、监控与续训

```powershell
# 顺序运行五个新增 50k 训练，随后自动渲染、评估和汇总
.\scripts\start_constitutive_benchmark.ps1 -Mode long
# 仅运行合成场景的八个 300 步工程对照
.\scripts\start_constitutive_benchmark.ps1 -Mode smoke
# 每五秒刷新监控：http://127.0.0.1:8765/
.\scripts\start_constitutive_monitor.ps1
# 请求保存完整状态后退出，队列不会进入下一个配置
.\scripts\control_constitutive.ps1 -Stop
# 确认训练子进程与控制器退出、resume.json 校验通过后续训
.\scripts\start_constitutive_benchmark.ps1 -Mode long -Resume
```

同一时间只允许一个队列控制器。重复启动会被拒绝。已有运行默认不覆盖，显式续训要求预算、数据、配置、源码哈希一致。
训练期间不要修改 `pidg_ext`、独立源码、矩阵或当前配置。新增代码应在停止并核验完整检查点后进行。
长训练每 1000 步保存完整状态，含原始优化器、随机状态、粒子索引、新材料/外力优化器、邻域缓存与阶段状态。
监控显示的断点为保存清单；读取监控不会每五秒重新计算大型模型校验和。
停训请求会等待保存与退出，不能通过挂起或强杀代替。异常无进展恢复也只使用已校验的完整状态，且保存请求存在时不会强杀。

## 结果与解释

`results.md`、`results.csv`、`results.json` 自动更新，原始运行日志和检查点位于 `runs/pidg_constitutive`。
官方 50k 表、合成 300 步表与解析歧义试验分开报告；未完成的长训练不填最终指标。
50k 结果只有场景、步数、分辨率、seed 和协议一致时才计算相对 A0 的变化。
原始历史训练没有保存源码哈希，目前的源码校验无法倒推证明其历史来源，详见 `../baseline_commands.md`。

合成数据是人为规定变形、材料分区和载荷的工程验证，不是完整物理仿真或官方数据。
条件雅可比诊断冻结神经场，满秩也不能证明全局可辨识；M9 短测仍有约 20% 材料误差、约 100% 外力相对误差。
沿用发布版 CMR 的采样缩放、密度与时间约定；本构应力使用物理帧时间。两者的时间约定差异限制物理解释。
采用原始基于测试集选取快照的协议，因此不将未来时刻图像指标称为无泄漏预测能力。

逐阶段记录：`milestones.json` 与 `M1_validation.json` 至 `M10_validation.json`。
M10 工程闭环通过与五个 50k 训练全部完成是两个独立状态。
