# PIDG staged variants

按照 `PIDG_Codex_Upgrade_Plan.md` 第 30 节，每轮只实现下一个未勾选阶段，
通过单元测试、短训练和比较门槛后记录结果。正式五场景基线为
`runs/pidg_official`，源代码为 `third_party/PIDG`。

| 目录 | 阶段 | 内容 |
|---|---|---|
| `phase00_baseline` | Phase 0 / A0 | 官方基线独立副本、来源锁定、成对复现验证 |

用户后续已要求执行全部剩余阶段，并让每个阶段进行五场景50k长训练。
变体、队列、状态与控制方式见 [upgrade/README.md](upgrade/README.md)。

后续阶段创建自己的目录，保留之前阶段的代码和输出。不要在原基线目录中
直接实现物理升级，也不要把 CPU 参考架构的结果混入官方 CUDA benchmark。

`code/` 是实际运行源码，`source_manifest.json` 锁定来源，`reports/` 保留可提交
的审计与比较报告，`runs/<tag>/` 保存日志、模型、图片和完整断点（Git 忽略）。
