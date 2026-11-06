# PIDG 复现进度与实测结果

项目：`F:\A_Edinburgh\PhysGauss`。独立 Conda 环境：`envs/pidg-cu128`。

已接入官方动态高斯、4D hash、材料场、CMR、光流 CUDA 渲染及正式 evaluator。五个场景的必需文件已下载并校验。

RTX 5070 Ti / CUDA 12.8 上的三项 GPU 梯度预检及真实 Mechanics Cloth 的 120 步训练→渲染→LPIPS 评估已通过。短训练只验证工程闭环，不代表论文复现完成。

正式首轮为公开代码设置：50,000 步、1600 像素宽度、seed0。与论文 seed42、CMR 权重等差异已在项目 `reproduction/README.md` 记录。

| 场景 | 状态 | 训练步数 | 本机 PSNR | 本机 SSIM | 本机 LPIPS | 论文 PSNR |
|---|---|---:|---:|---:|---:|---:|
| mechanics_cloth | completed | 50000/50000 | 39.47876 | 0.98534 | 0.03730 | 39.96 |
| balls_reaction | completed | 50000/50000 | 32.91008 | 0.97994 | 0.02967 | 33.31 |
| dry_ice | completed | 50000/50000 | 26.16062 | 0.90380 | 0.13741 | 26.12 |
| motion_kuro | completed | 50000/50000 | 26.91213 | 0.95177 | 0.05741 | 26.97 |
| rubber_duck | completed | 50000/50000 | 27.25678 | 0.97655 | 0.02962 | 28.43 |

只有完成 train→render→metrics 的场景才写入本机正式指标；未完成处保留为空。论文参考值来自 [Table 1](https://arxiv.org/html/2511.06299v1)，不作为本机结果。

训练日志、模型与图像保存于 `runs/pidg_official/<scene>`。实时文件 `reproduction/results_summary.json`、`results_summary.csv` 汇总状态。
