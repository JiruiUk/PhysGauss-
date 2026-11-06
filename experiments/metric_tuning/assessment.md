# 已完成训练的效果核查

六组筛选已完成，四组长训练均完成额外50k步；完整检查点的步数和SHA-256已核对。
T3的两个导出快照均有完整50张render/GT，GT与原baseline逐像素一致。以下使用原评估器的PNG结果。

| T3快照 | 对照 | 指标 | 对照值 | T3值 | 差值 | 改善帧数/50 |
|---|---|---|---:|---:|---:|---:|
| ours_metric_lpips | W256_D0 | PSNR | 27.25678062 | 28.26476860 | +1.00798798 | 27 |
| ours_metric_lpips | W256_D0 | SSIM | 0.97655141 | 0.97656614 | +0.00001472 | 19 |
| ours_metric_lpips | W256_D0 | LPIPS | 0.02962352 | 0.02788457 | -0.00173895 | 50 |
| ours_metric_lpips | W256_D0 | MS-SSIM | 0.97048265 | 0.97119719 | +0.00071454 | 27 |
| ours_metric_lpips | W512_D0 | PSNR | 28.57652855 | 28.26476860 | -0.31175995 | 25 |
| ours_metric_lpips | W512_D0 | SSIM | 0.97704917 | 0.97656614 | -0.00048304 | 20 |
| ours_metric_lpips | W512_D0 | LPIPS | 0.03461320 | 0.02788457 | -0.00672863 | 50 |
| ours_metric_lpips | W512_D0 | MS-SSIM | 0.97046602 | 0.97119719 | +0.00073117 | 28 |
| ours_metric_lpips | W512_D2 | PSNR | 27.61541939 | 28.26476860 | +0.64934921 | 21 |
| ours_metric_lpips | W512_D2 | SSIM | 0.97678393 | 0.97656614 | -0.00021780 | 19 |
| ours_metric_lpips | W512_D2 | LPIPS | 0.03329553 | 0.02788457 | -0.00541096 | 48 |
| ours_metric_lpips | W512_D2 | MS-SSIM | 0.97107154 | 0.97119719 | +0.00012565 | 24 |
| ours_metric_ms_ssim | W256_D0 | PSNR | 27.25678062 | 28.17252350 | +0.91574287 | 34 |
| ours_metric_ms_ssim | W256_D0 | SSIM | 0.97655141 | 0.97671539 | +0.00016397 | 30 |
| ours_metric_ms_ssim | W256_D0 | LPIPS | 0.02962352 | 0.02855911 | -0.00106441 | 44 |
| ours_metric_ms_ssim | W256_D0 | MS-SSIM | 0.97048265 | 0.97158617 | +0.00110352 | 33 |
| ours_metric_ms_ssim | W512_D0 | PSNR | 28.57652855 | 28.17252350 | -0.40400505 | 26 |
| ours_metric_ms_ssim | W512_D0 | SSIM | 0.97704917 | 0.97671539 | -0.00033379 | 20 |
| ours_metric_ms_ssim | W512_D0 | LPIPS | 0.03461320 | 0.02855911 | -0.00605409 | 48 |
| ours_metric_ms_ssim | W512_D0 | MS-SSIM | 0.97046602 | 0.97158617 | +0.00112015 | 28 |
| ours_metric_ms_ssim | W512_D2 | PSNR | 27.61541939 | 28.17252350 | +0.55710411 | 21 |
| ours_metric_ms_ssim | W512_D2 | SSIM | 0.97678393 | 0.97671539 | -0.00006855 | 22 |
| ours_metric_ms_ssim | W512_D2 | LPIPS | 0.03329553 | 0.02855911 | -0.00473642 | 47 |
| ours_metric_ms_ssim | W512_D2 | MS-SSIM | 0.97107154 | 0.97158617 | +0.00051463 | 24 |

结论：T3的整体微调方案在本次Rubber Duck官方50张测试图上有效；相对原W256_D0，其LPIPS最优快照四项平均指标均改善，其中SSIM增幅很小。
相对此前所有变体的最佳值，T3刷新LPIPS和MS-SSIM；PSNR和SSIM仍由W512_D0保持。
T3不增加MLP参数，采用W256_D0、L1+SSIM+训练图LPIPS、保守学习率及关闭增密。
不能将T3全部收益单独归因于LPIPS：尚缺相同W256_D0父模型、相同额外步数和学习率但不加LPIPS的续训对照。
残差分支/MSE组合没有刷新全局最佳，且改变多个变量；不能据此判定残差分支本身无效。T1与T5仅完成1000步筛选，负面结论更有限。
证据范围为一个场景、一个种子，且使用测试指标选快照；逐帧胜率为描述统计，不代表独立重复实验或跨场景保证。

[完整结构化核查](assessment.json) · [GT/render对照](../../runs/pidg_metric_tuning/render_comparisons/index.html)
