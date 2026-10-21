# Baseline PIDG: random missing training frames

Queued after the MLP capacity queue AND its rendering/metrics. Whole-frame dropout, not pixel masking.
Rubber Duck; baseline width256/depth0; 50k iterations; width1600; training seed0; mask seed0.

| Dropped | Kept / 150 | Withheld GT | Status | Iteration | Official-test PSNR | Delta vs 0% | Withheld-frame PSNR |
|---:|---:|---:|---|---:|---:|---:|---:|
| 0% | 150 | 0 | completed (reused) | 50000 | 27.256781 | 0.000000 | - |
| 10% | 135 | 15 | completed | 50000 | 28.252596 | 0.995815 | 32.994312 |
| 30% | 105 | 45 | completed | 50000 | 26.686220 | -0.570560 | 30.896891 |
| 50% | 75 | 75 | completed | 50000 | 27.043219 | -0.213562 | 30.978542 |
| 70% | 45 | 105 | completed | 50000 | 26.930286 | -0.326494 | 30.141739 |
| 90% | 15 | 135 | completed | 50000 | 23.062979 | -4.193802 | 23.520790 |

The official 50-view test set is identical at every ratio. Dropped training frames are evaluated only after fitting.
Withheld subsets are nested but not identical across ratios; use the fixed official test set for cross-ratio comparisons.
Original timestamps, poses and canonical point cloud remain fixed. Original input files are never deleted or masked in place.
Native PIDG time normalization uses the retained train_count; no physics correction is silently introduced.
0% is reused. Best snapshots use the historical test-selected protocol; single-seed results are exploratory.

[Five-second monitor](monitor.html)

Controller: completed
Dependency: All original MLP training, rendering and metrics are complete

Updated UTC: 2026-10-20T21:49:28.296096+00:00

<!-- split-evaluation -->
## 补充：保留、缺失与全量指标

上表的主 PSNR 仅覆盖官方测试 50 帧。补充评估保留相同 best 快照，不重新训练。

| 缺帧 | 保留 PSNR | 缺失 PSNR | 原训练150 PSNR | 官方测试50 PSNR | 全部200 PSNR |
|---:|---:|---:|---:|---:|---:|
| 0% | 36.167686 | N/A | 36.167686 | 27.256781 | 33.939960 |
| 10% | 38.033100 | 32.994310 | 37.529221 | 28.252597 | 35.210065 |
| 30% | 36.069752 | 30.896891 | 34.517894 | 26.686221 | 32.559976 |
| 50% | 37.392362 | 30.978540 | 34.185451 | 27.043217 | 32.399892 |
| 70% | 35.866489 | 30.141739 | 31.859164 | 26.930285 | 30.626944 |
| 90% | 36.384817 | 23.520788 | 24.807191 | 23.062979 | 24.371138 |

各项均为逐图指标的算术平均，合并按真实帧数加权；全部200含已见训练图像。
[全部四项指标、范围说明与绝对 render 地址](split_metrics.md) · [CSV](split_metrics.csv) · [JSON](split_metrics.json)

[更新后的 PDF 报告](../../output/pdf/PIDG_MLP_MaskedData_Report.pdf)
