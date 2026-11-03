# R0 static-pool evaluation — reviewed, not promoted

Two arms from identical full15k state to final50k;35000 actual updates each; no new transfer, shape, contact or constitutive loss.
The treatment includes a fixed rendering map for the accepted static rows and exclusion of those rows from dynamic CMR.
Historical S0 is a contextual reference, not the matched control.

| Model | Validation PSNR | Test PSNR | Test SSIM | Standard VGG LPIPS |
|---|---:|---:|---:|---:|
| H0_dynamic_control | 40.455640 | 40.846448 | 0.99494032 | 0.01293232 |
| H1_static_pool | 40.437735 | 40.797551 | 0.99483085 | 0.01354674 |

Predeclared image guardrails: {'validation_PSNR': True, 'validation_standard_LPIPS': True}. Passing them alone is not a successful geometry result.

| Diagnostic | H0 | H1 |
|---|---:|---:|
| max_center_time_difference_RMS | 0.32738113 | 0 |
| max_covariance_time_difference_RMS | 0.001602969 | 0 |
| static_image_track_reprojection_mean_px_400 | 1.8429509 | 7.5641654 |
| triangulation_held_observation_mean_px_400 | 1.8513734 | 7.4827476 |
| train selected-static blue contribution share | 0.516303 | 0.121142 |
| val selected-static blue contribution share | 0.515990 | 0.121233 |

Static image tracks come from original training images; their odd observations were held only from triangulation, not from model RGB training.
Color subsets and fixed-view morphology are proxies. Novel camera/time image GT and complete physical IDs are unavailable.
## 原生图像复核与决定（2026-11-03）

**R0 完成，但 H1 不晋级。下一步优先审计静态选择和参考几何。**

验证集 PSNR 变化为 -0.017905 dB，VGG LPIPS 恶化约 4.30%，均在预先设定的
门槛内；测试集 PSNR 变化为 -0.048897 dB，LPIPS 恶化约 4.75%。画质门槛通过，
但没有观察到要求的结构改善。

已实际查看固定相机 86、188 和原相机轨迹的 t=2/86/200 原生裁剪，使用相同
坐标且不重采样，见下方对照。固定 86 的早、晚时刻，两组均出现蓝色结构拉丝、
断裂以及黄色物体碎片；中间 t=86 的原观测组合较好。固定 188 的早、中时刻
同样严重碎裂，接近自身原观测时刻的 t=200 较完整。H1 未修复该模式；原轨迹
两组外观均较好。这是所查图像的定性观察，不是 3D 完整度或物理身份真值。

更关键的是，被固定子集的蓝色可见贡献占比在验证集从 51.60% 降至 12.12%；
其 canonical opacity 中位数从 0.999996 降至 0.003908，低于 0.01 的比例从
2.65% 升至 68.18%。原始训练图像轨迹的重投影均值也从 1.84 升至 7.56 px
（400 像素宽坐标系）。这支持“优化降低固定子集的不透明度，而其余动态/未知
行承担大部分图像拟合”的解释。它尚不能区分标签错误、参考几何不一致、
canonical 参数漂移或观测不足等原因。canonical opacity 不是经过遮挡合成后的
可见贡献，两者在此分别测量。

静态时变误差为零是实现的结构性质；当大部分静态点透明化时，它尤其不能证明
恢复了正确静态几何。逐点、不同参考时间的 warp 固定在同一子集中，是否构成
一致表面，需要单独审计。不得据此否定所有静态表示，也不得把颜色区域直接
视为物体或材料身份。

下一阶段限定为训练集上的静态选择/参考几何审计：定位被抑制行与参考帧，
比较初始化与终态的重投影、尺度、位置及可见贡献，并检查能否用一致参考几何
解释多时刻观测。先保存诊断和不确定性，再制定可审查的短实验；暂不启动另一轮
完整训练、R1 物体表示、权重扫描、本构或接触/未知力模块。

独立复核通过：159 项冻结源码/输入哈希、两份完整 50k 检查点、35000 条完全
匹配且连续的相机采样、160 张验证/测试观测的逐帧指标均值和 5084 项导出文件
哈希。16 段视频均经完整解码，各 200 帧。见 `final_audit.json`、
`review_decision.json` 和 `export_verification.json`。原先数值报告中的
“pending manual review”是流水线完成时的历史状态，本节与决定文件记录最终复核。

- [固定 86 原生裁剪](../../output/static_pool/review/fixed_0086_native_crops.png)
- [固定 188 原生裁剪](../../output/static_pool/review/fixed_0188_native_crops.png)
- [原相机轨迹原生裁剪](../../output/static_pool/review/moving_control_native_crops.png)

[Video and native image gallery](../../output/static_pool/index.html)
