# Rubber Duck GT / Render 对照

[打开可播放总览](F:/A_Edinburgh/PhysGauss/runs/pidg_constitutive/render_comparisons/rubber_duck_best_20261019_001152/index.html)

[五模型同步视频](F:/A_Edinburgh/PhysGauss/runs/pidg_constitutive/render_comparisons/rubber_duck_best_20261019_001152/gt_and_five_models.mp4)

五个模型均重新渲染全部 50 个测试视角，使用与指标一致的 ours_best；左 GT、右 Render。
并排 PNG 原像素保留；视频播放 10 fps，测试索引顺序，不代表物理时间速度。

| 模型 | 视频 | 并排 PNG | best 步数 |
|---|---|---|---:|
| A1 | [播放](F:/A_Edinburgh/PhysGauss/runs/pidg_constitutive/render_comparisons/rubber_duck_best_20261019_001152/A1/gt_render.mp4) | [50帧](F:/A_Edinburgh/PhysGauss/runs/pidg_constitutive/render_comparisons/rubber_duck_best_20261019_001152/A1/paired) | 24000 |
| A2a | [播放](F:/A_Edinburgh/PhysGauss/runs/pidg_constitutive/render_comparisons/rubber_duck_best_20261019_001152/A2a/gt_render.mp4) | [50帧](F:/A_Edinburgh/PhysGauss/runs/pidg_constitutive/render_comparisons/rubber_duck_best_20261019_001152/A2a/paired) | 24000 |
| A2b | [播放](F:/A_Edinburgh/PhysGauss/runs/pidg_constitutive/render_comparisons/rubber_duck_best_20261019_001152/A2b/gt_render.mp4) | [50帧](F:/A_Edinburgh/PhysGauss/runs/pidg_constitutive/render_comparisons/rubber_duck_best_20261019_001152/A2b/paired) | 31000 |
| A4_single | [播放](F:/A_Edinburgh/PhysGauss/runs/pidg_constitutive/render_comparisons/rubber_duck_best_20261019_001152/A4_single/gt_render.mp4) | [50帧](F:/A_Edinburgh/PhysGauss/runs/pidg_constitutive/render_comparisons/rubber_duck_best_20261019_001152/A4_single/paired) | 37000 |
| A6_single | [播放](F:/A_Edinburgh/PhysGauss/runs/pidg_constitutive/render_comparisons/rubber_duck_best_20261019_001152/A6_single/gt_render.mp4) | [50帧](F:/A_Edinburgh/PhysGauss/runs/pidg_constitutive/render_comparisons/rubber_duck_best_20261019_001152/A6_single/paired) | 50000 |

[校验记录](F:/A_Edinburgh/PhysGauss/runs/pidg_constitutive/render_comparisons/rubber_duck_best_20261019_001152/manifest.json)

每个视频均完成解码帧数检查；所有 PNG 的 GT/Render 区域像素与原帧一致，五模型的 GT 逐帧一致。原模型、数据、既有指标与源码哈希保持不变。
