# 数据接口 v1

## Manifest

```json
{
  "schema_version": 1,
  "name": "your-scene",
  "dataset_kind": "public",
  "protocol": "image_only",
  "geometry_source": "image",
  "time_unit": "seconds",
  "length_unit": "meters",
  "frames_file": "frames.npz",
  "canonical_file": "canonical.npz"
}
```

路径相对于 manifest。气管改为 `dataset_kind=trachea`、`protocol=ct_assisted`、`geometry_source=ct`。合成测试可用 `geometry_source=synthetic`；public 数据必须实际使用图像推断的 canonical cloud，不能将 CT 改标签绕过协议。

真实数据若使用 `length_unit=normalized`，必须提供正数 `meters_per_unit`。loader 同时转换 canonical xyz/scales、相机平移与 depth，保持投影不变。CT 常见毫米坐标应先除以 1000 转成米，或显式按归一化比例转换。

## frames.npz

| 数组 | Shape | 单位／规则 |
| --- | --- | --- |
| `rgb` | `[T,H,W,3]` | float32 RGB `[0,1]`，黑色背景，非 BGR |
| `K` | `[T,3,3]` | 像素坐标，zero skew，最后一行为 `[0,0,1]` |
| `world_to_camera` | `[T,4,4]` | 右手刚体变换，OpenCV：x 向右、y 向下、z 向前 |
| `times` | `[T]` | 严格递增的秒；t0 默认为 0 |
| `train` / `test` | `[T]` bool | 非空且不重叠，可将 validation 帧置为二者均 false |
| `rgb_valid` | `[T,H,W]` bool | 可选，默认全有效；有无效像素时跳过整图 SSIM |
| `depth` | `[T,H,W]` | 可选，camera-z 深度，必须与坐标同长度单位 |
| `depth_valid` | `[T,H,W]` bool | depth 存在时必需，valid 值必须有限且 >0 |
| `flow` | `[T,H,W,2]` | 可选，源像素到 target 的 forward flow，像素单位 `(du,dv)` |
| `flow_valid` | `[T,H,W]` bool | flow 存在时必需，去掉遮挡／越界像素 |
| `flow_next` | `[T]` int64 | flow 存在时必需，明确 target frame index，-1 表示无 target |

`T>=2,H,W>=3`。整个 split 必须沿用原 benchmark，不能用 toy 的每三帧留一帧规则转换真实数据。仅在 synthetic generator 内使用 toy split。

相机畸变必须先消除。Blender/OpenGL 相机不能直接作为 OpenCV world_to_camera 使用；需要显式变换坐标并验证投影。深度若是 ray distance，要转换成 camera-z；预测单目深度若只有相对尺度，需对齐几何尺度后再开启 depth loss。flow 必须包含相机运动，与参考渲染定义一致。

NPZ 使用 `allow_pickle=False`。无效 depth/flow 可为 NaN，loader 按 mask 清零；有效监督不允许 NaN。rgb 必须整体有限。

## canonical.npz

| 数组 | Shape | 规则 |
| --- | --- | --- |
| `xyz` | `[N,3]` | 有限 canonical 点，N>0 |
| `rgb` | `[N,3]` | 可选 `[0,1]`，默认灰色 |
| `scales` | `[N,3]` | 可选，严格正数，三轴标准差，默认 0.02 |

rotation 初始单位四元数；opacity 初始 sigmoid(1)。实际单位为米时不要盲用 0.02 m 默认尺度；根据点间距提供 scales。

公开场景可导入 COLMAP／图像重建点云，但初始化只能来自允许的训练输入。气管可导入**分割并采样的 CT 表面点云**，需先和视频相机配准。当前不直接读取 DICOM/NIfTI/mesh/PLY，转换为上述 NPZ 后加载。CT 不作为各帧损失监督；缺少颜色时 appearance 仍由视频学习。

## 最小转换片段

```python
import numpy as np

# 外部完成相机标定、坐标统一、CT 配准／图像重建和 splits。
np.savez_compressed("canonical.npz", xyz=points_m.astype("float32"),
                    rgb=colors.astype("float32"), scales=scales_m.astype("float32"))
np.savez_compressed("frames.npz", rgb=images.astype("float32"), K=intrinsics,
                    world_to_camera=extrinsics, times=times_seconds,
                    train=train_mask.astype(bool), test=test_mask.astype(bool))
```

这些变量需来自你们真实数据，不是可直接运行的预处理脚本。准备好后运行 `physgauss validate-data --manifest ...`。大场景必须先接入 CUDA backend；临时下采样只用于开发验证，不用于声称官方比较结果。
