# 初始实现验证记录

验证日期：2026-10-13。实际执行目录：`F:\A_Edinburgh\PhysGauss`。

## 环境

- Windows，Python 3.12.14。
- 独立 `.venv`；项目 editable install，包路径确认为项目 `src/physgauss`。
- PyTorch 2.14.1+cpu、NumPy 2.5.4、Pillow 12.3.0。
- `pip check`：No broken requirements found。
- 本地 Git 仓库已初始化，分支 `main`；未创建远程仓库，初始源文件保留为未提交内容。

## 检查结果

`python -m pytest -q --tb=short`：**17 passed**。覆盖解析 Cauchy 残差、时间／空间偏导与二阶参数梯度、轨迹速度、透视渲染／遮挡／相机运动、参数梯度、零光流梯度、协议与 split 验证、真实单位转换、训练／续训／评估／导出。

断点续训测试对比「连续 4 步」与「2 步后恢复至 4 步」，模型 state_dict 在同一环境中逐项精确一致。

`scripts/run_smoke.ps1 -RunName smoke`：**通过**。合成 9 帧、12 个高斯、24×24 图像；6 帧训练、3 帧测试，训练 30 步，开启不确定性、轨迹、动量与自适应采样。

| 参考测试指标 | 数值 |
| --- | --- |
| Test PSNR | 27.8531 dB |
| SSIM reference | 0.87034 |
| Depth MAE | 0.021515，合成归一化长度 |
| Flow EPE | 0.242423 pixels，存在有效 flow 的测试帧 |

这些数值仅用于记录 toy 工程验证，不是 PIDG 或临床数据指标；synthetic GT 来自同一个参考渲染器。

## 生成文件

```text
data/synthetic/manifest.json
data/synthetic/frames.npz
data/synthetic/canonical.npz
runs/smoke/config.json
runs/smoke/data_provenance.json
runs/smoke/train.jsonl
runs/smoke/checkpoint.pt
runs/smoke/evaluation/metrics.json
runs/smoke/evaluation/frame_0002.png
runs/smoke/evaluation/frame_0005.png
runs/smoke/evaluation/frame_0008.png
runs/smoke/state_t050.npz
```

评估 PNG 从左到右为 GT、预测、绝对误差。状态 NPZ 包含动态高斯参数、速度、对称应力、光度 variance 和时间。数据和运行产物被 Git 忽略。
