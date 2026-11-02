# 单调物理时钟 + 速度—轨迹一致性：完整训练结果

模型：J1_monotonic_kinematic。Rubber Duck，seed0，120训练/30验证/50历史已暴露测试。
复用与H3相同的完整3k起点，继续训练至总50k，不是从收敛H3再追加50k。
原PIDG代码与H3权重未修改；新分支保留原图像/覆盖/邻边应变/CMR，加入固定单调速度时钟与0.001联合运动学一致性。
数据帧编号/共同时间轴/原相机采样器未改；没有外力/本构/自由材料参数。

验证PSNR选中第49000步；选择在打开最终测试渲染前锁定。最终50k也完整导出。

| 快照 | PSNR↑ | SSIM↑ | LPIPS↓ | MS-SSIM↑ | 相对H3最终PSNR | 改善帧/50 |
|---|---:|---:|---:|---:|---:|---:|
| H3已有最终50k | 40.87823 | 0.9950647 | 0.0094411 | 0.9969448 | - | - |
| best | 40.08085 | 0.9944794 | 0.0107767 | 0.9966155 | -0.79738 | 17 |
| 50000 | 40.07172 | 0.9945284 | 0.0107166 | 0.9966325 | -0.80651 | 15 |

最终内部速度—轨迹相对RMSE：99.73%（5个训练探针平均；非物理速度真值）。

时钟与一致性同时改变，因此图像差异不能全部归因为时钟修正；也不能由自一致性改善推断材料或外力识别成功。

## Ground Truth / Render 导出

- [交互浏览全部帧](F:/A_Edinburgh/PhysGauss/output/monotonic_training/J1_monotonic_kinematic/index.html)
- [验证选择模型拼接视频](F:/A_Edinburgh/PhysGauss/output/monotonic_training/J1_monotonic_kinematic/GT_Render_best.mp4)
- [最终50k拼接视频](F:/A_Edinburgh/PhysGauss/output/monotonic_training/J1_monotonic_kinematic/GT_Render_50000.mp4)
- [原帧85的完整拼接PNG](F:/A_Edinburgh/PhysGauss/output/monotonic_training/J1_monotonic_kinematic/pairs/best/00021.png)

共100张3200×950拼接PNG，两段50帧视频；各PNG左右图像区与原GT/render逐像素一致，独立重算PSNR最大差0.000005612dB。

## 原始证据

[训练与导出协议](plan.json) | [预检](preflight.json) | [源码冻结](source_freeze.json) | [结果JSON](results.json)

最终测试有历史调参暴露；本轮仅用验证选择，但不把它描述为全新未暴露测试集。

## 2026-11-02 独立复核

100张PNG、两段视频、压缩包、完整断点、原模型哈希及速度诊断均独立复核通过。联合速度监督样本量比冻结试验少约40.86倍，局部梯度探针发现冲突；尚不能据此唯一归因。详见[完整复核报告](verification_20261102.md)。
