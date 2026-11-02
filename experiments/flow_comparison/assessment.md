# 光流与轨迹导出速度：完整训练对照

Rubber Duck，seed0；120训练/30验证/50测试。同一完整3k起点训练至总50k。
K1：独立速度头 + 运动光流。K2：轨迹有限差分速度 + 相同运动光流。两组共用单调时钟、局部材料坐标动量残差和图像损失。
光流与相机补偿深度为估计量；仅120训练图像生成监督。光流400×225，RGB1600×900。

| 模型 | 快照 | PSNR↑ | SSIM↑ | LPIPS↓ | MS-SSIM↑ |
|---|---|---:|---:|---:|---:|
| H3_combined | 50000 | 40.8782310 | 0.9950647 | 0.0094411 | 0.9969448 |
| J1_monotonic_kinematic | 50000 | 40.0717239 | 0.9945284 | 0.0107166 | 0.9966325 |
| K1_flow_independent | best | 38.3013039 | 0.9931610 | 0.0130301 | 0.9952919 |
| K1_flow_independent | 50000 | 38.3013039 | 0.9931610 | 0.0130301 | 0.9952919 |
| K2_flow_trajectory | best | 34.2064972 | 0.9882474 | 0.0219847 | 0.9905336 |
| K2_flow_trajectory | 50000 | 34.2064972 | 0.9882474 | 0.0219847 | 0.9905336 |

两组新模型之间只改变速度来源。相对历史H3/J1还共同改变了光流、时钟或动量离散化，不能把差异全归于光流。
K2内部速度误差为零是结构定义，不能作为物理预测正确的证据。没有增加外力、本构或自由材料参数。
本轮只用验证选择；最终测试此前有历史调参暴露，不能称全新未暴露测试。单场景、单随机种子的结论有限。

## GT / Render

- [K1_flow_independent 浏览全部帧](F:/A_Edinburgh/PhysGauss/output/flow_comparison/K1_flow_independent/index.html)
- [K1_flow_independent 验证选择拼接视频](F:/A_Edinburgh/PhysGauss/output/flow_comparison/K1_flow_independent/GT_Render_best.mp4)
- [K1_flow_independent 完整压缩包](F:/A_Edinburgh/PhysGauss/output/flow_comparison/K1_flow_independent_GT_Render.zip)
- [K2_flow_trajectory 浏览全部帧](F:/A_Edinburgh/PhysGauss/output/flow_comparison/K2_flow_trajectory/index.html)
- [K2_flow_trajectory 验证选择拼接视频](F:/A_Edinburgh/PhysGauss/output/flow_comparison/K2_flow_trajectory/GT_Render_best.mp4)
- [K2_flow_trajectory 完整压缩包](F:/A_Edinburgh/PhysGauss/output/flow_comparison/K2_flow_trajectory_GT_Render.zip)

## 核查后结论（V2 诊断）

两组均完成50k，验证选择锁定后测试。当前配置下 K2 未达到预先记录的综合改进门槛。

| 验证选择快照 | 验证 PSNR↑ | 验证 LPIPS↓ | 轨迹 EPE↓ (px) | 速度推进 EPE↓ (px) | 观测像素覆盖 |
|---|---:|---:|---:|---:|---:|
| K1_flow_independent | 38.02613 | 0.0137166 | 0.78371 | 1.19321 | 99.998% |
| K2_flow_trajectory | 33.84882 | 0.0227465 | 0.84683 | 0.84675 | 99.586% |

K2 验证 PSNR 改变 -4.177 dB；LPIPS 上升 65.8%；速度推进 EPE 下降 29.0%，几何轨迹 EPE 上升 8.1%。运动推进更接近观测，但渲染质量和轨迹运动没有同时提升。

K1 最终内部速度相对偏差约99.80%；其推进 EPE 与仅相机运动/物体静止 baseline 接近。K2 内部偏差为零是结构定义；独立二维观测显示部分运动收益，仍不能证明三维真实速度或应力正确。

### 后处理与训练流程缺陷

核验发现两个代码约定错误，旧模型、源码、缓存和原诊断全部保留：

1. 实际 PIDG 加载器把 FovX=fovy、FovY=fovx 传给 Camera；旧光流深度生成/运动诊断脚本使用常规 FoV 分配，与已有重建权重的投影约定不同。纠正诊断后，观测前景的模型覆盖从约48–50%升至99%以上。这不要求改写原 PIDG 加载器，而是让辅助流程遵守实际训练相机。
2. 实际 CUDA 的 x_mu=mean-pixel，仿射流应使用其负值；此外 metadata 的 alpha*T 在 T 更新后才写入，并非 RGB 合成权重。V2 从 opacity/conic 重建更新前贡献权重。CPU 仿射膨胀测试和真实 CUDA 单 Gaussian 测试验证了这两点。

旧 motion_validation.json、gradient_balance.json、motion_comparison.png 作为 V1 存档，已由带 _v2 的诊断替代。V1 final_verification.json 仍证明原实验文件/数值/导出符合其冻结协议，不证明目标几何正确或模型物理正确。训练使用了有上述问题的 V1 光流流程，因此不能据此否定运动光流或轨迹导数这一研究方向，也不能把退化唯一归于缺少外力。

### 梯度和差分敏感性

V2 梯度探针使用实际训练相机，但特意计算冻结的原训练目标，反映模型实际经历的目标；不是修正后新训练目标。
- K1_flow_independent：三个固定训练时刻，形变 MLP 的加权动量/光流梯度范数比为 360.5–679.1；速度差分步长减半/加倍相对变化 0.44–1.12%，该组实际加速度来源相对变化 1.2–32.2%。
- K2_flow_trajectory：三个固定训练时刻，形变 MLP 的加权动量/光流梯度范数比为 23157.1–453332.5；速度差分步长减半/加倍相对变化 0.50–1.22%，该组实际加速度来源相对变化 14.6–46.2%。

这些局部探针提示损失尺度和轨迹二阶差分需要重点核查；没有测量完整训练全部梯度，也没有与图像梯度做同批次平衡审计，不能据此宣称唯一因果。K1 加速度是独立速度的一阶导，K2 是轨迹二阶导，数值幅值不可当同一种物理真值直接排名。

### 下一轮

按用户授权进入统一分支。但先修正光流几何/相机约定，并给新结构配同流程的原结构对照，以免把流程修复的收益算到共享分支上。新候选共享 deformation Grid4D 和 latent 输出形变与预测应力，速度来自完整中心轨迹。保留原 PIDG、K1/K2、预测应力含义和开关关闭路径，不添加外力/本构/材料未知量。下一轮独立记录初始化和预算、先测试与短训再完整训练。

### 可视与文件核验

两组均有100张3200×950拼接PNG、两段50帧/5秒视频和ZIP；原导出逐像素及独立PSNR核验通过。已解码两组 best 视频50帧并生成早/中/晚(索引0/21/49)审阅图；原帧85为索引21。两组 best 均选中50k，与最终快照相同。原帧85，K2 有完整移动鸭体缺失、鸭体/平台接触区拉丝和边缘伪影；K1 也有局部孔洞/浮影，不能只看全图黑背景主导的PSNR。

- [修正运动与梯度图](F:/A_Edinburgh/PhysGauss/output/flow_comparison/motion_comparison_v2.png)
- [解码视频早中晚审阅图](F:/A_Edinburgh/PhysGauss/output/flow_comparison/decoded_video_review.png)
- [训练曲线](F:/A_Edinburgh/PhysGauss/output/flow_comparison/training_comparison.png)
