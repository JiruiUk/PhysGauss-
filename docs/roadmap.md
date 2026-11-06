# 后续研究实施顺序

## P0：真实数据与正式重建主干

1. 确定一份 public 场景和一份 CT 气管场景；记录可用 cameras、depth 来源、flow 约定、单位和原始 splits。
2. 实现 source-specific converter，分别支持 public image-derived cloud 和已配准的 CT mesh/point cloud。
3. 将 CUDA rasterizer 包装为 Renderer，验证 RGB/depth/flow/variance 输出及参数梯度；记录依赖 commit、许可证和 CUDA 环境。
4. 接入成熟动态 3DGS backbone 的 hash encoder、densification/pruning、SH appearance。修改点数时必须同步 optimizer 与 collocation scores。
5. 先复现只含 RGB reconstruction 的 baseline，再逐项启用方法损失；接入正式 SSIM 和 LPIPS evaluator。

## P1：对应方法贡献的可验证实验

- Baseline：相同 backbone 和初始化，仅标准视觉重建。
- `+depth`：只在输入协议允许时开启；public 默认 depth loss=0。
- `+uncertainty`：对比 RGB L1 与有界 NLL，检查 variance collapse/inflation 和遮挡区域。
- `+trajectory`：检验 3D 轨迹导数与 Eulerian velocity 的一致性。
- `+physics`：对比 warmup/ramp、自适应 residual sampling 与固定均匀采样。
- `full`：评估 reconstruction、motion、physics，记录多 seed 均值与方差。

当前 adaptive 只是 warmup 与残差采样；若论文提出 uncertainty-conditioned adaptive physics，需要明确定义权重和梯度策略并另外实现。不能将初版视为该贡献已完成。

## P2：可辨识的气管参数估计

在 velocity/stress 基础上选择实际材料模型与未知量，例如弹性参数或压力。补充 deformation gradient、本构约束、载荷／边界条件、SI 标度和独立物理 GT。只有动量与光流通常不足以唯一恢复 stiffness/pressure；需要设计合成可辨识性实验、加载条件和参数误差指标。

CT 只提供 canonical geometry，image-only 与 CT-assisted 分开报告。对比 CT-init 与 image-init 时，要明确几何先验不同。气管的 3D motion、geometry、参数误差由你们已有数据决定，当前评估器还没有这些 GT 接口。

## 短期优先级

先打通一个真实场景和官方评价，再做能支持论文主张的最少量消融。不要先批量跑 20 场景或把 smoke 指标写进论文。项目中的配置仅为启动模板，需结合真实单位、场景规模和监督可靠性调整。
