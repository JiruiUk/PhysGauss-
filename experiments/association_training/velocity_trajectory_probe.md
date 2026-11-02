# 当前速度分支与形变轨迹的只读比较

当前T1最终52k：两分支内部相对差异100.033%，速度网络幅度仅为完整形变轨迹速度的3.024%。鸭子观测区域的轨迹/速度推进EPE分别0.6948/1.8167px，形变轨迹更符合已观测二维运动，真实三维速度与材料身份仍无真值核验。

比较对象是网络输出的速度与完整渲染中心轨迹的导数；不同特征网格的参数不能直接相减。

Full final checkpoints; same normalized0-1 time;256 evenly spaced rows x5 times; pooled component RMS; complete center includes rotation+translation; FD h=.001.

30 existing validation-source to next-training frames; fixed raw-UniMatch FB-valid foreground; extra source-yellow subset.400x225 pixels. Future modeled covariance shared by both motion hypotheses.

| 完整状态 | 内部相对RMSE | 速度/轨迹RMS | 前景轨迹EPE | 前景速度EPE | 鸭子轨迹EPE | 鸭子速度EPE |
|---|---:|---:|---:|---:|---:|---:|
|C1_original_50k|100.033%|3.041%|0.742317|1.226415|0.698033|1.817304|
|T1_association_52k|100.033%|3.024%|0.741334|1.226241|0.694826|1.816729|
|O1_flow_on_50k|111.528%|55.695%|0.748556|1.024817|0.663923|1.380330|

内部误差以模型自身形变轨迹速度为分母，不是相对真实速度误差。速度单位为场景单位/归一化时间，不是m/s。
EPE单位px@400x225，先逐对取均值，再30对等权平均。观测像素掩码固定，未覆盖像素不通过剔除来降低误差。黄色鸭子区域由图像颜色/alpha提取。
形变轨迹与观测二维运动更一致，不证明真实三维速度、深度或材料身份。O1与其他模型拓扑可能不同，内部256行是各模型自身均匀抽样。
已有修正后的前20贡献层光流诊断用于两假设的同条件比较；不是先前点库评估的完整贡献积分。检查点SHA前后不变，零优化器更新。
