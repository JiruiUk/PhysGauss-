# J3文献核验与边界（2026-11-04）

- [DN-Splatter, arXiv v3](https://arxiv.org/html/2403.17822v3)：本轮重新打开并
  阅读4.1，使用归一化渲染深度及RGB边缘感知log深度损失；外部先验来自
  sensor或经过SfM标定的monocular估计。这里只借鉴在渲染量上施加稳健约束，
  不复现它的先验/法线/初始化/完整方法；本次父模型深度并非其可靠sensor。
- [GSPrior作者项目页](https://takeshie.github.io/GSPrior/)：本轮阅读abstract
  和method overview；从当前渲染深度融合TSDF，再利用场和带宽约束Gaussians。
  未读完整论文/未实现TSDF，不把J3称GSPrior复现；自约束不提供独立GT。
- [DRGSplat Google Research摘要](https://research.google/pubs/drgsplat-depth-regularized-3d-gaussian-splatting/)：
  只核验官方摘要，采用渲染深度/法线/不确定平坦性正则。未获得其完整方法，
  不宣称读完或复现，也不照搬它的外部深度/法线先验。

原生实现已读flow-based-diff-gaussian-rasterization的backward：累计depth和
alpha有至opacity梯度通路；仍须实际原生梯度及有限差分验证。J3固定0.1权重
和log1p稳健形式是本项目单次预声明选择，不是论文推荐数值，不按GT扫选。
不使用失败D0–D7的推断深度，不新增材料、力或本构模型。
