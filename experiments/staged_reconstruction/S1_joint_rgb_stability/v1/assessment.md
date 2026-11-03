# J1-R1：固定epsilon对照仍未通过恢复轨迹门槛

ac16bc95预声明单一共同Adam epsilon1e-8（原J1为1e-15），不是扫参，其他
学习率、损失和两条件结构门槛不变。历史源码、权重和原J1失败均未改动。
3个新单测通过，disabled恒等，启用时只变epsilon，所有rates/steps/moments
不变；两条件原生同值RGB/alpha、静态几何冻结及有限梯度smoke通过。

首组diffuse/frozen完整40与20+20仍未满足atol1e-6/rtol1e-5，因此停止其余
三组；没有正式6000步训练。动态坐标最大差0.00150481，opacity logit最大
差0.04262304，xyz-grid最大差0.000374913。部分grid差异比原J1小，但这
不是原恢复检查通过，也没有证明几何/拖影改善。initial完整state和最终
全部RNG、相机采样仍精确相同，loss差异也出现在20步断点之前。

原协议完整保留；不再试其他epsilon、不换seed、不放宽容差、不修改冻结
训练/原生kernel。全部失败日志、initial/step20/step40和分母统计保留。
无新PLY/视频/Rubber Duck86/188结果，所有GPU worker已退出，不影响并行I9。

当前需要用户选择验收定义：原J1把完整状态恢复与40步原生CUDA轨迹一致性
放在同一检查中。建议另立“完整state精确恢复+同份native梯度的Adam精确
重放”验收，保留原40步失败和native数值不确定性，并维持全部重建质量
门槛；具体见S1_joint_rgb/recovery_acceptance_proposal.md。用户未选择前
保持正式训练停止，不把建议当批准。只可继续不改变验收的独立诊断。
