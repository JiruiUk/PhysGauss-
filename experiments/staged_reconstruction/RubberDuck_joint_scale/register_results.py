"""Append only the completed repaired audit, using task-only cached patches."""
from variants.staged_reconstruction.i10_joint_scale import io
from experiments.staged_reconstruction.RubberDuck_joint_scale.register import append


def main():
    f=io.read(io.REPORT/'final_review.json');assert not f['pending_review']
    section='''### L7-J Rubber Duck跨帧联合尺度与RGB锚点扩展：完成

841d7c0d预声明后完整执行：新增1981个全图纹理种子/963条RGB合格轨迹，
265条静止相容，与原362条合计627条；507重复组跨帧fit/check隔离，
4px去重后5351个锚点观测。固定Small/J6，0次Gaussian更新，无新颜色/静态标签/数量/GT。
原稀疏数值失败保留，143a68cd独立修复相同目标/权重/门槛/预算的精确数值求解。
同扩展锚点逐帧26/120，联合39/120、早中晚17/19/3；原L7-P为24/120、7/7/10，
原分割方式不同，不当单因素对照。潜在二维fit凸包均值6.846%→12.884%，不是准确表面覆盖。
122条原生合格运动假设中仅2条进入数值检验、1条通过，1个不同动态行/5个fit深度提议。
晚期及动态覆盖门槛失败，状态completed_audit_not_admitted；保留J6、弱深度默认关闭。
34测试、独立进程数组/全部120决策重放、独立目标/KKT/组/held/动态/全53196边界核验通过；
7张原始训练图/诊断图实际看过。原T4全局失败、历史颜色来源、未知身份不改判。
详见RubberDuck_joint_scale/v1/assessment.md；无新86/188/PLY/视频或伪影减少声明。
'''
    for name in ['experiments/adaptive_reconstruction_plan.md','experiments/staged_reconstruction_plan.md']:append(name,section)


if __name__=='__main__':main()
