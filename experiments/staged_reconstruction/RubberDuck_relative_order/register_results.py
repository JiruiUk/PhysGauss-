"""Register current scale-free results, task-only cached patches for shared plans."""
from experiments.staged_reconstruction.RubberDuck_relative_order.register import append
from variants.staged_reconstruction.i10_relative_order import runtime as a
from variants.staged_reconstruction.i10_tracking.io import read


if __name__=='__main__':
    f=read(a.REPORT/'final_review.json');e=read(a.REPORT/'evaluation.json');m=e['validation']['means']
    o=e['ordinal_diagnostic']['models'];u=read(a.REPORT/'controller_audit.json')['arms']['ordinal_depth']
    text=f'''### L8-R Rubber Duck局部相对顺序：匹配训练完成

用户明确允许相对关系试训。首个绝对标定区间方案0训练对，bc46dbc6保留失败，未训练。
5e1a0800另预声明无尺度Small raw顺序，不改旧门槛：至少4偶数同符号、4奇数验证，
组隔离/5x5边界/间隔/每图4对固定；7493假设、322提议全保留，30fit/32诊断对，
仅2动态行、晚期100—115索引，19对原生初始准入，覆盖不足仍须披露。
两臂同完整J6/双Adam/RNG/相机各600更新原生1600，静态/physics/拓扑冻结，
纯RGB vs RGB+native400 D/alpha弱log顺序，weight.001，无绝对深度、新颜色/标签/数量/GT。
候选实际{u['positive_auxiliary_steps']}步非零辅助损失，{u['used_pair_observations']}次有效pair观测。
全30val PSNR：J6 {m['J6_parent']['PSNR']:.6f}、RGB {m['rgb_control']['PSNR']:.6f}、
候选 {m['ordinal_depth']['PSNR']:.6f}dB；两项0.3dB门槛分别{f['numeric_RGB_gates']['vs_control']}/{f['numeric_RGB_gates']['vs_parent']}。
固定fit与先验顺序一致：J6 {o['J6_parent']['fit']['order_correct']}/30、RGB {o['rgb_control']['fit']['order_correct']}/30、候选 {o['ordinal_depth']['fit']['order_correct']}/30；
诊断held：{o['J6_parent']['held']['order_correct']}/32、{o['rgb_control']['held']['order_correct']}/32、{o['ordinal_depth']['held']['order_correct']}/32，仍是RGB训练图像且非GT。
首个控制100步后measurement元数据重复合并失败保留；独立适配修复，不改冻结训练源/损失/门槛，
20测试及重新两独立进程恢复通过后，从J6重跑完整两臂。完整状态/1200采样/10刷新模型/原始patch与符号/
全部候选/静态冻结/6完整PLY+9XYZ/两H264各200帧解码审计通过；实际看86/188早中晚及分解/视频。
视觉清理{f['visual_cleanup_pass']}、完整性{f['visual_completeness_pass']}，状态{f['status']}、pending:false，保留J6不自动加训。
图像射线D/alpha及排序不是高斯语义身份、整鸭距离/厚度或准确几何，父历史颜色来源及未知保留。
详见RubberDuck_relative_order/v1/assessment.md；旧T4/L7失败不改判。
'''
    for name in ['experiments/adaptive_reconstruction_plan.md','experiments/staged_reconstruction_plan.md']:append(name,text)
