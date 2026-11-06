"""Measured report; confidence and render proxies are never geometry truth."""
from pathlib import Path
from variants.staged_reconstruction.i10_tracking.io import ROOT,read

B=Path(__file__).parent;V=B/'v1'


def main():
    for name in ['independent_audit.json','controller_audit.json','native_check.json','recovery.json']:
        assert read(V/name)['passed'],name
    ev=read(V/'evaluation.json');m=ev['validation']['means'];manual=read(V/'manual_review.json')
    ca=read(V/'controller_audit.json');tracks=ev['training_track_diagnostic']['models']
    passed=ev['validation']['PSNR_nonregression_gate'] and ev['validation']['parent_nonregression_gate'] and manual['dynamic_completeness_and_cleanup_passed']
    lines=['# L2：可撤销表面片绑定与局部连接，实际训练结果','',
        '用户授权执行，03499339先固定协议，3c45f466提交实现与资格。两组从完全相同J6模型/两个Adam/RNG/相机/physics缓存开始，各完成2000次原生1600更新，共4000正式更新，另各20步资格短训。原J6及失败T3/T4/L1权重/门槛不变。','',
        '使用既有全RGB双向图像轨迹与T4有界原生可见候选，无新颜色静态参考/语义/实例/物体数/GT深度法线。fit偶数观测选择最多3个高斯的持续表面片及固定贡献权重，held奇数观测只作附加约束资格检查；所有120帧仍有普通RGB，因此held不是独立图像验证。','',
        '122片中12片为多高斯，涉及129个独立高斯ID；其余片只获得单点支持，未伪称全部成为真实表面。独立绑定的不同片，仅提出3条非重叠近邻连接，初态2条通过。图覆盖很窄，不能代表整个鸭子材料连通性。连接约束为标量距离的父时变参照，允许旋转与继承形变，但也可能保留父形状错误。','',
        '两组均完整RGB0.8L1+0.2DSSIM，附加重投影和连接权重均0.01、连接尺度0.05，无扫参。fixed_surface保持初态证据；revocable_surface每250个完成更新复核原生完整深度排序的表面片贡献、投影和原RGB局部NCC。低纹理/遮挡保留未知并暂停；连续两次矛盾撤销，连续两次通过恢复。置信度为detached证据得分，无可学习逃避项；撤销约束，不删除点。仅原窗口偶数且当前投影检查通过的帧施加附加损失。','',
        '| 全30张val，原生1600 | J6父 | 固定表面 | 可撤销表面 |',
        '|---|---:|---:|---:|']
    for title,key in [('PSNR(dB)','PSNR'),('SSIM','SSIM'),('L1','L1')]:
        lines.append('| '+title+' | '+' | '.join(f"{m[n][key]:.6f}" for n in ['J6_parent','fixed_surface','revocable_surface'])+' |')
    lines+=['',f"候选较固定对照{m['revocable_surface']['PSNR']-m['fixed_surface']['PSNR']:+.6f}dB，较J6{m['revocable_surface']['PSNR']-m['J6_parent']['PSNR']:+.6f}dB；预定最大下降0.3dB两门槛分别{ev['validation']['PSNR_nonregression_gate']}/{ev['validation']['parent_nonregression_gate']}，无事后放宽。",'',
        '| 真实更新/控制器使用 | 固定表面 | 可撤销表面 |','|---|---:|---:|']
    for title,key in [('有轨迹约束的采样步','samples_with_track'),('有连接约束的采样步','samples_with_edge'),('表面片约束累计应用','total_patch_applications'),('连接累计应用','total_edge_applications')]:
        lines.append('| '+title+' | '+' | '.join(str(ca['arms'][n][key]) for n in ['fixed_surface','revocable_surface'])+' |')
    lines+=['','| 刷新前已完成步数 | 活跃表面片 | 暂停片 | 撤销片 | 活跃连接 |','|---|---:|---:|---:|---:|']
    for h in ca['arms']['revocable_surface']['history']:
        lines.append(f"| {h['completed_updates']} | {h['active_patches']} | {h['suspended_patches']} | {h['revoked_patches']} | {h['active_edges']} |")
    lines+=['','所有122片不按最终置信度过滤的重投影诊断：']
    for n in ['J6_parent','fixed_surface','revocable_surface']:
        q=tracks[n];lines.append(f"- {n}：fit均值{q['fit_mean_px']:.6f}px，held均值{q['held_mean_px']:.6f}px，缺失{q['missing']}/{q['total']}。")
    lines+=['','这些是数据推断/训练对应诊断，不是独立三维误差，置信度不是校准概率。最后2000步模型额外只读复核见terminal_probe.json，不修改已训练控制器。','',
        '实际人工结论：'+manual['conclusion']+'。','']+['- '+s for s in manual['findings']]
    lines+=['','16单测、原生默认关闭RGB/alpha/D及梯度、完整122片资格独立进程同值、异常RNG/flags恢复、组打包贡献和局部坐标/grid/形变非零梯度通过。两组各20步短训，两个独立进程完整模型/双Adam/绑定/RNG/相机/physics/控制器及同份native梯度Adam精确重放通过。原40步CUDA轨迹失败保留，单对比较不代表稳定因果。','',
        '冻结354训练输入/导入源；额外单独冻结只保存证据的I/O适配器，没有改冻结训练公式/决策。保存七次刷新所用模型（仅审计，不能当恢复权重）及全部前后控制器/证据；独立进程原生重放七次、显式CPU状态/置信度方程核验通过。独立复核4000采样、参数与双Adam/静态/physics状态、六完整Gaussian PLY所有字段、九世界XYZ诊断、180图和30指标；两H264各完整解码200帧。实际查看文件及SHA见manual_review.json，未宣称浏览器实测。','',
        '静态4126点所有参数/Adam、physics参数/Adam800/缓存全冻，统一动态池15507点Adam26k→28k，原学习率/eps1e-15/拓扑保留。J6继承早期颜色条件历史，不称从零未知静态发现；无新增物理更新。alpha/D仅渲染代理，黑色缺失不是零误差，无同一固定视角跨时间独立几何GT。','',
        '最终状态：'+('completed_pilot_visual_improvement' if passed else 'completed_not_promoted')+'。完整保留两权重、资格/采样/刷新/视频与负结果；不自动扫权重、扩邻域、延长或恢复旧PAUSED任务。']
    (V/'assessment.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')


if __name__=='__main__':main()
