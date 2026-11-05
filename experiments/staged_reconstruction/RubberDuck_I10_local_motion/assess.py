"""Write measured results after native, recovery, artifact and manual reviews."""
from pathlib import Path
from variants.staged_reconstruction.i10_tracking.io import read

B=Path(__file__).parent;V=B/'v1'


def main():
    for name in ['native_check.json','recovery.json','independent_audit.json','support_audit.json']:
        assert read(V/name)['passed']
    ev=read(V/'evaluation.json');manual=read(V/'manual_review.json');m=ev['validation']['means']
    tracks=ev['training_track_diagnostic']['models'];local=ev['local_graph_diagnostic']['models']
    passed=ev['validation']['PSNR_nonregression_gate'] and ev['validation']['parent_nonregression_gate'] and manual['dynamic_completeness_and_cleanup_passed']
    lines=['# L1：有支撑时段的局部共同运动训练已实际完成','',
        '用户指出前轮审计后未更新模型。此前把全时段覆盖准入当作一切训练的开关过于严格；保留T3/T4失败记录，另立397e3b35受限局部试验。两组各完成2000次原生1600 Gaussian更新，共4000正式更新，另各20步资格短训。新权重已生成，是否采用取决于结果，不把构图或准备代码叫训练。','',
        '两组从同一J6完整模型、两个Adam、RNG、相机剩余栈和physics缓存开始：均用完整RGB0.8L1+0.2DSSIM与同一可见审计轨迹重投影0.01；候选再加固定0.01局部相对修正。父时变邻域偏移按0.05归一化，只要求邻居与锚点的修正协同，不将整个鸭子视为刚体，也不把父偏移当正确深度。它可能保留错误局部形状，且会抵抗某些需要的局部旋转修正。','',
        '1325个去重可见观测覆盖89/120训练帧，9367个去重无向邻接对覆盖71/120帧。轨迹仅在其当前帧原生可见检查通过时生效；邻域仅在原12帧审计窗口内生效。其余帧仅RGB，没有外推假图。共享形变网络仍可能让更新影响其他时刻，因此不能把“这些时刻没有附加损失”说成模型在这些时刻不变。','',
        '| 全30张val，原生1600 | J6父 | 审计轨迹对照 | 轨迹＋局部运动 |',
        '|---|---:|---:|---:|']
    for name,key in [('PSNR(dB)','PSNR'),('SSIM','SSIM'),('L1','L1')]:
        lines.append('| '+name+' | '+' | '.join(f"{m[n][key]:.6f}" for n in ['J6_parent','audited_tracks','local_motion'])+' |')
    lines+=['',f"局部候选较匹配对照PSNR差{m['local_motion']['PSNR']-m['audited_tracks']['PSNR']:+.6f}dB，较J6差{m['local_motion']['PSNR']-m['J6_parent']['PSNR']:+.6f}dB。预定两项最大下降0.3dB门槛分别为{ev['validation']['PSNR_nonregression_gate']}/{ev['validation']['parent_nonregression_gate']}，均未改。",'',
        f"同一1325训练对应的平均投影误差：父{tracks['J6_parent']['mean_px']:.6f}、对照{tracks['audited_tracks']['mean_px']:.6f}、局部{tracks['local_motion']['mean_px']:.6f}px。9367父邻接对偏移的平均世界误差：对照{local['audited_tracks']['mean_world']:.8f}、局部{local['local_motion']['mean_world']:.8f}。前者是优化过的对应拟合，后者是父参照一致性，二者均不是独立几何/完整性真值。",'',
        '实际人工结论：'+manual['conclusion']+'。']
    lines+=['']+['- '+finding for finding in manual['findings']]
    lines+=['','14单测、默认关闭/原生RGB/alpha/D及梯度、局部canonical/grid/deform有限非零梯度、两臂20步短训与每臂两个独立进程精确模型/双Adam/RNG/相机/physics恢复、同一native梯度Adam更新复现通过。原40步CUDA确定性轨迹失败记录及用户授权的状态恢复标准保留；单对照不证明稳定因果效应。首次CLI检查遗漏PYTHONPATH的环境失败单独保留，修正环境后通过，未修改冻结训练源码。','',
        '最终独立审计复核4000相机采样、实际参数改变、完整状态及静态/physics不可变、6完整Gaussian PLY所有字段和9世界XYZ诊断、180原生图像及全30指标均值；两H264 Main/yuv420p/faststart视频各完整解码200帧。实际看过的图和SHA列于manual_review.json；未宣称浏览器实测播放。','',
        '静态4126点全部参数/Adam及physics参数/Adam800/缓存全冻；动态统一池15507点Adam26k→28k，eps1e-15/原率/拓扑保留。点数不是鸭子数，未赋予物理身份。新训练没有颜色静态参考、语义/实例标签、GT深度法线或物体数，但J6父有历史颜色条件训练；不是从零未知静态发现，也不是physics续训。原生固定86/188跨时间无独立同一时刻多视角GT；alpha/D是渲染代理，黑色缺失不是零误差。','',
        '结果：'+('completed_pilot_visual_improvement' if passed else 'completed_not_promoted')+'。保留全部权重与记录，不自动调权、重跑或延长，不恢复旧PAUSED任务。']
    (V/'assessment.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')


if __name__=='__main__':main()
