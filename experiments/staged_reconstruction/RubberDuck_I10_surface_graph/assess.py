"""Measured L3 report; transport, confidence and rendered depth remain proxies."""
from pathlib import Path
from variants.staged_reconstruction.i10_tracking.io import read

B = Path(__file__).parent; V = B/'v1'


def main():
    for name in ['input_audit.json','held_audit.json','title_review.json','native_check.json','recovery.json','independent_audit.json','controller_audit.json']:
        assert read(V/name)['passed'], name
    ev = read(V/'evaluation.json'); manual = read(V/'manual_review.json')
    m = ev['validation']['means']; ca = read(V/'controller_audit.json')
    h = read(V/'hypotheses.json')['summary']; initial = read(V/'initial_health.json')['summary']
    p = read(B/'protocol.json')
    passed = ev['validation']['PSNR_nonregression_gate'] and ev['validation']['parent_nonregression_gate'] and manual['dynamic_completeness_and_cleanup_passed']
    lines = ['# L3：统一动态池、锚点局部表面片与可撤销形变图', '',
        '用户再次授权实际尝试四项方法。ec06e5ac在任何新候选测量前固定协议，0ffc1731提交实现/资格。两组从精确J6完整模型/两个Adam/RNG/相机/physics缓存起点，各完成2000步原生1600训练，共4000正式更新，另各20步短训。L2负结果、历史权重、门槛及原生40步非确定性失败均保留。', '',
        '共享15507个动态高斯，不估计物体数或物理身份。保留122条既有全RGB/T4局部锚点；fit偶数观测仅提议最多8个近邻，按距离/ID排序，同一高斯最多保留3个重叠归属并均分固定权重。held检查不选择成员，但仍参与普通RGB训练，因此不是独立图像验证。', '',
        '邻域节点的检查位置由“锚点原RGB轨迹＋父模型邻点相对投影”搬运得到。这是父模型条件代理，不是新测得的独立3D/2D轨迹。每个节点检查自己的原生完整深度排序贡献、原RGB局部纹理/NCC和投影；附加重投影损失只作用于原始锚点，不作用于邻域代理目标。', '',
        f"fit提议{h['neighbor_hypotheses']}个邻域假设，涉及{h['distinct_rows']}个独立高斯（含锚点）；共{h['proposed_edges']}条提议边。初态保留{initial['active_neighbor_rows']}个不同邻域行、{initial['active_edges']}条连接，通过预定至少25行/50边门槛。L2初态仅2条独立锚点边。新增覆盖仍局部，不能称全鸭表面或正确材质归属。", '',
        '局部图含锚点-邻域辐条和邻域间三角连接，按父时变标量距离施加旋转不变的拉伸/剪切约束，尺度0.05、总局部权重0.01，锚点重投影权重0.01，无扫参。它允许旋转和继承形变，也可能保护父错误；未增加缺失点或施加强不透明度/法线监督。', '',
        '每250个完成更新复核一次；未知立即暂停，连续两次矛盾撤销、两次通过恢复。根锚点不活跃时立即暂停所属邻域约束；保留点、候选与未知归属。置信度为detached证据分数，不是学习变量或校准概率。纯RGB对照的轨迹/连接附加损失精确为零。', '',
        '| 全30张val，原生1600 | J6父 | 纯RGB续训 | 表面图 |', '|---|---:|---:|---:|']
    for title, key in [('PSNR(dB)','PSNR'),('SSIM','SSIM'),('L1','L1')]:
        lines.append('| '+title+' | '+' | '.join(f"{m[n][key]:.6f}" for n in ['J6_parent','rgb_control','surface_graph'])+' |')
    lines += ['', f"表面图相对纯RGB {m['surface_graph']['PSNR']-m['rgb_control']['PSNR']:+.6f}dB，相对J6 {m['surface_graph']['PSNR']-m['J6_parent']['PSNR']:+.6f}dB；原最大下降0.3dB两门槛：{ev['validation']['PSNR_nonregression_gate']}/{ev['validation']['parent_nonregression_gate']}。单次匹配比较不代表跨随机轨迹稳定因果。", '',
        '| 真实更新中的约束使用 | 纯RGB | 表面图 |', '|---|---:|---:|']
    for title, key in [('有锚点约束的步数','samples_with_track'),('有图约束的步数','samples_with_edge'),('锚点累计应用','total_patch_applications'),('图边累计应用','total_edge_applications')]:
        lines.append('| '+title+' | '+' | '.join(str(ca['arms'][n][key]) for n in p['arms'])+' |')
    lines += ['', '初态覆盖的时间分布：']
    for q in read(V/'held_audit.json')['temporal_coverage']:
        lines.append(f"- 时间三分段{q['temporal_third']}：{q['anchors']}锚点，{q['proposed_neighbor_hypotheses']}个邻域提议，{q['active_neighbor_rows']}个合格邻域行、{q['active_edges']}条连接。")
    lines += ['', '连接集中前段，后段初态零合格邻域边。这是可推广范围的实际限制，不按全局总边数掩盖；完整训练仍按固定协议完成，无结果驱动改门槛。', '', '| 刷新前完成步数 | 活跃锚点 | 活跃邻域行 | 活跃节点 | 暂停节点 | 撤销节点 | 活跃连接 |', '|---|---:|---:|---:|---:|---:|---:|']
    for q in ca['arms']['surface_graph']['history']:
        lines.append('| '+' | '.join(str(q[k]) for k in ['completed_updates','active_roots','active_neighbor_rows','active_patches','suspended_patches','revoked_patches','active_edges'])+' |')
    lines += ['', '原122锚点全部保留、不按最终置信度筛选的诊断：']
    for name, q in ev['training_track_diagnostic']['models'].items():
        lines.append(f"- {name}：fit {q['fit_mean_px']:.6f}px，held {q['held_mean_px']:.6f}px，缺失{q['missing']}/{q['total']}。")
    lines += ['', '训练对应误差和父长度一致性不是独立几何GT。固定六相机-时间的动态暗色高alpha像素仅作渲染代理：', '',
        '| 相机/时间 | J6父 | 纯RGB | 表面图 |', '|---|---:|---:|---:|']
    proxies = ev['diagnostics']['high_alpha_proxies']
    for camera in p['fixed_cameras']:
        for time in p['fixed_times']:
            values = [next(q for q in proxies if q['camera']==camera and q['time']==time and q['model']==n) for n in ['J6_parent','rgb_control','surface_graph']]
            assert len({q['pixels'] for q in values}) == 1
            lines.append(f'| {camera}/{time} | '+' | '.join(f"{q['dynamic_dark_high_alpha']}/{q['pixels']}" for q in values)+' |')
    lines += ['', '定义为动态渲染所有RGB通道<0.05且alpha>0.5，只供评估，不进训练或删点。低alpha/黑色缺失不是零误差，D/alpha是渲染代理；无固定时间多视角独立几何GT。', '',
        '实际人工结论：'+manual['conclusion']+'。', '']+['- '+q for q in manual['findings']]
    lines += ['', '15项CPU单测、原生flags-off RGB/alpha/D精确一致及原梯度容差、全部789节点独立进程原生测量一致、异常RNG/flags/模型双Adam无变化、打包贡献和canonical/grid/deform局部非零有限梯度通过。两臂20步短训、四独立进程完整模型/双Adam/绑定/RNG/相机/physics/控制器与同份native梯度Adam精确恢复通过。', '',
        '冻结356训练输入/导入源，另冻结只记录模型的I/O适配器和36个评估/验证文件。输入图全量重放及122窗口held扰动不影响提议的审计通过。七次刷新模型及所有证据在独立进程原生逐值重放、显式CPU状态/置信度与根依赖核验通过。独立核验4000采样/使用记录、六完整Gaussian PLY字段、九世界XYZ诊断、180图/30指标、两H264各完整解码200帧；实际查看32图及SHA见manual_review.json，未实测浏览器播放。', '',
        '静态4126点全参数/Adam及physics参数/Adam800/缓存精确冻结；动态Adam26k→28k，原学习率/eps1e-15/拓扑不变。新训练无颜色静态参考/语义/物体数/GT深度法线，J6历史颜色条件来源仍披露，不能称从零未知静态发现或物理训练。实现期import/syntax失败在冻结前修复并保留日志；冻结后训练公式/门槛不变。', '',
        '最终：'+('completed_pilot_visual_improvement' if passed else 'completed_not_promoted')+'。保留两臂完整权重和全部证据，不自动扫权重/阈值、延长、换候选或恢复旧PAUSED任务。']
    (V/'assessment.md').write_text('\n'.join(lines)+'\n', encoding='utf-8', newline='\n')


if __name__ == '__main__':main()
