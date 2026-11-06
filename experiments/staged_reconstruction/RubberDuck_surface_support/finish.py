"""Close the fixed trial only after independent numerical and actual visual review."""
from pathlib import Path
from variants.staged_reconstruction.i10_tracking.io import ROOT,read,write,sha

B=Path(__file__).parent;V=B/'v1';R=ROOT/'runs/pidg_staged_reconstruction/RubberDuck_surface_support/v1'
NAMES=['J6_parent','temporal_control','surface_support']

def main():
    required=['unit_tests.json','native_check.json','recovery.json','initial_coverage.json',
              'independent_audit.json','controller_audit.json','structural_diagnostic.json']
    for name in required:assert read(V/name)['passed'],name
    ev=read(V/'evaluation.json');manual=read(V/'manual_review.json');ca=read(V/'controller_audit.json')
    assert manual['actual_visual_review'] and manual['fixed_native_RGB_alpha_D_reviewed']
    assert len(manual['reviewed_images_sha256'])==32
    for path,h in manual['reviewed_images_sha256'].items():assert sha(ROOT/path)==h,path
    sd=read(V/'structural_diagnostic.json')['models'];m=ev['validation']['means']
    bank=read(V/'surface_bank.json')['summary']
    passed=ev['validation']['PSNR_nonregression_gate'] and ev['validation']['parent_nonregression_gate'] and manual['dynamic_completeness_and_cleanup_passed']
    status='completed_pilot_visual_improvement' if passed else 'completed_not_promoted'
    lines=['# L5：表面连接、局部支撑厚度与观测完整性','',
        '3ca738e1在新测量前固定单候选，024e8a3a提交资格。两臂同精确J6完整状态与L4时序损失，各2000步原生1600，共4000正式更新；另各20步资格短训。原PIDG与冻结历史源码/权重不改，没有自动延长、阈值或权重搜索。','',
        '**约束实现**：可靠边沿当前中心方向的2-sigma椭球支撑间隙，加半父长度防塌缩；局部完整协方差与中心分布的最小主轴宽度保持父参照的0.8至1.25倍；两大主轴乘积的切向面积保持同区间，单点最大尺度上限1.25倍；实测根轨迹5×5足迹中原生全场深度排序的组贡献保持至少父参照0.8倍。四项平均总结构权重0.01，每步最多3组可见支撑。','',
        '厚度只是局部高斯及中心分布的支撑宽度，不是真实体厚或壁厚；缺乏可靠前后表面关联，不能给真实厚度标签。面积及贡献项保护已观测局部支撑，不补未见表面、不保证封闭。父参照可能保留错误碎片；不按GT选点。三维几何项不受整体刚体转动或相机运动影响，像素贡献使用各帧已知相机投影。','',
        f"初态{bank['local_patches']}片、{bank['edges']}条片内连接应用、{bank['rows']}个不同高斯行；按起始帧时间三段为{bank['temporal_thirds']}。275/15507约1.77%的动态行有初始结构覆盖，不能视为全鸭、全时段覆盖。原全局覆盖失败仍保留。所有原L3假设/未接受片保留，当前根/成员不可靠即暂停，两次矛盾撤销、两次通过恢复。结构只在原fit观测施加。",'',
        '两臂共同启用相同L4实测RGB时序记忆及250步刷新；候选另刷新表面资格，固定原提议分母。持久记忆22行、372个观测也偏早。奇数held不进入辅助目标，但仍参与普通RGB训练，不是独立验证。没有物体数、新颜色静态参考、语义、GT深度/法线或物理更新。J6沿用历史颜色条件父来源。','',
        '| 全30张val，原生1600 | J6父 | 时序对照 | 结构候选 |','|---|---:|---:|---:|']
    for title,key in [('PSNR(dB)','PSNR'),('SSIM','SSIM'),('L1','L1')]:
        lines.append('| '+title+' | '+' | '.join(f'{m[n][key]:.6f}' for n in NAMES)+' |')
    lines+=['',f"候选相对匹配对照{m['surface_support']['PSNR']-m['temporal_control']['PSNR']:+.6f}dB、相对J6{m['surface_support']['PSNR']-m['J6_parent']['PSNR']:+.6f}dB。原最大下降0.3dB门槛：对照{ev['validation']['PSNR_nonregression_gate']}、父{ev['validation']['parent_nonregression_gate']}。单次匹配未证明跨CUDA轨迹稳定因果。",'',
        '| 固定初始片/帧分母，自身参照代理 | J6父 | 时序对照 | 结构候选 |','|---|---:|---:|---:|']
    for title,key in [('全部初始片帧','all_original_patch_frames'),('父贡献已知','known_parent_coverage'),('贡献缺失','missing_coverage'),('连接惩罚均值','mean_connection_penalty'),('厚度超区间片帧','thickness_outside_interval'),('面积超区间片帧','area_outside_interval'),('贡献低于0.8倍片帧','coverage_below_lower'),('贡献比例均值','mean_coverage_ratio')]:
        lines.append('| '+title+' | '+' | '.join(str(sd[n][key]) for n in NAMES)+' |')
    lines+=['','固定原始片/fit帧全部评估，不按最后活跃子集筛选。360片帧中359个父贡献可计算，1个父贡献未知；未知不当零误差。连接间隙惩罚相对对照约下降58.92%，面积越界19→13、支撑不足96→67，是有限局部改善；厚度越界2→3没有改善。这里是软约束，不保证每片都满足边界。支撑谱、连接、面积、D/alpha与渲染贡献均不是独立GT，父模型自身满足区间并不能证明正确几何。','',
        '| 实际约束应用 | 时序对照 | 结构候选 |','|---|---:|---:|']
    for title,key in [('时序使用步数','steps_with_memory'),('时序端点对','endpoint_pair_applications'),('结构使用步数','steps_with_structure'),('结构片应用','structural_patch_applications'),('连接应用','structural_edge_applications'),('足迹组应用','native_coverage_group_applications'),('加权结构损失均值','mean_weighted_structure_loss')]:
        lines.append('| '+title+' | '+' | '.join(str(ca['arms'][n][key]) for n in NAMES[1:])+' |')
    lines+=['','实际人工复核：'+manual['conclusion']+'。','']+['- '+q for q in manual['findings']]
    lines+=['','27项相关测试、原生flags-off RGB/alpha/D精确及原梯度容差、异常RNG/flags恢复、有限非零canonical/grid/deform/scale梯度通过。两臂20步短训与四独立进程完整模型/两Adam/绑定/RNG/相机/physics/控制器恢复和固定原生梯度精确Adam回放通过；保留原40步CUDA非确定性失败记录，原epsilon1e-15不改。',
        '首次准备在单个足迹缺失时中止，未作优化更新；第二次以未知记录缺失足迹，并添加测试。原失败与修复说明保留，未改变门槛或冻结训练。14次原生刷新与4000相机轨迹、实际应用量独立重放通过；动态Adam更新、静态/physics模型及其Adam状态与缓存冻结均核验通过。PLY全部高斯字段及世界XYZ、180图、两段H264各200帧完整解码和30图指标核算通过。人工查看32张图，未实测浏览器播放或逐帧人工播放。报告收尾脚本一处引号语法错误已独立修复，原失败记录保留，未改变训练或评估。',
        '静态4126点与Adam完全冻结；动态15507统一池26k→28k，拓扑不变；physics模型/Adam/缓存冻结。本轮是RGB及局部结构训练，不能叫物理续训。未知/遮挡保持未知，不改叫背景或动态，也不按暗色/残差删点。','',
        '状态：'+status+'。没有独立动态三维GT；全图PSNR、局部支撑保持或变暗都不能证明几何完整。保留完整新权重及结果，当前模型仍为J6；不自动延长本轮或启动下一候选。']
    (V/'assessment.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    result=dict(status=status,pending=False,pending_review=False,optimizer_updates=4000,each_arm_updates=2000,
        current_model_replaced=False,manual_completeness_cleanup_passed=manual['dynamic_completeness_and_cleanup_passed'],
        RGB_nonregression_passed=ev['validation']['PSNR_nonregression_gate'],parent_nonregression_passed=ev['validation']['parent_nonregression_gate'],
        independent_geometry_GT=False,true_physical_thickness_known=False,historical_color_parent_disclosed=True,automatic_extension=False)
    artifacts=['assessment.md','manual_review.json','evaluation.json','videos.json','surface_bank.json','source_freeze.json','evaluation_freeze.json',
               'native_check.json','recovery.json','unit_tests.json','initial_health.json','initial_coverage.json','independent_audit.json','controller_audit.json','structural_diagnostic.json']
    write(V/'final_review.json',dict(**result,artifacts_sha256={n:sha(V/n) for n in artifacts}))
    for path in [V/'final_status.json',R/'final_status.json',R/'active.json',R/'pipeline.json']:write(path,result)
    for arm in NAMES[1:]:write(R/'formal'/arm/'final_status.json',dict(status='completed_reviewed',iteration=2000,pending=False,pending_review=False,trial_status=status,checkpoint_sha256=sha(R/'formal'/arm/'checkpoint.pt')))
    entry=f"\n- Matched formal4000 updates,14 native controller replays, all30 native1600 validation, fixed86/188 RGB/alpha/D, fixed original-patch structural diagnostics, PLY, two full200-frame H264 decodes and32 actual image reviews completed. PSNR control {m['temporal_control']['PSNR']:.6f}, candidate {m['surface_support']['PSNR']:.6f}; {status}. See v1/assessment.md. No automatic extension.\n"
    if entry not in (B/'progress.md').read_text(encoding='utf-8'):
        with (B/'progress.md').open('a',encoding='utf-8',newline='\n') as f:f.write(entry)
    print('L5_FINAL_REVIEW',result,flush=True)

if __name__=='__main__':main()
