"""Write measured assessment and close only after complete artifact/manual review."""
from pathlib import Path
from variants.staged_reconstruction.i10_tracking.io import ROOT,read,write,sha

B=Path(__file__).parent;V=B/'v1';R=ROOT/'runs/pidg_staged_reconstruction/RubberDuck_temporal_memory/v1'

def main():
    for name in ['unit_tests.json','native_check.json','recovery.json','independent_input_audit.json','independent_audit.json','controller_audit.json','title_review.json','disabled_cli.json']:
        assert read(V/name)['passed'],name
    ev=read(V/'evaluation.json');manual=read(V/'manual_review.json');ca=read(V/'controller_audit.json')
    assert manual['actual_visual_review'] and manual['fixed_native_RGB_alpha_D_reviewed']
    m=ev['validation']['means'];summary=read(V/'bank.json')['summary'];health=read(V/'initial_health.json')['summary']
    scope=read(V/'fusion_scope.json')
    passed=ev['validation']['PSNR_nonregression_gate'] and ev['validation']['parent_nonregression_gate'] and manual['dynamic_completeness_and_cleanup_passed']
    status='completed_pilot_visual_improvement' if passed else 'completed_not_promoted'
    lines=['# L4：跨帧表面记忆与运动补偿融合原型','',
        '用户授权进一步尝试显式时序融合。aaf46932在候选输出前固定单协议；15003145提交资格，59feb90d仅修正Git自动换行的原始字节保存，失败记录保留。训练源码、输入、权重和门槛没有因Git修复改变。','',
        '沿用精确J6完整模型/两个Adam/RNG/相机/physics起点，两臂各2000步原生1600训练，共4000正式更新，另各20步短训。原PIDG、历史失败和旧40步CUDA轨迹非确定性记录保持不变。','',
        f"原730条双向RGB片段完整保留为{summary['memories']}个记忆单元，{summary['fit_links']}条fit重叠连接形成{summary['long_memories']}条长记忆；跨度16帧61条、20帧14条。{summary['root_nominated_long_memories']}条有合格T4根提议，{summary['geometric_bindings']}条通过几何及原生完整深度排序可见性，涉及{health['active_rows']}个canonical行和{health['active_observations']}个融合观测。全部8760个原始观测已核算，无删片段或插值遮挡目标。",'',
        f"合格观测按时间三段为{health['active_frame_thirds']}。后段没有合格长记忆；这是早期局部16至20帧原型，未完成全时段表面融合，也未证明全表面已观测。原全局覆盖失败门槛没有改判。",'',
        f"事后只读范围核算：{scope['memories_with_additional_observations']}条记忆包含原T4根窗口之外的{scope['additional_observations_outside_T4_root_windows']}个额外实测帧观测；总融合观测{scope['all_bound_observations']}个。其余为已有窗口内的重复或原观测。该核算没有选择训练点或反馈训练，新增观测不等于新增独立表面面积。",'',
        '成员只用偶数fit重叠，互为最近、RMS<=0.75px、最大<=1.5px和歧义差>=0.25px；奇数held验证，不挑替代连接。原RGB双向追踪是数据推断，T4 canonical归属/可见性仍依赖父模型，不是独立3D或物理身份。遮挡/歧义保持未知，不把未知叫背景。','',
        '显式记忆保留原RGB的轨迹与重复观测来源；候选每步检索最多4个实际前后fit关键帧，对同一canonical点的当前投影、关键帧投影、两帧投影位移联合反传，权重0.01。全部目标为实测RGB坐标，没有父长度、伪深度或未见视角伪图像。原PIDG多时间形变调用共享canonical参数，普通全图RGB仍为0.8L1+0.2DSSIM。','',
        '每250步原生复核记忆；未知暂停、两次矛盾撤销、两次通过恢复。记忆保留但不在未观测帧施加约束。纯RGB对照附加损失精确为零。未新增Transformer/RNN、增点或独立的新表面三角化，不能以局部重投影称完整重建。','',
        '| 全30张val，原生1600 | J6父 | 纯RGB续训 | 时序记忆 |','|---|---:|---:|---:|']
    for title,key in [('PSNR(dB)','PSNR'),('SSIM','SSIM'),('L1','L1')]:
        lines.append('| '+title+' | '+' | '.join(f"{m[n][key]:.6f}" for n in ['J6_parent','rgb_control','temporal_memory'])+' |')
    lines+=['',f"时序记忆相对匹配对照{m['temporal_memory']['PSNR']-m['rgb_control']['PSNR']:+.6f}dB，相对J6{m['temporal_memory']['PSNR']-m['J6_parent']['PSNR']:+.6f}dB。原最大下降0.3dB两项门槛：{ev['validation']['PSNR_nonregression_gate']}/{ev['validation']['parent_nonregression_gate']}；单次匹配不代表跨随机轨迹稳定因果。",'',
        '| 实际约束使用 | 纯RGB | 时序记忆 |','|---|---:|---:|']
    for title,key in [('使用记忆的步数','steps_with_memory'),('关键帧对累计应用','total_endpoint_pairs'),('记忆累计应用','total_memory_applications'),('加权记忆损失均值','mean_weighted_memory_loss')]:
        lines.append('| '+title+' | '+' | '.join(str(ca['arms'][n][key]) for n in ['rgb_control','temporal_memory'])+' |')
    lines+=['','| 完成更新 | 活跃记忆 | 行数 | 观测 | 暂停 | 撤销 |','|---|---:|---:|---:|---:|---:|']
    for q in ca['arms']['temporal_memory']['final_history']:
        lines.append('| '+' | '.join(str(q[k]) for k in ['completed_updates','active_memories','active_rows','active_observations','suspended','revoked'])+' |')
    lines+=['','原绑定22条长记忆全部保留的重投影诊断（不是独立GT，held仍参与普通RGB）：']
    for name,q in ev['training_track_diagnostic']['models'].items():
        lines.append(f"- {name}：fit{q['fit_mean_px']:.6f}px，held{q['held_mean_px']:.6f}px，缺失{q['missing']}/{q['total']}。")
    lines+=['','固定86/188时间2/100/200暗RGB且高alpha的动态像素，仅是渲染代理：','',
        '| 相机/时间 | J6 | 纯RGB | 时序记忆 |','|---|---:|---:|---:|']
    proxies=ev['diagnostics']['high_alpha_proxies']
    for camera in [86,188]:
        for time in [2,100,200]:
            values=[next(q for q in proxies if q['camera']==camera and q['time']==time and q['model']==n) for n in ['J6_parent','rgb_control','temporal_memory']]
            lines.append(f'| {camera}/{time} | '+' | '.join(f"{q['dynamic_dark_high_alpha']}/{q['pixels']}" for q in values)+' |')
    lines+=['','上述代理定义为动态RGB所有通道<0.05且alpha>0.5，仅供评估，不用于训练/选点/删点。低alpha黑色缺失不是零误差。','',
        '实际人工复核：'+manual['conclusion']+'。','']+['- '+q for q in manual['findings']]
    lines+=['','14项CPU测试、原生flags-off RGB/alpha/D精确与原梯度容差、全部记忆独立原生重放、异常RNG/flags恢复、canonical/grid/deform有限非零时序梯度和冻结分支无梯度通过。独立CPU全730/8760输入与fit/held隔离、两臂20步短训、四进程完整模型/双Adam/RNG/相机/physics/记忆及固定native梯度精确Adam恢复通过。',
        '全部七次刷新模型/证据独立原生重放，4000相机采样及实际使用、冻结状态、六完整Gaussian PLY/九世界XYZ、180图和30指标、两段H264各200帧完整解码核验通过。实际查看图像及SHA见manual_review.json；未实测浏览器播放。',
        '静态4126点/Adam与physics模型/Adam/缓存完全冻结，动态15507统一池26k→28k，无物体数、新颜色静态参考/语义/GT深度法线或物理更新。J6历史颜色条件来源仍披露。D/alpha不是独立GT，黑色低alpha为缺失，PSNR/连接数/轨迹一致性不能证明正确三维几何。','',
        '状态：'+status+'。保留完整结果，不自动延长、换候选或扫阈值/权重；保留J6为当前模型。']
    (V/'assessment.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    result=dict(status=status,pending=False,pending_review=False,optimizer_updates=4000,each_arm_updates=2000,
        current_model_replaced=False,manual_completeness_cleanup_passed=manual['dynamic_completeness_and_cleanup_passed'],
        RGB_nonregression_passed=ev['validation']['PSNR_nonregression_gate'],parent_nonregression_passed=ev['validation']['parent_nonregression_gate'],
        independent_geometry_GT=False,historical_color_parent_disclosed=True,automatic_extension=False)
    artifacts=['assessment.md','manual_review.json','evaluation.json','videos.json','bank.json','source_freeze.json','evaluation_freeze.json',
        'native_check.json','recovery.json','unit_tests.json','initial_health.json','initial_coverage.json','independent_input_audit.json',
        'independent_audit.json','controller_audit.json','title_review.json','disabled_cli.json','memory_figures.json','git_storage_repair.json','git_validation.json','fusion_scope.json']
    write(V/'final_review.json',dict(**result,artifacts_sha256={n:sha(V/n) for n in artifacts}))
    for path in [V/'final_status.json',R/'final_status.json',R/'active.json',R/'pipeline.json']:write(path,result)
    for arm in ['rgb_control','temporal_memory']:
        write(R/'formal'/arm/'final_status.json',dict(status='completed_reviewed',iteration=2000,pending=False,pending_review=False,
            trial_status=status,checkpoint_sha256=sha(R/'formal'/arm/'checkpoint.pt')))
    write(V/'final_validation.json',dict(passed=True,all_required_review_complete=True,actual_models_updated=True,
        completion_source_sha256=sha(Path(__file__)),final_review_sha256=sha(V/'final_review.json')))

if __name__=='__main__':main()
