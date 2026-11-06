"""Close L6 from saved independent results and actual manual review only."""
from pathlib import Path
from variants.staged_reconstruction.i10_tracking.io import ROOT, read, write, sha

B=Path(__file__).parent;V=B/'v1';R=ROOT/'runs/pidg_staged_reconstruction/RubberDuck_revocable_springs/v1'
NAMES=['J6_parent','temporal_control','revocable_spring']


def main():
    required=['unit_tests.json','native_check.json','recovery.json','initial_coverage.json',
              'independent_audit.json','controller_audit.json','structural_diagnostic.json','spring_reference_audit.json']
    for name in required:assert read(V/name)['passed'],name
    ev=read(V/'evaluation.json');manual=read(V/'manual_review.json');ca=read(V/'controller_audit.json')
    assert manual['actual_visual_review'] and manual['fixed_native_RGB_alpha_D_reviewed']
    assert len(manual['reviewed_images_sha256'])==32
    for path,h in manual['reviewed_images_sha256'].items():assert sha(ROOT/path)==h,path
    sd=read(V/'structural_diagnostic.json')['models'];m=ev['validation']['means'];bank=read(V/'surface_bank.json')['summary']
    passed=ev['validation']['PSNR_nonregression_gate'] and ev['validation']['parent_nonregression_gate'] and manual['dynamic_completeness_and_cleanup_passed']
    status='completed_pilot_visual_improvement' if passed else 'completed_not_promoted'
    lines=['# L6：可靠表面节点与可撤销共享固定刚度弹簧','',
        'dcac0409在新测量前预声明。两臂同J6完整模型/双Adam/参数绑定/RNG/相机/physics图，各2000步原生1600，共4000正式更新，另各20步资格短训。两臂共同L4时序记忆；候选只加弹簧几何，不叠加L5厚度/面积/贡献保持或阻尼。','',
        '弹簧能量为sqrt(1+strain²)-1，strain=(当前世界边长-固定参考)/固定参考；共享固定k=1、总权重0.01，片内边均值再以固定初始适用片分母归一化。参考为原6fit帧J6长度中位数，不是真实物质静止长度。置信度停止梯度。每250步复核原生自身可见贡献/RGB/投影及原L3图资格，未知暂停、两矛盾撤销、两通过恢复；刷新之间用缓存逐帧资格与当前根投影防护，非每步全量原生复核。','',
        f"初态{bank['local_patches']}片、{bank['edges']}片内边应用、{bank['rows']}不同动态行；起始时间三段{bank['temporal_thirds']}。覆盖偏早，行数不是表面积。长记忆22行/372坐标也偏早，held不收辅助目标但仍受普通RGB训练，不是独立GT。",'',
        '静态4126及Adam、physics模型/Adam/缓存精确冻结；动态15507统一未知数量池26k→28k，拓扑不变。无质量/本构/材料/接触/外力识别，不是新增物理续训。无新颜色静态阈值/语义/物体数/GT，父历史颜色条件来源披露。','',
        '| 全30val原生1600 | J6 | 时序对照 | 弹簧候选 |','|---|---:|---:|---:|']
    for title,key in [('PSNR(dB)','PSNR'),('SSIM','SSIM'),('L1','L1')]:
        lines.append('| '+title+' | '+' | '.join(f'{m[n][key]:.6f}' for n in NAMES)+' |')
    lines+=['',f"候选较对照{m['revocable_spring']['PSNR']-m['temporal_control']['PSNR']:+.6f}dB，较J6{m['revocable_spring']['PSNR']-m['J6_parent']['PSNR']:+.6f}dB。原两项最大下降0.3dB门槛分别{ev['validation']['PSNR_nonregression_gate']}/{ev['validation']['parent_nonregression_gate']}。单对CUDA轨迹不证明重复种子稳定因果。",'',
        '| 全初始360片帧、自身参照代理 | J6 | 对照 | 候选 |','|---|---:|---:|---:|']
    for title,key in [('弹簧能量均值','mean_spring_energy'),('归一化绝对应变均值','mean_abs_strain'),('连接间隙惩罚','mean_connection_penalty'),('厚度支撑越界片帧','thickness_outside_interval'),('面积越界片帧','area_outside_interval'),('父贡献已知','known_parent_coverage'),('当前贡献缺失','missing_coverage'),('贡献低于0.8片帧','coverage_below_lower'),('贡献比例均值','mean_coverage_ratio')]:
        lines.append('| '+title+' | '+' | '.join(str(sd[n][key]) for n in NAMES)+' |')
    lines+=['','厚度/面积/贡献仅是沿用L5的独立评估代码代理，本轮无对应损失；全初始分母保留，不按最终活跃片筛选。360中1个父贡献未知，已知分母359，不能当零误差。父参考及应变改善不能证明独立几何正确、真实体厚或完整物体。','',
        '| 实际应用 | 时序对照 | 弹簧候选 |','|---|---:|---:|']
    for title,key in [('有记忆步数','steps_with_memory'),('端点对','endpoint_pair_applications'),('有弹簧步数','steps_with_structure'),('片应用','structural_patch_applications'),('边应用','structural_edge_applications'),('平均加权弹簧损失','mean_weighted_structure_loss')]:
        lines.append('| '+title+' | '+' | '.join(str(ca['arms'][n][key]) for n in NAMES[1:])+' |')
    lines+=['','## 原始绑定记忆重投影（非独立GT）','',
        '| 模型 | fit均值px | held均值px | 缺失/372 |','|---|---:|---:|---:|']
    for n,q in ev['training_track_diagnostic']['models'].items():lines.append(f"| {n} | {q['fit_mean_px']:.6f} | {q['held_mean_px']:.6f} | {q['missing']} |")
    lines+=['','## 实际图像复核','',manual['conclusion'],'']+['- '+f for f in manual['findings']]
    lines+=['','## 验证与局限','',
        '28相关测试、默认关闭原生RGB/alpha/D精确及原梯度容差、孤立弹簧canonical/grid/deformation有限非零梯度、冻结分支无梯度、异常RNG/flags恢复通过。两臂20步短训、四独立恢复进程及同份native梯度精确Adam回放通过。原40步CUDA轨迹非确定性失败保留，epsilon仍1e-15。',
        '14个刷新模型、4000采样与使用量重放，双Adam更新与冻结分支/图完整核验，6完整Gaussian PLY所有字段及9世界XYZ诊断PLY、180图及30val均值、两H264各200帧完整解码通过；实际查看32图，未测浏览器播放或人工逐帧播放。固定前期/中期/晚期图仍无独立跨时间固定视角GT。',
        '首次准备因未激活仓库CUDA环境、PATH中缺少ninja中止，未优化；加载原enable_cuda.ps1后重试，保留失败日志。没有修改原PIDG、冻结源、历史权重或门槛。',
        'RGB/alpha/D/应变均不直接证明几何正确；黑色低alpha缺失不等于零误差。固定参考可能抑制真实形变、保存错误碎片；本轮没有增点、表面延拓、闭合或真实厚度证据。','',
        f'最终状态：{status}；全部新权重和记录保存，当前工作基线不自动替换，不自动延长/调权重/扫参。']
    (V/'assessment.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    result=dict(status=status,pending=False,pending_review=False,optimizer_updates=4000,each_arm_updates=2000,current_model_replaced=False,
        manual_completeness_cleanup_passed=manual['dynamic_completeness_and_cleanup_passed'],RGB_nonregression_passed=ev['validation']['PSNR_nonregression_gate'],
        parent_nonregression_passed=ev['validation']['parent_nonregression_gate'],independent_geometry_GT=False,true_physical_thickness_known=False,
        historical_color_parent_disclosed=True,automatic_extension=False)
    artifacts=['assessment.md','manual_review.json','evaluation.json','videos.json','surface_bank.json','source_freeze.json','evaluation_freeze.json',
        'native_check.json','recovery.json','unit_tests.json','initial_health.json','initial_coverage.json','independent_audit.json','controller_audit.json',
        'structural_diagnostic.json','spring_reference_audit.json']
    write(V/'final_review.json',dict(**result,artifacts_sha256={n:sha(V/n) for n in artifacts}))
    for path in [V/'final_status.json',R/'final_status.json',R/'active.json',R/'pipeline.json']:write(path,result)
    for arm in NAMES[1:]:write(R/'formal'/arm/'final_status.json',dict(status='completed_reviewed',iteration=2000,pending=False,pending_review=False,trial_status=status,checkpoint_sha256=sha(R/'formal'/arm/'checkpoint.pt')))
    with (B/'progress.md').open('a',encoding='utf-8',newline='\n') as f:f.write(f'\n- Full4000 updates, native evaluation/controller/spring-reference audit, PLY/videos and32 actual image reviews complete. {status}; no automatic extension. See v1/assessment.md.\n')
    print('L6_FINAL_REVIEW',result,flush=True)


if __name__=='__main__':main()
