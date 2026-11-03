"""Measured final report; refuses to substitute planned stages for actual runs."""
import hashlib
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from variants.staged_reconstruction.prepare_data import REPORT,RUNS,read,write,sha


def main():
    s1=read(REPORT/'S1_final_verification.json');s2=read(REPORT/'S2_geometry_recovery/final_verification.json')
    rows=[{'stage':'S1','updates':s1['iteration'],'dynamic_PSNR':None,'dynamic_alpha':None,
        'static_PSNR':s1['validation']['candidate_static_PSNR'],'track_px400':None,'conditional_interior':None,
        'checkpoint_sha256':s1['checkpoint_sha256'],'scope':'Conditional static initialization'},
        {'stage':'S2','updates':s2['iteration'],'dynamic_PSNR':s2['native_validation']['dynamic_PSNR'],
        'dynamic_alpha':s2['native_validation']['dynamic_alpha'],'static_PSNR':s2['native_validation']['static_PSNR'],
        'track_px400':read(REPORT/'S3/binding_baseline_audit.json')['track_observation_mean_px400'],
        'track_value_note':'Baseline tracks evaluated on S3 parent, not S2; excluded from plots',
        'conditional_interior':None,'checkpoint_sha256':s2['checkpoint_sha256'],
        'scope':'Third16k attempt; two earlier failures retained; novel-view geometry unresolved'}]
    for stage in ['S3','S4','S5','S6']:
        result=read(REPORT/stage/'final_verification.json');protocol=read(REPORT/stage/'protocol.json')
        assert result['iteration']==protocol['updates'] and result['native_images_reviewed']
        assert result['videos_independently_decoded'] and result['admitted_for_conditional_continuation']
        assert sha(RUNS/stage/'formal/checkpoint.pt')==result['checkpoint_sha256']
        m=result['native_validation']
        rows.append({'stage':stage,'updates':result['iteration'],'dynamic_PSNR':m['dynamic_PSNR'],
            'dynamic_alpha':m['dynamic_alpha'],'static_PSNR':m['static_PSNR'],
            'track_px400':result['track_observation_mean_px400'],'conditional_interior':result['conditional_interior'],
            'whole_run_conditional_interior':result.get('training_conditional_interior'),
            'checkpoint_sha256':result['checkpoint_sha256'],'scope':'Conditional reconstruction curriculum; physical geometry/identity unknown'})
    baseline=read(REPORT/'S3/binding_baseline_audit.json');rows[2]['track_px400']=baseline['track_observation_mean_px400']
    rows[1]['track_px400']=None;rows[1].pop('track_value_note')
    assert [r['updates'] for r in rows]==[12000,16000,6000,4000,4000,4000]
    bindings=read(REPORT/'S4/bindings.json')
    quality={}
    for r in rows[1:]:
        folder='S2_geometry_recovery' if r['stage']=='S2' else r['stage']
        value=read(REPORT/folder/'native_full_quality.json');assert value['checkpoint_sha256']==r['checkpoint_sha256']
        quality[r['stage']]={split:item['metrics'] for split,item in value['splits'].items()}
    summary={'authorized_reconstruction_curriculum_training_and_review_completed':True,
        'complete_geometry_or_physical_inference_success':False,'original_full_contact_scope_completed':False,
        'stages':rows,'standard_native_full_quality':quality,
        'accepted_lineage_updates':sum(r['updates'] for r in rows),'failed_S2_additional_updates':32000,
        'sparse_tracklets':len(bindings['bindings']),'sparse_observations':sum(len(x['frames']) for x in bindings['bindings']),
        'physical_ID_after_ambiguity':'unknown','certified_contact_surface':False,'acquisition_seconds_available':False,
        'contact_force_friction_restitution_enabled':False,'object_object_nonpenetration_enabled':False,'unknown_force_or_constitutive_enabled':False,
        'sequential_change_not_causal_ablation':{
            'S2_to_S6_dynamic_PSNR_dB':rows[5]['dynamic_PSNR']-rows[1]['dynamic_PSNR'],
            'S3_to_S6_qualified_track_error_relative_reduction':1-rows[5]['track_px400']/rows[2]['track_px400']},
        'test_suite':read(REPORT/'S3/test_receipt.json'),
        'limits':['Fixed-view sheets/fragments remain; original-camera image fit is not complete geometry',
            'Sparse flows and row bindings come from training images, not independent material/3D GT',
            'Physics residuals are internal sequence-normalized diagnostics, not calibrated stress or force truth',
            'S5 conditional interior prior does not certify collision; no eligible/active violations means no demonstrated correction',
            'Sequential checkpoints are not a matched causal ablation or legacy PIDG camera comparison']}
    write(REPORT/'final_summary.json',summary)
    def f(value,precision=3):return '—' if value is None else f'{value:.{precision}f}'
    text=['# S1–S6 实测报告', '',
        '修订后的条件重建范围内，六阶段训练、原生评估、完整恢复审计与图像复核已执行。**完整三维几何和物理身份仍未验证成功，原方案完整接触阶段尚未完成**；不能把阶段训练完成表述为物理建模成功。', '',
        '[图像/视频合集](../../output/staged_reconstruction/index.html) · [实测趋势图](measured_curriculum.png) · [修订计划与未完成依赖](../staged_reconstruction_plan.md)', '',
        '| 阶段 | 正式更新 | 验证动态 PSNR | 动态 alpha | 静态 PSNR | 轨迹误差 px@400 |',
        '|---|---:|---:|---:|---:|---:|']
    for r in rows:text.append(f"| {r['stage']} | {r['updates']} | {f(r['dynamic_PSNR'])} | {f(r['dynamic_alpha'])} | {f(r['static_PSNR'])} | {f(r['track_px400'])} |")
    text += ['', '原生全画幅标准图像指标（8bit RGB协议，黑色背景会稀释局部误差；不是独立几何指标）：', '',
        '| 阶段 | 验证 PSNR | 验证 SSIM | 验证 VGG LPIPS | 测试 VGG LPIPS |',
        '|---|---:|---:|---:|---:|']
    for stage,value in quality.items():
        v=value['val'];t=value['test']
        text.append(f"| {stage} | {f(v['PSNR'])} | {f(v['SSIM'],5)} | {f(v['LPIPS'],5)} | {f(t['LPIPS'],5)} |")
    text += ['',f"条件继承链共 {summary['accepted_lineage_updates']} 次更新；另保留两次失败 S2 的32,000次更新与结果。91条合格轨迹、455次观测来自训练图像，未知身份保持未知。各阶段短训分别检验40→80步完整恢复。", '',
        f"S4 轨迹误差由 S3 的{rows[2]['track_px400']:.6f}升至{rows[3]['track_px400']:.6f}px@400，属于负结果；S6 为{rows[5]['track_px400']:.6f}px@400。完整误差趋势必须与 RGB 指标同时保留。", '',
        f"S6 相对条件 S2 终态的动态图像 PSNR 提高{summary['sequential_change_not_causal_ablation']['S2_to_S6_dynamic_PSNR_dB']:.3f}dB；455次合格训练图像轨迹误差相对 S3 基线降低{100*summary['sequential_change_not_causal_ablation']['S3_to_S6_qualified_track_error_relative_reduction']:.2f}%。这是继承训练链的实测变化，不能归因于某一单独损失，也不能外推到未观测材料点或跨遮挡身份。", '',
        'S1 提供条件静态初始化；S2 数值门槛通过但固定视角仍碎裂；S3 加入原 PIDG velocity/stress/CMR；S4 使用合格固定行轨迹；S5 使用已声明的静态深内部先验；S6 联合已有目标并解冻静态外观，静态几何仍固定。', '',
        '**S5 仅执行条件静态体非穿透；原方案的物体间非穿透、接触力、摩擦和恢复系数未获真实场景验证**：现有物体表面、物理身份和秒时钟未获认证。其受控球/平面300步验证只验证算子与优化器，不能代替 Rubber Duck 接触证据。可见点全部位于条件体外不能排除被遮挡区域的真实穿透。具体有效/激活样本在 S5/assessment.md 和 final_verification.json。', '',
        *[f"{r['stage']} 全程条件体先验：{r['whole_run_conditional_interior']['objective_calls']}次调用，{r['whole_run_conditional_interior']['eligible_observation_instances']}次合格观测实例，{r['whole_run_conditional_interior']['active_violation_instances']}次激活违规，{r['whole_run_conditional_interior']['nonzero_prior_objective_calls']}次非零先验。" for r in rows[4:]], '',
        '实际查看各阶段九个观测/组贡献面板，以及固定86、固定188、原轨迹在2/86/188的原生图像/像素裁剪。每阶段三段视频均独立解码200帧；没有声称人工看完所有视频帧。新视角仍有片状碎裂与空洞，晚期物理身份仍未知。', '',
        '静态/动态 alpha、训练图像轨迹误差、内部动量残差和条件 SDF 都是代理量；没有独立三维、材料点、应力或力真值。后续研究应先解决三维几何与持续身份证据，不自动增加损失权重、扫参或加入自由力。', '',
        '逐阶段检查点哈希、输入/源冻结、采样重放、门槛、评估图像哈希及视觉结论见各阶段 final_verification.json、manual_visual_review.json 和 assessment.md。', '',
        '最终全链校验见 [completion_integrity.json](completion_integrity.json)：冻结文件、六个终态检查点、人工复核图像和十五段视频均核对；任务训练/评估进程已退出。后续先补可校准的RGB合成几何/身份证据，完整接触目标继续保留，不自动开始新一轮训练。']
    (REPORT/'final_summary.md').write_text('\n'.join(text)+'\n',encoding='utf8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,3.6),constrained_layout=True)
    axes[0].plot(list(range(1,len(rows))),[r['dynamic_PSNR'] for r in rows[1:]],'o-',label='dynamic image region')
    axes[0].plot(list(range(len(rows))),[r['static_PSNR'] for r in rows],'s-',label='conditional static region')
    axes[0].set_xticks(list(range(len(rows))),[r['stage'] for r in rows])
    axes[0].set_ylabel('Native validation PSNR (dB)');axes[0].legend(fontsize=8);axes[0].grid(alpha=.2)
    axes[1].plot([r['stage'] for r in rows[2:]],[r['track_px400'] for r in rows[2:]],'o-',color='#a55324')
    axes[1].set_ylabel('Qualified training-track error (px @400)');axes[1].grid(alpha=.2)
    fig.suptitle('Sequential final checkpoints; image/track proxies, not geometry or physical GT',fontsize=10)
    fig.savefig(REPORT/'measured_curriculum.png',dpi=180);plt.close(fig)
    out=ROOT/'output/staged_reconstruction';cards=[]
    for stage,folder in [('S2','S2_geometry_recovery'),('S3','S3'),('S4','S4'),('S5','S5'),('S6','S6')]:
        step=16000 if stage=='S2' else 6000 if stage=='S3' else 4000
        for view in ['fixed86','fixed188','original']:
            assert (out/folder/f'{view}.browser.mp4').is_file(), 'Run prepare_browser_videos.py before building the gallery'
        cards.append(f'<section><h2>{stage}</h2><p>Conditional reconstruction; novel-view geometry and late physical IDs remain unresolved.</p><a href="{folder}/{stage}_review_step{step:05d}.png"><img src="{folder}/{stage}_review_step{step:05d}.png" alt="observed RGB and group contributions"></a><div>'+''.join(f'<figure><figcaption>{view};20 display fps, acquisition seconds unknown</figcaption><video controls playsinline preload="metadata"><source src="{folder}/{view}.browser.mp4" type="video/mp4"></video><a href="{folder}/{view}.browser.mp4">打开视频</a></figure>' for view in ['fixed86','fixed188','original'])+'</div></section>')
    comparison=out/'baseline_comparison/index.html'
    comparison_link='<section><h2>C1_repaired 参考模型对比</h2><p><a href="baseline_comparison/index.html">查看 C1 对 S6 并排视频，以及 C1、S2–S6 六模型总览</a></p><p>包含固定86、固定188与原相机轨迹。C1 与本轮相机约定、模型容量和训练预算不同；用于诊断展示，不能作为严格消融。</p></section>' if comparison.is_file() else ''
    html='<!doctype html><meta charset="utf-8"><title>S1–S6 measured reconstruction review</title><style>body{font:16px system-ui;max-width:1200px;margin:40px auto;padding:20px;background:#12151c;color:#eee}section{margin:40px 0;padding:20px;background:#1c222d}img{max-width:100%;max-height:650px}section div{display:flex;flex-wrap:wrap}figure{margin:10px;max-width:340px}video{width:100%}a{color:#9cf}</style><h1>S1–S6 measured review</h1><p>Training and review completion is not complete-geometry or physical-inference success. Fixed-view fragments and unknown late identities remain. No contact force/friction/restitution inference.</p>'+comparison_link+''.join(cards)
    (out/'index.html').write_text(html,encoding='utf8')
    print('ALL_AUTHORIZED_STAGED_RUNS_REPORTED',summary['accepted_lineage_updates'])


if __name__=='__main__':main()
