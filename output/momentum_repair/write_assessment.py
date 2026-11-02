"""Produce final evidence-based assessment from verified immutable outputs."""
import json
from pathlib import Path
import sys
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from variants.metric_tuning.run import read_json as read,write_json as write,sha256 as sha
from variants.momentum_guard.run import assert_frozen
REPORTS=ROOT/'experiments/momentum_repair';OUT=Path(__file__).parent
assert_frozen(ROOT/'variants/momentum_guard/code')
r=read(REPORTS/'results.json');fixed=read(REPORTS/'fixed_field_probes.json');audit=read(REPORTS/'reproducibility_audit.json')
verification=read(REPORTS/'final_verification.json');assert verification['status']=='passed'
best=r['metrics']['ours_best'];control=r['control_metrics']['ours_best'];sel=r['selection'];cs=r['control_selection']
mm=r['motion'][0]['metrics'];prior=next(v['metrics'] for v in read(ROOT/'experiments/unified_flow/motion_validation.json')['results'] if v['model']=='L1_shared_stress' and v['snapshot']=='best')
lines=['# 动量梯度修正完整结果：M1 对照 L1','',
'**结论：重建质量明显恢复，观测二维运动有所改善；物理动量一致性未解决。M1可保留为重建候选，不能称为更可靠的物理模型。**',
'',f"M1完成总50k，验证PSNR选择49k。相对L1：验证PSNR +{sel['value']-cs['value']:.4f} dB，测试PSNR +{best['PSNR']-control['PSNR']:.4f} dB；独立二维速度推进EPE降低8.99%。预先综合标准要求速度EPE至少降低10%，因此综合收益判定仍为false，不事后降低门槛。",'',
'## 1. 实验与因果边界','',
'同一Rubber Duck场景、seed0、120训练/30验证/50测试，相同完整扩展3k状态训练到总50k。3k以后新增47k步，无前缀额外拟合，没有新增模型参数、初始化、优化器状态或外力/本构/材料未知量。共享stress头继承前一轮相同seed20261102、std0.001、bias0的初始化与共同完整状态。',
'',
'网络、Grid4D、轨迹差分步长、raw光流数据、RGB/光流/动量原权重与频率、学习率保持原配方。增加momentum_gradient_guard开关与0.1比例：将共享Grid4D和形变trunk/warp各组的加权动量梯度上限设为同批RGB目标梯度的10%，私有stress头及原应变率系数force_sigma梯度完整保留。前向报告真实物理损失，反向是显式一阶代理策略，不等价于只改一个标量损失权重。限制的是动量梯度分量，不是Adam总参数更新。',
'',
'复现审计确认：fork仅改训练器runtime导入；模型配置字节相同，metric配置仅增加新开关/比例，训练opt仅诊断目录不同；相同完整起点、相机顺序、光流帧日程和最终Python/CPU Torch RNG。关闭开关和首个物理事件前的wrapper均返回原tensor、不额外消耗Torch CPU RNG。',
'',
'**但不能将全部收益严格归因于新策略。** 在首个动量事件8100以前，4k–8k验证PSNR已存在约+0.26～+0.65dB差异；最终CUDA RNG也不同。GPU非确定性/稠密化的放大是可能解释，目前未证明它是唯一原因。相同种子/起点不等于逐位相同运行；本轮为单场景单种子受控配方对照，未额外重训来掩盖该限制。证据见reproducibility_audit.json。',
'',
'## 2. 图像质量','',
'| 数据/快照 | L1 | M1 |', '|---|---:|---:|',
f"| 最佳验证PSNR | {cs['value']:.6f} | {sel['value']:.6f} |",
f"| 最佳验证LPIPS | {cs['metrics']['LPIPS']:.8f} | {sel['metrics']['LPIPS']:.8f} |",
f"| 测试best PSNR | {control['PSNR']:.6f} | {best['PSNR']:.6f} |",
f"| 测试best SSIM | {control['SSIM']:.8f} | {best['SSIM']:.8f} |",
f"| 测试best LPIPS | {control['LPIPS']:.8f} | {best['LPIPS']:.8f} |",
f"| 测试best MS-SSIM | {control['MS-SSIM']:.8f} | {best['MS-SSIM']:.8f} |",
f"| 测试final PSNR | {r['control_metrics']['ours_50000']['PSNR']:.6f} | {r['metrics']['ours_50000']['PSNR']:.6f} |",'',
f"验证LPIPS降低{(1-sel['metrics']['LPIPS']/cs['metrics']['LPIPS'])*100:.2f}%，测试best LPIPS降低{(1-best['LPIPS']/control['LPIPS'])*100:.2f}%。best49k和final50k测试PSNR相差仅约0.0064dB。验证先锁定，随后才打开新测试；测试此前有历史暴露，不能当作从未接触的全新测试。",'',
'人工查看原帧1/85/101/197的PNG与解码视频：M1平台边缘、接触区域与鸭子轮廓较清楚；原帧85中L1漏掉的悬空鸭子在M1基本恢复，但仍有小块漂浮重影和局部接触形变误差。原帧101仍有局部模糊，晚期两组均较接近GT。完整50帧视频已解码成逐帧总览检查，不能据此声称完全无闪烁或几何真实。',
'',
'## 3. 独立二维观测运动','',
'使用冻结V2验证源帧→后继训练帧的raw UniMatch估计光流，400×225，未用于本轮训练/选择，无H3估计深度。',
'', '| 指标（像素，越低越好） | L1 best | M1 best | 改善 |','|---|---:|---:|---:|',
f"| 完整轨迹EPE | {prior['geometric_EPE_px']:.6f} | {mm['geometric_EPE_px']:.6f} | {(1-mm['geometric_EPE_px']/prior['geometric_EPE_px'])*100:.2f}% |",
f"| 速度推进EPE | {prior['velocity_EPE_px']:.6f} | {mm['velocity_EPE_px']:.6f} | {(1-mm['velocity_EPE_px']/prior['velocity_EPE_px'])*100:.2f}% |",'',
'这支持可观测图像运动有所改善，但速度推进改善8.99%仍未达到预先10%门槛；无统计重复、3D速度或应力真值，不能推断真实物理速度更准。完整轨迹导出速度使内部速度偏差为零是定义，不是额外实验胜利。',
'',
'## 4. 真实动量残差和剩余瓶颈','',
'没有以小梯度或代理前向掩盖残差。各模型实际最终Gaussian点集上的三个固定训练时刻未缩放Huber残差如下（best快照，诊断采样日程固定在iteration50000，不是重新训练）：','',
'| 原训练帧 | L1 raw Huber | M1 raw Huber |','|---|---:|---:|']
probes=read(REPORTS/'unweighted_momentum.json')['probes']
for fid in [2,83,155]:
    v=[next(p['raw_huber'] for p in probes if p['model']==name and p['training_frame']==fid) for name in ['L1_shared_stress','M1_guarded_shared']]
    lines.append(f'| {fid} | {v[0]:.3f} | {v[1]:.3f} |')
lines += ['',
'三个探针均恶化，但两模型最终点集/邻域不同，因此又使用共同3k canonical点、完全相同64个锚点及邻接索引、相同时刻做连续场探针，排除“只因最终采样不同”的简单解释。共同点可能后来被剪枝，不能当物理真值。','',
'| 共同材料点探针 | 帧 | 动量RMS | 加速度RMS | stress_pred散度RMS | 应变率修正散度RMS |','|---|---:|---:|---:|---:|---:|']
for p in fixed['probes']:
    lines.append(f"| {p['model']} | {p['frame']} | {p['residual_rms']:.3f} | {p['acceleration_rms_on_valid_anchors']:.3f} | {p['predicted_stress_divergence_rms']:.4f} | {p['rate_correction_divergence_rms']:.3f} |")
lines += ['',
'共同点结果不是所有时刻都变差：M1帧2改善，帧83明显恶化，帧155恶化。帧83的M1动量RMS约4711.6，几乎由原有应变率修正项的散度贡献（约4734.3），直接stress_pred散度仅约0.075；该项依赖轨迹速度的空间梯度及后续散度，可能放大场/邻域的空间高频和数值误差。这里没有新增外力；force_sigma是原有率修正系数，M1约0.6894，L1约0.7368。',
'',
'把率修正系数临时置零的只读分解中，M1帧83残差RMS降到约55.1，接近加速度RMS。**这只是定位数值贡献的探针，不能证明删除该项符合物理，也未据此修改训练或重训。** 数值上直接stress_pred头目前难以抵消这个大得多的散度项；单纯共享latent和限制梯度并未解决方程离散化/表达的兼容问题。',
'']
mprobes=[p for p in fixed['probes'] if p['model']=='M1_guarded_shared']
ahalf=[next(v['relative_difference_to_base'] for v in p['acceleration_step_sweep'] if v['factor']==.5) for p in mprobes]
vmax=max(v['relative_difference_to_base'] for p in mprobes for v in p['velocity_step_sweep'])
lines += [f"同一共同点上，加速度差分步长减半的相对变化约{min(ahalf)*100:.1f}～{max(ahalf)*100:.1f}%；速度步长半/双的最大相对变化约{vmax*100:.2f}%。更小的FP32二阶差分仍未显示稳定收敛，且存在浮点消减；最细差分不是物理真值。详见fixed_field_probes.json。",'',
'本轮证据支持“强动量梯度与图像目标冲突”这一有用优化诊断，不能支持“缺外力是根因”或“物理已解决”。若继续研究，应先在已知常量/仿射/二次速度解析场上核验局部空间梯度、率修正散度、单位和条件数，再评估时间/空间更平滑的轨迹表示及差分稳定性；不直接联合拟合外力/未知材料，也不只靠降低物理梯度宣称成功。本轮不自动增加完整训练。',
'',
'## 5. 核验与预算','',
'15项单元/CUDA测试、100步GPU短训、70→100完整状态恢复、20步flags-off、真实优化器更新、私有梯度精确保留均通过。420次动量评估/419次实际对应更新，468次光流评估；最后50k只做backward没有optimizer.step是继承行为。所有共享梯度约束记录满足上限，遗留独立物理网不变、新共享stress头确实更新；全部模型参数及最终优化器张量有限。',
'',
f"完整共同3k SHA：`{verification['common_extended3k_sha256']}`；最终完整检查点SHA：`{verification['full_checkpoint_sha256']}`。源码/数据SHA、best/final实际扩展架构加载、最终快照与完整状态对应、验证最大值/锁定时序均核验。100张拼接图逐像素正确，独立PSNR最大差{verification['independent_PSNR_max_error']:.2g}dB；两段视频各50帧/5秒，ZIP CRC通过。",'',
'后处理曾因Windows并行创建pairs目录WinError5中断，且状态读取默认GBK不能解码中文错误。已用独立finish_report.py预先串行建目录并按UTF-8读取；原训练、配置、选择锁、完成的渲染和旧模型保持原样，没有重复训练或渲染。完整恢复凭据postprocess_recovery.json。',
'',
'## 6. 交付','',
f'- [新旧视频及逐帧比较页]({(OUT/"index.html").as_posix()})',
f'- [M1验证选择GT|Render视频]({(OUT/"M1_guarded_shared/GT_Render_best.mp4").as_posix()})',
f'- [M1最终50k GT|Render视频]({(OUT/"M1_guarded_shared/GT_Render_50000.mp4").as_posix()})',
f'- [L1对照视频]({(ROOT/"output/unified_flow/L1_shared_stress/GT_Render_best.mp4").as_posix()})',
f'- [M1 PNG/视频ZIP]({(OUT/"M1_guarded_shared_GT_Render.zip").as_posix()})',
f'- [独立核验]({(REPORTS/"final_verification.json").as_posix()})',
f'- [共同材料点/差分/率修正分解]({(REPORTS/"fixed_field_probes.json").as_posix()})',
f'- [可复现性与早期漂移审计]({(REPORTS/"reproducibility_audit.json").as_posix()})','']
(REPORTS/'assessment.md').write_text('\n'.join(lines),encoding='utf8')
verification.update(manual_review_pending=False,manual_review_completed_utc=datetime.now(timezone.utc).isoformat(),
    manual_review={'PNG_and_decoded_video_frames':[1,85,101,197],
        'entire_video_decoded_to_temporal_sheet':True,
        'observations':'Airborne duck at85 recovered with remaining small floating ghost; mid-frame contact blur remains, platform/duck boundaries improve; late scene close. No claim of fully flicker-free or physical 3D correctness.'},
    all_final_optimizer_tensors_finite=audit['all_final_optimizer_tensors_finite'],
    unresolved_pre_guard_metric_drift=True,
    analysis_scripts_sha256={p.name:sha(p) for p in OUT.glob('*.py')})
write(REPORTS/'final_verification.json',verification)
write(REPORTS/'delivery.json',{'status':'verified','completed_utc':datetime.now(timezone.utc).isoformat(),
    'image_recovery':True,'composite_benefit':False,'physical_momentum_improved':False,
    'new_full_training_launched_after_this_run':False,'report_sha256':sha(REPORTS/'assessment.md'),
    'report':str(REPORTS/'assessment.md'),'gallery':str(OUT/'index.html'),
    'selection_lock_sha256':sha(ROOT/'runs/pidg_momentum_guard/selection_lock.json'),
    'automation_delete_after_delivery':'pidg'})
print('Assessment and final reviewed receipts written.')
