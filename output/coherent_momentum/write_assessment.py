"""Finalize the verified result, comparison page and explicit interpretation."""
import json
import statistics
from datetime import datetime,timezone
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from variants.metric_tuning.run import read_json as read,write_json as write,sha256 as sha
from variants.coherent_momentum.run import assert_frozen,RUNS,NAME,REPORTS
OUT=Path(__file__).parent;assert_frozen(ROOT/'variants/coherent_momentum/code')
r=read(REPORTS/'results.json');q=read(REPORTS/'final_verification.json');assert q['status']=='passed'
physics=read(REPORTS/'common_momentum.json');sensitivity=read(REPORTS/'field_sensitivity.json')
operational=read(REPORTS/'operational_motion.json')['result']['metrics']
new=r['metrics']['ours_best'];old=r['control_metrics']['ours_best'];sel=r['selection'];cs=r['control_selection']
motion=r['motion'][0]['metrics'];prior=read(ROOT/'experiments/momentum_repair/motion_validation.json')['results'][0]['metrics']
run=RUNS/NAME/'rubber_duck'
guards=[json.loads(s) for s in (run/'momentum_gradient_guard.jsonl').read_text().splitlines()]
events=[json.loads(s) for s in (run/'coherent_momentum.jsonl').read_text().splitlines()]
scales=[g['scale'] for g in guards]
state=torch.load(run/'checkpoints/resume.pt',map_location='cpu',weights_only=False)
counter=[0]
def finite(value):
    if isinstance(value,torch.Tensor):
        counter[0]+=1
        if value.is_floating_point() or value.is_complex():assert value.isfinite().all()
    elif isinstance(value,dict):
        for item in value.values():finite(item)
    elif isinstance(value,(list,tuple)):
        for item in value:finite(item)
for key in ['gaussian_optimizer','deform_optimizer','velocity_optimizer']:
    if key in state:finite(state[key])
assert counter[0]>0
del state
diagnostics={'status':'passed','optimizer_tensors_checked':counter[0],
    'all_optimizer_tensors_finite':True,'guard_scale':{'minimum':min(scales),'median':statistics.median(scales),'maximum':max(scales)},
    'training_valid_fraction_min':min(e['valid_fraction'] for e in events),
    'training_valid_fraction_median':statistics.median(e['valid_fraction'] for e in events),
    'boundary_shift_events':[{'iteration':e['iteration'],'requested_t':e['requested_t'],'center_shift':e['boundary_center_shift']}
        for e in events if abs(e['boundary_center_shift'])>1e-8]}
write(REPORTS/'additional_verification.json',diagnostics)
lines=['# 统一离散轨迹动量：N1 对照 M1','',
'**结论：新方案保持重建质量、略改善独立二维运动，并明显缓解第83帧的动量异常；但未达到预先规定的跨时刻动量改进门槛，物理推进与形变轨迹仍未充分一致。**','',
'## 1. 实际修改及“统一”的含义','',
'L1/M1原本就从完整渲染中心 `x(X,t)=R(q(X,t))X+b(X,t)` 导出速度和加速度，并由同一形变Grid4D/latent输出stress_pred。本轮不能宣称首次消除独立速度分支。',
'N1进一步统一归一化时间区间h=1/199：两段弦速度生成中心速度与加速度；原stress_pred及原应变率系数保留，物理推进位置由同一材料点和同一时间区间计算。','',
'```text\nv_minus=(x(t)-x(t-h))/h\nv_plus =(x(t+h)-x(t))/h\nv(t)=(v_minus+v_plus)/2\na(t)=(v_plus-v_minus)/h\nx_physics(t+h)=2*x(t)-x(t-h)+h^2*div(sigma)\nloss=Huber((x(t+h)-x_physics(t+h))/h^2)\n```','',
'损失仍以加速度单位训练，避免h²缩放造成“推进误差很小”的假象。推进误差与动量残差只是代数缩放关系，不能把它们当作两份独立证据。密度仍取1，时间为序列归一化时间，指标不是有真实单位标定的牛顿/秒。',
'空间求导改为32邻居、归一化坐标上的二次WLS：直接估计stress Jacobian与velocity Hessian，替换连续两次线性拟合。保留原率修正，使用3D恒等式 div(dev(sym(grad(v))))=0.5*laplacian(v)+grad(div(v))/6。场值仍是FP32，差分和求解转FP64；正则1e-5，秩比阈值1e-6。边界中心移入[h,1-h]并记录，不外推。',
'这是时间区间、邻域、空间阶次和数值精度的组合改动，不能归为单个标量的因果收益。M1梯度guard保持，未新增网络参数、优化器状态、外力、本构或未知材料。','',
'## 2. 预算与可比性','',
'Rubber Duck，seed0，120训练/30验证/50测试。N1从与M1相同扩展完整3k状态到总50k，新增47k步，仅这一个候选，没有额外前缀拟合。共同检查点字节SHA、张量/优化器/RNG起点相同，实际相机/光流帧日程匹配。历史模型保持原样。',
f"M1验证选择{cs['iteration']}步，N1选择{sel['iteration']}步；N1 best与final50k权重相同。新测试在{read(RUNS/'selection_lock.json')['locked_utc']}锁定验证选择后打开。单场景单种子、历史测试暴露及CUDA非确定性限制因果与泛化结论，不能将组合收益唯一归因某一环节。",'',
'## 3. 图像与独立观测运动','',
'| 指标 | M1 | N1 |','|---|---:|---:|',
f"| 最佳验证PSNR | {cs['value']:.6f} | {sel['value']:.6f} |",
f"| 选择快照验证LPIPS | {cs['metrics']['LPIPS']:.8f} | {sel['metrics']['LPIPS']:.8f} |"]
for key in ['PSNR','SSIM','LPIPS','MS-SSIM']:lines.append(f"| 测试best {key} | {old[key]:.8f} | {new[key]:.8f} |")
lines += [f"| 验证完整轨迹EPE / px | {prior['geometric_EPE_px']:.6f} | {motion['geometric_EPE_px']:.6f} |",
f"| 验证速度推进EPE / px | {prior['velocity_EPE_px']:.6f} | {motion['velocity_EPE_px']:.6f} |",'',
f"测试PSNR提高{new['PSNR']-old['PSNR']:.4f}dB，但验证PSNR只提高{sel['value']-cs['value']:.4f}dB；测试LPIPS略恶化{(new['LPIPS']/old['LPIPS']-1)*100:.2f}%，SSIM也略低，不能称全面画质提升。人工查看原帧1/85/101/197的PNG和解码视频：整体两组接近，N1第85帧悬空鸭子仍有拖影/漂浮小片，接触区局部模糊仍在。全部50帧视频已解码总览审阅，没有据此声称完全无闪烁或3D几何真实。",'',
f"独立二维轨迹EPE降低{(1-motion['geometric_EPE_px']/prior['geometric_EPE_px'])*100:.2f}%，速度推进EPE降低{(1-motion['velocity_EPE_px']/prior['velocity_EPE_px'])*100:.2f}%。主比较沿用冻结V2 h=.001，对M1/N1都相同；N1实际训练h=1/199的补充只读评估EPE为{operational['velocity_EPE_px']:.6f}px，未用于选择或替换预先比较。观测来自冻结验证raw UniMatch估计流，不是3D物理真值。",'',
'## 4. 同材料点、双算子的动量结果','',
'共同3k canonical点集、相同64锚点、相同训练帧2/83/155；每种算子分别应用于两模型。当前形变使邻域条件可能不同，本次配对有效率均100%。前缀点后来可能被剪枝，因此这是连续场探针，不是物理真值。','',
'| 帧 | 原算子M1 Huber | 原算子N1 Huber | 新算子M1 Huber | 新算子N1 Huber |','|---|---:|---:|---:|---:|']
for frame in [2,83,155]:
    lookup={(v['model'],v['operator']):v for v in physics['rows'] if v['frame']==frame}
    vals=[lookup[(model,op)]['raw_huber'] for op in ['legacy','coherent'] for model in ['M1_guarded_shared',NAME]]
    lines.append('| '+str(frame)+' | '+' | '.join(f'{v:.3f}' for v in vals)+' |')
lines += ['',
'原算子跨三个时刻的Huber中位数119.157→115.870，仅降低2.76%；新算子114.981→106.689，降低7.21%。第83帧改善很明显：原算子Huber降低约78.69%，新算子约48.93%；但第2帧两种算子都恶化，第155帧仅小幅改善。不能只展示最好时刻宣称整体物理已解决。',
'新算子第83帧残差RMS约2589.67→898.74，对应一步位置偏差RMS约0.06539→0.02269（归一化世界坐标）；这仍是同一残差的缩放。三维图提供世界位置与全部64点局部位移两种视角，使用共同坐标尺度、不人为放大；8个高速度点的局部图不替代64点全体统计。','',
'## 5. 为什么仍未充分一致','',
'只读分解显示N1第83帧加速度RMS约73.17，stress_pred散度约0.183，原应变率修正散度约888.63，残差约898.74，率修正仍主导。原可学习率系数由M1约0.6894变为N1约0.6251；它不是外力。不能据此归因缺外力，也不能直接删掉该项后声称物理正确。',
'时间步长减半后，N1三时刻加速度相对变化约18.8%～22.1%；残差向量相对变化约9.7%、11.7%、70.7%。邻域32→48时残差向量变化约16.3%～82.6%，16邻居更敏感。说明这仍依赖求导尺度/邻域，FP64算术没有让已学FP32场自动变得平滑，更小步长不是已验证的真值。',
f"训练保留M1 guard，动量共享梯度缩放中位数{statistics.median(scales):.3g}，范围{min(scales):.3g}～{max(scales):.3g}。它防止动量梯度破坏图像目标，但也使共享轨迹的物理更新受到强限制；guard通过不等于动量方程满足。这与率修正仍主导是有证据的剩余瓶颈，不能认定为全部退化的唯一原因。",'',
'预先标准要求：验证PSNR损失≤0.5dB、LPIPS相对恶化≤10%、观测速度EPE恶化≤5%，并且两种算子各自的共同锚点残差中位数至少降低25%、有效率≥50%。图像、观测运动和覆盖标准通过，两个动量门槛未通过，**综合收益为false**，不事后降低标准。','',
'## 6. 对原始想法的判断与下一步边界','',
'“由同一轨迹定义运动状态”是合理的，本轮也进一步改善离散兼容；但N1仍是轨迹预测网络加软物理目标，不是应力积分器直接生成渲染轨迹，无法保证严格共轨。如果要求严格结构约束，后续应研究“应力驱动积分状态作为唯一渲染中心”的模型，同时核验数值稳定性、独立观测运动及未见时刻推进；积分方程自身零残差仍是结构定义，不是额外物理真值。',
'本轮已按授权完成一个独立候选及比较；保留N1作为数值兼容性研究候选，不宣称物理成功，不自动增加完整训练。未引入外力/本构/未知材料，也未修改原PIDG及历史源码。','',
'## 7. 独立核验及交付','',
f"24项解析/CUDA/接口测试、100步GPU短训70→100完整恢复、20步flags-off和实际更新/私有梯度保留检查通过。468次光流、420次动量评估、419次实际动量更新（继承trainer最后一步不step）。训练有效邻域最小{diagnostics['training_valid_fraction_min']*100:.2f}%，中位{diagnostics['training_valid_fraction_median']*100:.2f}%；有{len(diagnostics['boundary_shift_events'])}次边界中心移动，见additional_verification.json。遗留独立物理网络不变，实际共享头更新，最终参数和{counter[0]}个优化器状态张量有限。",
f"源码/数据/完整检查点/快照SHA及验证锁定时序核验通过。100张拼接PNG逐像素一致，独立PSNR最大误差{q['independent_PSNR_max_error']:.2g}dB，两段视频各50帧/5秒，ZIP CRC通过。N1 best=final50k模型权重相同，两段视频标题不同。",
f"共同3k SHA：`{q['common_extended3k_sha256']}`；完整50k SHA：`{q['full_checkpoint_sha256']}`。",'',
f"- [新旧同步视频与逐帧对比]({(OUT/'index.html').as_posix()})",
f"- [N1 GT|Render视频]({(OUT/NAME/'GT_Render_best.mp4').as_posix()})",
f"- [N1 PNG/视频ZIP]({(OUT/(NAME+'_GT_Render.zip')).as_posix()})",
f"- [全部64点三维位移对齐]({(OUT/'rollout_displacement_3d.png').as_posix()})",
f"- [双算子共同材料点证据]({(REPORTS/'common_momentum.json').as_posix()})",
f"- [时间/空间尺度敏感性]({(REPORTS/'field_sensitivity.json').as_posix()})",
f"- [独立核验]({(REPORTS/'final_verification.json').as_posix()})",'']
(REPORTS/'assessment.md').write_text('\n'.join(lines),encoding='utf8')
q.update(manual_review_pending=False,manual_review_completed_utc=datetime.now(timezone.utc).isoformat(),
    manual_review={'PNGs_and_decoded_video_frames':[1,85,101,197],'complete_50_frame_temporal_sheet':True,
        'plots':['comparison.png','rollout_alignment_3d.png','rollout_displacement_3d.png'],
        'finding':'Similar reconstruction, airborne-duck ghosting remains; frame83 force rollout closer but global momentum criteria fail.'},
    all_optimizer_tensors_finite=True,analysis_scripts_sha256={p.name:sha(p) for p in OUT.glob('*.py')})
write(REPORTS/'final_verification.json',q)
write(REPORTS/'delivery.json',{'status':'verified','completed_utc':datetime.now(timezone.utc).isoformat(),
    'composite_benefit':False,'physical_momentum_solved':False,'new_full_training_launched_after_this_run':False,
    'report_sha256':sha(REPORTS/'assessment.md'),'report':str(REPORTS/'assessment.md'),'gallery':str(OUT/'index.html'),
    'selection_lock_sha256':sha(RUNS/'selection_lock.json'),'automation_delete_after_delivery':'automation'})
print('Final assessment and manual review receipt written')
