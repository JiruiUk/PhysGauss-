"""Build the reviewed comparison using corrected post-selection diagnostics."""
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
import imageio_ffmpeg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT/'experiments/flow_comparison'
OUT = Path(__file__).parent
read = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
motion = read(REPORT/'motion_validation_v2.json')
gradient = read(REPORT/'gradient_balance_v2.json')
selection = read(ROOT/'runs/pidg_flow_comparison/selection_lock.json')['models']
names = ['K1_flow_independent', 'K2_flow_trajectory']
best = {r['model']: r for r in motion['results'] if r['snapshot'] == 'best'}
assert set(best) == set(names)
assert read(REPORT/'final_verification.json')['status'] == 'passed'
a, b = [selection[n]['metrics'] for n in names]
ma, mb = [best[n]['metrics'] for n in names]
decision = {'PSNR_delta_dB': b['PSNR']-a['PSNR'], 'LPIPS_relative_change': b['LPIPS']/a['LPIPS']-1,
    'velocity_EPE_relative_change': mb['velocity_EPE_px']/ma['velocity_EPE_px']-1,
    'geometric_EPE_relative_change': mb['geometric_EPE_px']/ma['geometric_EPE_px']-1}
passed = [decision['PSNR_delta_dB']>=-.2, decision['LPIPS_relative_change']<=.05,
          decision['velocity_EPE_relative_change']<=-.1, decision['geometric_EPE_relative_change']<=.05]
decision.update(criteria_passed=passed, overall_benefit=all(passed), next_path='shared_trajectory_and_stress_branch_with_matched_corrected_flow_control')
(REPORT/'followup_decision.json').write_text(json.dumps(decision, indent=2)+'\n')

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
for offset, key, label in [(-.24, 'camera_only_static_EPE_px', 'Camera only / static object'),
    (0, 'geometric_EPE_px', 'Deformed trajectory'), (.24, 'velocity_EPE_px', 'Velocity advection')]:
    axes[0].bar(np.arange(2)+offset, [best[n]['metrics'][key] for n in names], width=.23, label=label)
axes[0].set_xticks(np.arange(2), ['K1 independent', 'K2 trajectory'])
axes[0].set(ylabel='Endpoint error (pixels)', title='Corrected diagnostic: raw observed flow')
axes[0].set_ylim(0, 1.8)
axes[0].legend(fontsize=8)
for i, n in enumerate(names):
    rows = [r for r in gradient['probes'] if r['model'] == n]
    axes[1].scatter([i]*len(rows), [r['groups']['deformation']['momentum_to_flow_norm_ratio'] for r in rows], s=55)
axes[1].set_xticks(np.arange(2), ['K1 independent', 'K2 trajectory'])
axes[1].set(yscale='log', ylabel='Weighted momentum / flow gradient norm', title='Actual training-objective probes (3 times)')
for ax in axes: ax.grid(axis='y', alpha=.2)
fig.savefig(OUT/'motion_comparison_v2.png', dpi=180); plt.close(fig)

fig, axes = plt.subplots(2, 3, figsize=(17, 6), constrained_layout=True)
video_checks = []
for row, n in enumerate(names):
    folder = OUT/n
    frames = imageio_ffmpeg.read_frames(str(folder/'GT_Render_best.mp4'), pix_fmt='rgb24')
    meta = next(frames); selected = {}
    for i, data in enumerate(frames):
        if i in [0, 21, 49]: selected[i] = np.frombuffer(data, np.uint8).reshape(meta['size'][1], meta['size'][0], 3).copy()
    assert i+1 == 50 and set(selected) == {0, 21, 49}
    for col, index in enumerate([0, 21, 49]):
        # A decoded-video montage inspects the actual encoded artifact.
        axes[row, col].imshow(selected[index]); axes[row, col].axis('off')
        axes[row, col].set_title(f'{n} | test index {index}', fontsize=9)
    for tag in ['best', '50000']:
        count, seconds = imageio_ffmpeg.count_frames_and_secs(str(folder/f'GT_Render_{tag}.mp4'))
        assert count == 50 and abs(seconds-5)<.11
        # Best and final both selected 50k: compare the actual source image regions.
        for index in [0, 21, 49]:
            with Image.open(folder/'pairs'/tag/f'{index:05d}.png') as im:
                assert im.size == (3200, 950)
    video_checks.append({'model': n, 'decoded_best_frames': 50, 'review_indices': [0, 21, 49]})
fig.savefig(OUT/'decoded_video_review.png', dpi=180); plt.close(fig)

marker = '\n## 核查后结论（V2 诊断）\n'
report = (REPORT/'assessment.md').read_text(encoding='utf-8').split(marker)[0]
lines = [marker, '两组均完成50k，验证选择锁定后测试。当前配置下 K2 未达到预先记录的综合改进门槛。', '',
    '| 验证选择快照 | 验证 PSNR↑ | 验证 LPIPS↓ | 轨迹 EPE↓ (px) | 速度推进 EPE↓ (px) | 观测像素覆盖 |',
    '|---|---:|---:|---:|---:|---:|']
for n in names:
    x = selection[n]['metrics']; m = best[n]['metrics']
    lines.append(f"| {n} | {x['PSNR']:.5f} | {x['LPIPS']:.7f} | {m['geometric_EPE_px']:.5f} | {m['velocity_EPE_px']:.5f} | {100*m['model_coverage']:.3f}% |")
lines += ['', f"K2 验证 PSNR 改变 {decision['PSNR_delta_dB']:.3f} dB；LPIPS 上升 {100*decision['LPIPS_relative_change']:.1f}%；速度推进 EPE 下降 {-100*decision['velocity_EPE_relative_change']:.1f}%，几何轨迹 EPE 上升 {100*decision['geometric_EPE_relative_change']:.1f}%。运动推进更接近观测，但渲染质量和轨迹运动没有同时提升。", '',
    'K1 最终内部速度相对偏差约99.80%；其推进 EPE 与仅相机运动/物体静止 baseline 接近。K2 内部偏差为零是结构定义；独立二维观测显示部分运动收益，仍不能证明三维真实速度或应力正确。', '',
    '### 后处理与训练流程缺陷', '',
    '核验发现两个代码约定错误，旧模型、源码、缓存和原诊断全部保留：', '',
    '1. 实际 PIDG 加载器把 FovX=fovy、FovY=fovx 传给 Camera；旧光流深度生成/运动诊断脚本使用常规 FoV 分配，与已有重建权重的投影约定不同。纠正诊断后，观测前景的模型覆盖从约48–50%升至99%以上。这不要求改写原 PIDG 加载器，而是让辅助流程遵守实际训练相机。',
    '2. 实际 CUDA 的 x_mu=mean-pixel，仿射流应使用其负值；此外 metadata 的 alpha*T 在 T 更新后才写入，并非 RGB 合成权重。V2 从 opacity/conic 重建更新前贡献权重。CPU 仿射膨胀测试和真实 CUDA 单 Gaussian 测试验证了这两点。', '',
    '旧 motion_validation.json、gradient_balance.json、motion_comparison.png 作为 V1 存档，已由带 _v2 的诊断替代。V1 final_verification.json 仍证明原实验文件/数值/导出符合其冻结协议，不证明目标几何正确或模型物理正确。训练使用了有上述问题的 V1 光流流程，因此不能据此否定运动光流或轨迹导数这一研究方向，也不能把退化唯一归于缺少外力。', '',
    '### 梯度和差分敏感性', '',
    'V2 梯度探针使用实际训练相机，但特意计算冻结的原训练目标，反映模型实际经历的目标；不是修正后新训练目标。']
for n in names:
    probes = [r for r in gradient['probes'] if r['model'] == n]
    ratios = [r['groups']['deformation']['momentum_to_flow_norm_ratio'] for r in probes]
    sensitivity = best[n]['finite_difference_sensitivity']
    vr = [r['velocity_step_ratio_'+str(q)] for r in sensitivity for q in [.5, 2.]]
    ar = [r['acceleration_step_ratio_'+str(q)] for r in sensitivity for q in [.5, 2.]]
    lines += [f"- {n}：三个固定训练时刻，形变 MLP 的加权动量/光流梯度范数比为 {min(ratios):.1f}–{max(ratios):.1f}；速度差分步长减半/加倍相对变化 {100*min(vr):.2f}–{100*max(vr):.2f}%，该组实际加速度来源相对变化 {100*min(ar):.1f}–{100*max(ar):.1f}%。"]
lines += ['', '这些局部探针提示损失尺度和轨迹二阶差分需要重点核查；没有测量完整训练全部梯度，也没有与图像梯度做同批次平衡审计，不能据此宣称唯一因果。K1 加速度是独立速度的一阶导，K2 是轨迹二阶导，数值幅值不可当同一种物理真值直接排名。', '',
    '### 下一轮', '',
    '按用户授权进入统一分支。但先修正光流几何/相机约定，并给新结构配同流程的原结构对照，以免把流程修复的收益算到共享分支上。新候选共享 deformation Grid4D 和 latent 输出形变与预测应力，速度来自完整中心轨迹。保留原 PIDG、K1/K2、预测应力含义和开关关闭路径，不添加外力/本构/材料未知量。下一轮独立记录初始化和预算、先测试与短训再完整训练。', '',
    '### 可视与文件核验', '',
    '两组均有100张3200×950拼接PNG、两段50帧/5秒视频和ZIP；原导出逐像素及独立PSNR核验通过。已解码两组 best 视频50帧并生成早/中/晚(索引0/21/49)审阅图；原帧85为索引21。两组 best 均选中50k，与最终快照相同。原帧85，K2 有完整移动鸭体缺失、鸭体/平台接触区拉丝和边缘伪影；K1 也有局部孔洞/浮影，不能只看全图黑背景主导的PSNR。', '',
    f'- [修正运动与梯度图]({(OUT/"motion_comparison_v2.png").as_posix()})',
    f'- [解码视频早中晚审阅图]({(OUT/"decoded_video_review.png").as_posix()})',
    f'- [训练曲线]({(OUT/"training_comparison.png").as_posix()})', '']
(REPORT/'assessment.md').write_text(report+'\n'.join(lines), encoding='utf-8')
receipt = {'status': 'passed', 'diagnostic_version': 2, 'original_training_unchanged': True,
    'V1_motion_diagnostic_superseded': True, 'V1_training_flow_has_known_geometry_errors': True,
    'decision': decision, 'decoded_video_checks': video_checks,
    'files_sha256': {str(p.relative_to(ROOT)): sha(p) for p in [
        REPORT/'motion_validation_v2.json', REPORT/'gradient_balance_v2.json',
        ROOT/'variants/unified_flow/camera_contract.py', ROOT/'variants/unified_flow/flow.py',
        ROOT/'tests/test_unified_flow_geometry.py', OUT/'geometry_v2_tests.log',
        OUT/'evaluate_motion_v2.py', OUT/'gradient_balance_v2.py', Path(__file__)]}}
(REPORT/'diagnostic_audit_v2.json').write_text(json.dumps(receipt, indent=2)+'\n')
print(json.dumps(decision, indent=2))
