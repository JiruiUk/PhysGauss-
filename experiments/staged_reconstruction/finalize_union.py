"""Compile reviewed S3-U results without changing training thresholds or weights."""
import sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from variants.staged_reconstruction.prepare_data import REPORT, RUNS, read, write, sha
from variants.staged_reconstruction.union.prepare import UNION_REPORT, UNION_RUNS, verify_sources


def main():
    verify_sources()
    original = read(REPORT/'S3/final_verification.json')
    warmup = read(UNION_REPORT/'warmup_state_audit.json')
    final = read(UNION_REPORT/'physics_state_audit.json')
    reviewed = read(UNION_REPORT/'physics_manual_review.json')
    admission = read(UNION_REPORT/'warmup_admission.json')
    exports = read(UNION_REPORT/'physics_exports.json')
    assert admission['admitted_for_conditional_physics'] and admission['native_images_reviewed']
    assert reviewed['native_images_reviewed'] and reviewed['checkpoint_sha256'] == final['checkpoint_sha256']
    assert exports['checkpoint_sha256'] == final['checkpoint_sha256']
    assert len(exports['videos']) == 6
    for row in exports['videos']:
        assert row['frames'] == 200 and sha(ROOT/row['path']) == row['sha256']
    for path, expected in reviewed['viewed_image_sha256'].items():
        assert sha(ROOT/path) == expected
    for path, expected in exports['images'].items():
        assert sha(ROOT/path) == expected
    protected = []
    for row in read(REPORT/'completion_integrity.json')['final_checkpoints']:
        assert sha(ROOT/row['path']) == row['sha256']
        protected.append(row)
    c1 = read(REPORT/'C1_baseline_comparison.json')
    for path, expected in c1['protected_checkpoints'].items():
        assert sha(ROOT/path) == expected
    quality = {split: read(UNION_REPORT/'physics'/f'{split}_06000_1600_standard.json')['metrics'] for split in ['val', 'test']}
    old_quality = read(REPORT/'S3/native_full_quality.json')['splits']
    old_state = torch.load(RUNS/'S3/formal/checkpoint.pt', map_location='cpu', weights_only=False)['model']
    old_parts = {name: sum(v.numel() for k, v in old_state.items() if k.startswith(prefix) and not k.endswith('initial_xyz'))
        for name, prefix in [('dynamic_parameters', 'objects.'), ('grid_parameters', 'grids.'),
                             ('deform_parameters', 'networks.'), ('velocity_stress_parameters', 'physics.fields.')]}
    new_parts = final['parameters']
    intermediate = read(UNION_REPORT/'physics/val_04000_1600.json')['metrics']
    result = {'stage': 'S3-U', 'scope': 'Unified dynamic population; neither input nor estimate physical body number',
        'training_completed': True, 'warmup_updates': 16000, 'physics_updates': 6000,
        'physical_body_number': None, 'instance_identity_available': False,
        'geometry_or_physical_success_claimed': False, 'warmup_audit': warmup, 'physics_audit': final,
        'preselected_final_iteration': 6000,
        'intermediate_4000_dynamic_PSNR': intermediate['dynamic_PSNR'],
        'final_selection': 'Report the preselected 6000-update endpoint; no intermediate checkpoint selection or retraining',
        'test_suite': read(UNION_REPORT/'test_receipt.json'),
        'standard_native_quality': quality,
        'original_S3_native_validation': original['native_validation'],
        'comparison_to_original_S3': {
            'dynamic_PSNR_delta': final['native_validation']['dynamic_PSNR']-original['native_validation']['dynamic_PSNR'],
            'static_PSNR_delta': final['native_validation']['static_PSNR']-original['native_validation']['static_PSNR'],
            'full_PSNR8bit_delta': quality['val']['PSNR_8bit']-old_quality['val']['metrics']['PSNR'],
            'SSIM8bit_delta': quality['val']['SSIM_8bit']-old_quality['val']['metrics']['SSIM'],
            'LPIPS8bit_delta': quality['val']['LPIPS_8bit']-old_quality['val']['metrics']['LPIPS']},
        'parameter_comparison': {'original_S3': old_parts, 'S3_U': {k: new_parts[k] for k in old_parts},
            'limit': 'Same total dynamic Gaussian count; shared network and grid have different actual parameter counts'},
        'videos': exports['videos'], 'fixed_frame_internal_physics': exports['fixed_frame_internal_physics'],
        'visual_review': reviewed, 'original_final_checkpoints_preserved': protected,
        'limits': ['No instance discovery or number inference', 'Appearance mask and monocular depth are conditional priors, not GT',
            'Dynamic-only kNN can cross physical boundaries at close approach', 'Stress/velocity/CMR are internal sequence-normalized diagnostics',
            'Fresh union initialization, union coverage and shared capacity differ; not a matched single-switch causal ablation']}
    write(UNION_REPORT/'final_summary.json', result)
    def f(v):
        return f'{v:.6f}'
    a = original['native_validation']
    b = final['native_validation']
    text = ['# S3-U 统一动态池实测结果', '',
        '按用户选择，不输入也不估计动态物体数量。一个 canonical 动态高斯池、一个原 PIDG 形变场、一个速度/应力场覆盖所有动态内容；没有四组物体初始化或实例 ID。', '',
        '16k 重建预热和6k S3运动学/CMR训练已执行；各自经过独立进程40→80恢复检查。共享并完全冻结S1静态支撑。**这不代表完整几何或物理身份已验证成功。**', '',
        '[固定86/188及原轨迹并排视频](../../../output/staged_reconstruction/S3_union/physics/index.html)', '',
        '| 原生验证指标 | 原 S3 | S3-U |', '|---|---:|---:|',
        *[f'| {name} | {f(a[key])} | {f(b[key])} |' for key, name in [('dynamic_PSNR', '动态PSNR dB'), ('dynamic_alpha', '动态单独alpha'), ('static_PSNR', '静态区域PSNR dB')]], '',
        '| 全画幅8bit原生验证 | 原 S3 | S3-U |', '|---|---:|---:|',
        *[f'| {name} | {f(old_quality["val"]["metrics"][old])} | {f(quality["val"][new])} |' for old, new, name in [('PSNR', 'PSNR_8bit', 'PSNR dB'), ('SSIM', 'SSIM_8bit', 'SSIM'), ('LPIPS', 'LPIPS_8bit', 'VGG-LPIPS')]], '',
        f'S3-U测试动态PSNR {f(final["native_test"]["dynamic_PSNR"])}dB；测试全画幅PSNR {f(quality["test"]["PSNR_8bit"])}dB、SSIM {f(quality["test"]["SSIM_8bit"])}、LPIPS {f(quality["test"]["LPIPS_8bit"])}。', '',
        f'4k中途验证动态PSNR为{f(intermediate["dynamic_PSNR"])}dB，6k终态下降至{f(b["dynamic_PSNR"])}dB。报告预先指定的6k终态，不挑中途最优点，也未据此自动调权重重训。', '',
        f'全部{final["physics_objective_calls"]}次物理调用的有效二阶空间邻域比例最低为{f(final["physics_valid_fraction_min"])}。它只描述数值算子适用性，不认证邻域属于同一物理物体。', '',
        '| 可训练参数 | 原 S3 | S3-U |', '|---|---:|---:|',
        *[f'| {k} | {old_parts[k]} | {new_parts[k]} |' for k in old_parts], '',
        f'动态高斯总点数均为15,507，可训练参数总计原S3为{sum(old_parts.values()):,}、S3-U为{new_parts["trainable_parameters"]:,}；共享网格、形变网络和物理场的实际容量不同，同时初始化和覆盖约束从逐组变为并集。结果用于表示方案比较，不能归因于一个单独开关。', '',
        '49个相关测试通过，原生渲染对照最大差3.5763e-7；预热与正式物理分支分别通过独立进程40→80步恢复检查。完整采样/RNG、Adam计数与静态冻结审计通过，训练/评估已退出。本次到S3-U结束，不自动扩展S4-U–S6-U。', '',
        '实际查看图像：', '', *[f'- {item}' for item in reviewed['findings']], '',
        '未估计物体数，未证明遮挡后的物理身份、材料点对应或完整三维表面。没有接触、本构、未知力、轨迹损失或新增损失权重。空间kNN跨物体风险保持未知。原S4逐物体固定行绑定不能直接移植，因此不伪造可比的逐物体轨迹误差。', '',
        '六段H264视频均完整解码200帧；相机为metadata CV，固定86/188及原轨迹按(frame-1)/199同步。固定跨时间视角没有独立图像/几何真值。仅对记录的图像进行了人工复核，没有声称人工看完全部视频帧。', '',
        '来源、检查点/静态冻结、Adam计数、全部采样和原历史检查点保护见 final_summary.json 与各阶段 state_audit、smoke_verification 和 exports 收据。最终文件、视频码流、页面链接及历史结果保护检查见 [completion_integrity.json](completion_integrity.json)。审计收据中的pending字段是人工复核前状态；最终复核以physics_manual_review.json为准。']
    (UNION_REPORT/'assessment.md').write_text('\n'.join(text)+'\n', encoding='utf8')
    print('S3_UNION_FINALIZED', result['comparison_to_original_S3'], flush=True)


if __name__ == '__main__':
    main()
