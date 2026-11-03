"""Report the completed automatic-static pilot without hiding limited coverage."""
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
import sys
import msvcrt
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from variants.staged_reconstruction.prepare_data import read, write, sha
from variants.staged_reconstruction.automatic_static.discover import REPORT, RUNS
from variants.staged_reconstruction.automatic_static.initialize import verify


class Links(HTMLParser):
    def __init__(self):
        super().__init__(); self.links = []

    def handle_starttag(self, tag, attrs):
        self.links.extend(value for key, value in attrs if key in ['href', 'src', 'poster'])


def main():
    verify()
    audit = read(REPORT/'formal_state_audit.json')
    exports = read(REPORT/'exports.json')
    manual = read(REPORT/'manual_review.json')
    artifacts = read(REPORT/'artifact_verification.json')
    assert artifacts['PLY_all_Gaussian_parameters_equal_checkpoint'] and artifacts['video_moov_before_mdat']
    assert audit['passed'] and audit['checkpoint_sha256'] == exports['checkpoint_sha256'] == manual['checkpoint_sha256']
    assert manual['native_images_reviewed']
    for path, expected in exports['images'].items():
        assert sha(ROOT/path) == expected
    for path, expected in manual['viewed_image_sha256'].items():
        assert sha(ROOT/path) == expected
    for row in [exports['PLY'], exports['video']]:
        assert sha(ROOT/row['path']) == row['sha256']
    assert exports['video']['decoded_frames'] == 200
    assert sha(RUNS/'formal/checkpoint.pt') == audit['checkpoint_sha256']
    discovery = read(REPORT/'discovery.json')
    historical = read(ROOT/'experiments/staged_reconstruction/completion_integrity.json')
    for row in historical['final_checkpoints']:
        assert sha(ROOT/row['path']) == row['sha256']
    union = read(ROOT/'experiments/staged_reconstruction/S3_union/physics_state_audit.json')
    assert sha(ROOT/'runs/pidg_staged_reconstruction/S3_union/physics/formal/checkpoint.pt') == union['checkpoint_sha256']
    for path, expected in read(ROOT/'experiments/staged_reconstruction/C1_baseline_comparison.json')['protected_checkpoints'].items():
        assert sha(ROOT/path) == expected
    with (RUNS/'training.lock').open('r+b') as lock:
        lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
    coverage = exports['inferred_positive_pixel_fraction_all_training_images']
    result = {'verified_utc': datetime.now(timezone.utc).isoformat(), 'stage': 'S1-A',
        'training_and_pilot_review_completed': True, 'full_automatic_scene_reconstruction_completed': False,
        'checkpoint_sha256': audit['checkpoint_sha256'], 'updates': 12000, 'static_points': exports['static_points'],
        'static_candidate_tracklets': discovery['counts']['static_candidate'],
        'incompatible_fixed_point_tracklets': discovery['counts']['incompatible_fixed_point'],
        'unknown_tracklets': discovery['counts']['unknown'],
        'input_known_static_color_or_masks': False, 'input_camera_calibration_supplied': True,
        'original_alpha_used_only_for_dataset_RGB_compositing': True,
        'inferred_positive_fraction_all_training_pixels': coverage,
        'fraction_is_not_static_recall': True, 'frames_with_any_static_support': exports['frames_with_any_inferred_static_pixels'],
        'all_train_images': exports['training_image_count'],
        'frozen_source_input_protocol_files_verified': len(read(REPORT/'source_freeze.json')),
        'source_input_protocols_Adam_all_RNG_sampling_and_native_parity_passed': True,
        'historical_S1_S6_S3_U_C1_preserved': True, 'training_lock_free': True,
        'manual_review': manual, 'artifact_verification': artifacts, 'PLY': exports['PLY'], 'video': exports['video'],
        'no_automatic_physics_or_S3_U_replacement': True}
    write(REPORT/'completion_integrity.json', result)
    lines = ['# S1-A：未知静态区域发现与3DGS重建实测', '',
        '用户要求不预知蓝色/任何部分是静态。新分支输入训练图像、时间和数据提供的相机；RGBA仅按数据协议合成黑底RGB，alpha不作分类或监督。没有输入蓝/黄阈值、静态掩膜、S1权重/视觉壳、旧颜色筛选轨迹或物体数量。', '',
        '**预定12k静态训练已执行并导出3DGS，但当前仅完成部分静态支撑恢复，未完成全场景自动静动分解。**', '',
        '[原生图片/相机轨迹视频/3DGS PLY](../../../output/staged_reconstruction/S1_automatic/index.html)', '',
        f'全图通用跟踪得到97条符合固定世界点假设的候选、200条不符合、515条证据不足。轨迹是一段图像证据，不等于97个不同物理点或任何物体数量。固定点使用交错观测三角化，在其余观测检查重投影；低视差/短时间跨度保持未知。', '',
        f'由几何种子支持通用图像区域，再在三角化点附近建立543个高斯。120张训练图中51张有正区域，41张达到训练的64像素条件；所有训练像素中的正支持占{coverage*100:.4f}%。分母包含背景与动态内容，因此这不是静态召回率，也不能将其余像素认定为动态。', '',
        f'非空训练推断区域的平均PSNR为{exports["mean_inferred_region_PSNR_nonempty_training_frames_only"]:.6f}dB，属于选择条件下的训练代理量；不能与S1基于蓝色区域的验证35.644dB直接比较。没有独立静态掩膜或几何GT，不能报告全场景静态精度。', '',
        '17项相关测试通过。独立进程40→80步恢复、6组Adam计数、全部相机采样/Python/NumPy/Torch/CUDA RNG、原生GaussianModel精确渲染对照及冻结输入/源码通过。第一次40步审计发现原生类构造消耗随机数；保留失败代码与运行，修复仅涉及新分支只读对照的RNG保护，正式训练使用已验证的revision2。', '',
        '实际查看：', '', *['- '+s for s in manual['findings']], '',
        '静态候选只代表在观测区间内符合固定世界点；运动物体暂时静止、跟踪误配、弱纹理或遮挡都可能产生歧义。通用区域传播还有外观平滑先验。没有学习材料分割、估计物体数或输入语义类别。未加入物理/接触/本构/未知力。', '',
        '模型权重与标准3DGS PLY已保存。首次PLY导出因兼容适配器的dynamic_mask为空而失败；仅修复独立导出适配器并保留失败源/日志，未改训练代码或权重。H264视频逐帧解码200帧；仅记录图像经过人工查看，不声称人工看完整段视频或浏览器播放。原S1–S6、S3-U与C1检查点保持不变。', '',
        '计划修正：未知覆盖仍是当前瓶颈，下一步应改进全图长基线对应及几何覆盖的独立验证，不用颜色标签补全未知区域。S3-U现有黄色初始化/覆盖目标也有先验；只替换静态池不能声称全流程无语义先验。当前结果不直接替换S3-U、不接物理、不自动扫参。']
    (REPORT/'assessment.md').write_text('\n'.join(lines)+'\n', encoding='utf8')
    page = ROOT/'output/staged_reconstruction/S1_automatic/index.html'
    parser = Links(); parser.feed(page.read_text(encoding='utf8'))
    for link in parser.links:
        assert (page.parent/link).resolve().is_file(), link
    write(RUNS/'state.json', {'status': 'completed_reviewed_partial_static', 'iteration': 12000,
        'full_automatic_scene_reconstruction_complete': False, 'report': str((REPORT/'assessment.md').relative_to(ROOT)),
        'no_automatic_follow_on_training': True})
    print('AUTOMATIC_STATIC_FINALIZED', result['static_points'], 'Gaussians; incomplete static coverage retained', flush=True)


if __name__ == '__main__':
    main()
