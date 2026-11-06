"""Record the completed human-visible image inspection; never train or select points."""
from pathlib import Path
import argparse
from variants.staged_reconstruction.i10_tracking.io import ROOT, read, write, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--record-completed-manual-review', action='store_true')
    if not parser.parse_args().record_completed_manual_review:
        raise SystemExit('Only record after actually inspecting every listed image')
    base = Path(__file__).parent
    out = ROOT / 'output/staged_reconstruction/RubberDuck_revocable_springs/v1'
    assert read(base / 'v1/independent_audit.json')['passed']
    images = []
    for camera in [86, 188]:
        for time in [2, 100, 200]:
            prefix = f'c{camera:03d}_t{time:03d}'
            images += [f'{prefix}_joint_comparison.png', f'{prefix}_decomposition.png']
            images += [f'{prefix}_{arm}_jointRGB_native.png'
                       for arm in ['temporal_control', 'revocable_spring']]
    images += [f'observed_train_index{index:03d}.png' for index in [0, 24, 48, 72, 96, 119]]
    images += [f'fixed_c{camera:03d}_parent_control_joint.decoded.png' for camera in [86, 188]]
    assert len(images) == len(set(images)) == 32
    evidence = {(out / name).relative_to(ROOT).as_posix(): sha(out / name) for name in images}
    write(base / 'v1/manual_review.json', dict(
        actual_visual_review=True,
        fixed_native_RGB_alpha_D_reviewed=True,
        dynamic_completeness_and_cleanup_passed=False,
        conclusion='弹簧候选较匹配对照下降0.349571dB、较J6下降0.405645dB，两项0.3dB门槛均失败；实际固定视角仍碎裂和拖尾，没有一致的完整性/清理收益，保留J6，不晋级',
        findings=[
            '实际检查6张三模型固定比较、6张RGB/alpha/D分解、12张两臂1600原生RGB、6张训练轨迹对照和2张解码接触图，共32张。',
            '相机86早期表面松散与孔隙仍在；中期右上颗粒尾仍在；晚期黄色拉长颗粒片及右上暗但非零alpha/D支撑仍在。',
            '相机188早期碎片和缺口仍在；中期左右细长拖尾和断裂表面仍在；晚期RGB相对干净但动态alpha/D仍向左延伸，不能将变暗称为消除。',
            '六张观测训练轨迹图总体比离轨迹固定相机更像原RGB；各模型细节不同，没有一致全局补全。这些是不同时间相机的训练观测，不是固定相机全时段独立GT。',
            '固定分解静态层相同；本轮静态参数/Adam完全冻结，不能将动态改变造成的联合遮挡差异称为静态重建提升。',
            '两张解码接触图仅作时段抽样人工复核；两条视频各200帧程序完整解码，未测试浏览器播放，未人工逐帧观看。',
            '原评估器部分顶部标题沿用surface support字样，正确列标签为revocable_spring；阶段L6、输入路径及权重哈希独立核验，不覆盖冻结原图。',
            '无独立动态3D或真实厚度GT；父中位参考和D/alpha均为代理，黑色低alpha缺失不是零误差。'
        ],
        reviewed_images_sha256=evidence,
        historical_color_parent_disclosed=True,
        independent_3D_GT=False,
        true_physical_thickness_known=False,
        browser_playback_tested=False,
        full_videos_decoded_programmatically=True,
        full_videos_watched_frame_by_frame=False,
        source_sha256=sha(Path(__file__))))
    print('L6_ACTUAL_32_IMAGE_REVIEW_RECORDED')


if __name__ == '__main__':
    main()
