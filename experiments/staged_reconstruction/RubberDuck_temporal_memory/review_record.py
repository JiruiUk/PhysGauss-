"""Record only the 34 images actually inspected during the L4 review."""
from pathlib import Path
from variants.staged_reconstruction.i10_tracking.io import ROOT, sha, write


def main():
    output = Path('output/staged_reconstruction/RubberDuck_temporal_memory/v1')
    images = ['memory_coverage.png', 'observed_memory_examples.png']
    for camera in [86, 188]:
        for time in [2, 100, 200]:
            prefix = f'c{camera:03d}_t{time:03d}'
            images.extend(f'{prefix}_{arm}_jointRGB_native.png'
                          for arm in ['rgb_control', 'temporal_memory'])
            images.extend(f'{prefix}_{kind}_review.png'
                          for kind in ['joint_comparison', 'decomposition'])
    images.extend(f'observed_train_index{index:03d}.png'
                  for index in [0, 24, 48, 72, 96, 119])
    images.extend(f'fixed_c{camera:03d}_parent_control_joint.decoded.png'
                  for camera in [86, 188])
    assert len(set(images)) == 34
    review = dict(
        actual_visual_review=True,
        fixed_native_RGB_alpha_D_reviewed=True,
        dynamic_completeness_and_cleanup_passed=False,
        conclusion='局部轨迹对应改善，但固定视角碎裂及高alpha暗拖尾没有一致消除，两项RGB不退化门槛失败，保留J6，不晋级本轮候选',
        findings=[
            '固定86早期两训练臂仍有不完整且边缘松散的鸭子表面，中期部分形状可辨但仍有散点；晚期两臂均有明显拉长粒子串及缺失，未见稳定整体改善。',
            '固定188早期表面仍碎裂，中期鸭子附近有条带状拖尾；晚期RGB较整洁，但对应动态alpha及D/alpha仍有向左延伸支撑，不能以较暗RGB认定已清理。',
            '六张RGB/alpha/D分解图中静态支撑一致；动态暗RGB区域仍有非零乃至高alpha支撑，原生深度代理存在条带或团块。D/alpha不是独立几何真值，黑色低alpha是缺失。',
            '六个原训练相机与时间观测对照中，三模型大体贴合输入外观，部分边缘及遮挡处仍有差异；这些图属于训练轨迹RGB，不能作为固定86/188跨时间独立GT。',
            '时间覆盖图与原RGB轨迹示例显示合格记忆集中早期；后期未绑定示例位于立柱，仍保持未知，未强行关联至动态点。没有证明所有动态表面均被追踪覆盖。',
            '全部22条原绑定记忆、372个观测的重投影诊断中，对照held均值0.895909像素、记忆0.561905像素，fit也改善；held只从拼接及辅助损失隔离，仍参与普通RGB，非独立几何验证。',
            '两段视频各完整程序解码200帧，人工实际查看各含早中晚的解码接触图；未实际测试浏览器播放，也未声称逐帧人工播放。接触图同样未见一致的碎裂和拖尾消除。',
            '22个canonical点仅覆盖16至20帧局部观测，372个合格观测时间三段356/16/0；18条记忆新增76个根窗口外实测观测，但新增观测不等于新增表面面积或完整三维融合。',
        ],
        reviewed_images_sha256={(output / name).as_posix(): sha(ROOT / output / name)
                                for name in images},
        historical_color_parent_disclosed=True,
        independent_3D_GT=False,
        browser_playback_tested=False,
        full_videos_decoded_programmatically=True,
        full_videos_watched_frame_by_frame=False,
    )
    write(Path(__file__).parent / 'v1/manual_review.json', review)


if __name__ == '__main__':
    main()
