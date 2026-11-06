"""Record the 32 images actually inspected, after both video contacts were viewed."""
from pathlib import Path
from variants.staged_reconstruction.i10_tracking.io import ROOT,read,write,sha

def main():
    out=Path('output/staged_reconstruction/RubberDuck_surface_support/v1');names=[]
    for camera in [86,188]:
        for time in [2,100,200]:
            prefix=f'c{camera:03d}_t{time:03d}'
            names.extend(f'{prefix}_{arm}_jointRGB_native.png' for arm in ['temporal_control','surface_support'])
            names.extend(f'{prefix}_{kind}.png' for kind in ['joint_comparison','decomposition'])
    names.extend(f'observed_train_index{index:03d}.png' for index in [0,24,48,72,96,119])
    names.extend(f'fixed_c{camera:03d}_parent_control_joint.decoded.png' for camera in [86,188])
    assert len(set(names))==32
    b=Path(__file__).parent
    assert read(b/'v1/independent_audit.json')['passed']
    ev=read(b/'v1/evaluation.json')['validation'];assert ev['PSNR_nonregression_gate'] and ev['parent_nonregression_gate']
    review=dict(actual_visual_review=True,fixed_native_RGB_alpha_D_reviewed=True,
        dynamic_completeness_and_cleanup_passed=False,
        conclusion='结构候选通过两项RGB不退化门槛，但固定86/188早中晚的整体碎裂及暗高alpha拖尾没有一致消除，整体完整性与清理门槛未过，保留J6',
        findings=[
            '固定86早期两臂仍有松散、缺口和碎片状鸭子表面；中期形状较可辨，但后侧仍有粒子串；晚期两臂均有明显长条碎片和缺失。结构候选未把这些表面连成稳定完整物体。',
            '固定188早期局部仍有缺口和散点，中期两臂都有拉长拖尾及破碎表面；晚期RGB看似整洁，但动态alpha和D/alpha仍向左延伸，不能由变暗或RGB改善认定清理完成。',
            '六张分解图的静态支撑保持一致。固定86晚期候选右上仍有宽大非零alpha团块，188中晚期仍有条带支撑；未见跨相机、跨时间一致消除。黑色低alpha为缺失，深度是渲染代理。',
            '六张原训练相机/时间图中外观总体贴合源RGB，但index24前方柱顶鸭子仍不完整，index72边缘仍松散；候选没有稳定修复这些差异。训练轨迹RGB不是固定86/188跨时间独立GT。',
            '60个局部表面片只涉及275/15507个动态高斯行，按起始时刻为59/1/0；覆盖偏早，不能把局部支撑指标变化解释为全时段全表面完整。真实身体/壁厚仍无可靠前后表面对应。',
            '两段视频分别程序完整解码200帧，人工实际查看两张早中晚解码接触图，仍有上述碎裂和拖影。未实测浏览器播放，未声称逐帧人工播放。',
            'PSNR候选31.859706dB、对照31.223379dB、J6父31.994673dB；相对对照收益保留。没有独立动态三维GT，单次匹配未证明稳定因果，未凭RGB收益替换当前模型。',
        ],reviewed_images_sha256={(out/name).as_posix():sha(ROOT/out/name) for name in names},
        historical_color_parent_disclosed=True,independent_3D_GT=False,true_physical_thickness_known=False,
        browser_playback_tested=False,full_videos_decoded_programmatically=True,full_videos_watched_frame_by_frame=False)
    write(b/'v1/manual_review.json',review)

if __name__=='__main__':main()
