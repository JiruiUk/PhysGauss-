"""Record the image inspection actually performed in this chat; no auto review."""
from pathlib import Path
import sys
import numpy as np
from PIL import Image
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from variants.staged_reconstruction.prepare_data import read, write, sha


def main():
    report = ROOT/'experiments/staged_reconstruction/S1_cross_temporal'
    out = ROOT/'output/staged_reconstruction/S1_cross_temporal'
    result = read(report/'exports.json')
    names = ['native_comparison.png', 'progressive_t0086_native.png', 'progressive_t0188_native.png',
             'fixed_t0086_native.png', 'fixed_t0188_native.png', 'confidence_final.png']
    files = {(out/'evaluation_v2'/name).relative_to(ROOT).as_posix(): sha(out/'evaluation_v2'/name) for name in names}
    assert sha(out/'native_comparison.png') == sha(out/'evaluation_v2/native_comparison.png')
    assert not (report/'manual_review.json').exists()
    write(report/'manual_review.json', {
        'checkpoint_sha256': result['checkpoint_sha256'], 'native_images_actually_viewed': True, 'files_reviewed': files,
        'native_comparison_viewed_via_identical_attempt1_bytes': True,
        'findings': [
            '实际查看2/27/43/86/166/188/200的四列对照，并单独查看progressive固定86/188、fixed固定86/188的1600x900原图和最终置信度叠图。',
            '相较固定组，progressive扩展部分早期下方支撑及局部连接，部分离散亮色伪影减弱；晚期水平支撑仍是最明确的恢复区域。',
            '固定86仍有多个断裂的模糊团块、边界外溢和亮色斑点；固定188高立柱主体大部分缺失，只剩顶端和孤立斑块。两组均未恢复完整静态支撑。',
            '旧S1-A、fixed和progressive都有尖刺/光晕或游离碎片；progressive有些轮廓更饱满但也更模糊，不能把像素覆盖增加等同于准确几何。',
            '局部亮色斑点的来源未确定，不能凭外观给出永久静态或物体归属；通用区域传播与投影RGB一致性仍可能混入遮挡/动态内容。',
            '候选置信度覆盖仍偏晚；部分早中期图像没有几何候选，更新机制无法凭空建立缺失的立柱表面。',
            '实际人工复核限于PNG；视频仅机器逐帧完整解码，不声称看完整段或在浏览器播放。'],
        'verdict': '置信度更新改善共同候选区域和未参与训练的验证相机推断区域的图像拟合/alpha支撑，但完整静态重建仍未成功；保留机制结果，不替换S3-U静态池。旧种子区域PSNR还有小幅退化，不能只报最有利区域。',
        'full_static_reconstruction_success': False,
        'next_step': '下一步优先审计几何候选覆盖、遮挡/深度可见性和长基线对应，而不是继续增加置信度迭代或扫损失权重。需要有独立静态/几何真值的可控合成评估，检验投影区域的误接纳与遗漏；再设计可增补缺失表面的候选生成。保持无已知静态颜色/物体数，保留未知，不接本构/接触/未知力。'})
    image = Image.open(out/'evaluation_v2/native_comparison.png')
    Image.fromarray(np.concatenate([np.asarray(image.crop((0, i*250, 1600, (i+1)*250))) for i in [3, 5]])).save(out/'evaluation_v2/comparison_86_188.png')
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>S1-CT completed</title>'
        '<h1>S1-CT static reconstruction comparison</h1><p>Complete reviewed outputs: '
        '<a href="evaluation_v2/index.html">open comparison, video and PLY</a></p>'
        '<p>Partial geometry only. First failed export is preserved separately.</p>'
        '<img style="max-width:100%" src="evaluation_v2/comparison_86_188.png">', encoding='utf8')
    print('CONFIDENCE_FINAL', {a: {k: v for k, v in d.items() if k != 'pixels_per_frame'} for a, d in result['confidence_summary'].items()})


if __name__ == '__main__': main()
