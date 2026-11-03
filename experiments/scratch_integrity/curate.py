"""Present the human-reviewed verdict beside the final videos."""
from pathlib import Path
import html
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from variants.metric_tuning.runtime import read_json, write_json, sha256

REPORT = ROOT / 'experiments/scratch_integrity'
OUT = ROOT / 'output/scratch_integrity'


def main():
    review = read_json(REPORT / 'manual_review.json')
    metrics = read_json(REPORT / 'numeric_summary.json')['models']
    page = OUT / 'index.html'
    text = page.read_text(encoding='utf-8')
    assert 'id="final-verdict"' not in text
    a = metrics['S0_scratch_control']['test']['PSNR']
    b = metrics['S1_scratch_transfer']['test']['PSNR']
    verdict = (f'<p id="final-verdict"><strong>{html.escape(review["finding_zh"])}</strong></p>'
               f'<p>匹配对照：S0/S1 各从零训练至总50k；测试PSNR {a:.4f} / {b:.4f} dB。'
               '<a href="native_crop_fixed86_t2.png">原生像素局部对照</a>；'
               '<a href="native_inspection_fixed86.png">固定86相机早/中/晚对照</a>。'
               '固定新视角没有独立图像真值，点库支撑不等于物理身份准确率。</p>')
    end = text.index('</h1>') + len('</h1>')
    text = text[:end] + '\n' + verdict + text[end:]
    page.write_text('\n'.join(line.rstrip() for line in text.splitlines()) + '\n', encoding='utf-8')
    write_json(REPORT / 'artifact_manifest.json', {
        p.relative_to(ROOT).as_posix(): sha256(p) for p in OUT.rglob('*')
        if p.is_file() and '__pycache__' not in str(p)})


if __name__ == '__main__':
    main()
