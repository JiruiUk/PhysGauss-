"""Create labeled native-pixel comparisons from the saved final renders."""
from pathlib import Path
import sys
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from variants.metric_tuning.runtime import write_json, sha256

OUT = ROOT / 'output/scratch_integrity'
REPORT = ROOT / 'experiments/scratch_integrity'
NAMES = ['S0_scratch_control', 'S1_scratch_transfer']
BOX = (600, 220, 1250, 800)


def main():
    source = {}
    for frame in [2, 86, 200]:
        canvas = Image.new('RGB', (1300, 607), '#171a21')
        draw = ImageDraw.Draw(canvas)
        for column, name in enumerate(NAMES):
            path = OUT / name / 'fixed_0086/native' / f'rgb{frame:04d}.png'
            with Image.open(path) as image:
                assert image.size == (1600, 900)
                canvas.paste(image.crop(BOX), (column * 650, 27))
            draw.text((column * 650 + 8, 7), f'{name} final 50k | fixed86 | time{frame} | native pixels', fill='white')
            source[path.relative_to(OUT).as_posix()] = sha256(path)
        canvas.save(OUT / f'native_crop_fixed86_t{frame}.png')
    contact = Image.new('RGB', (1600, 3 * 477), '#171a21')
    draw = ImageDraw.Draw(contact)
    for row, frame in enumerate([2, 86, 200]):
        for column, name in enumerate(NAMES):
            path = OUT / name / 'fixed_0086/native' / f'rgb{frame:04d}.png'
            with Image.open(path) as image:
                contact.paste(image.resize((800, 450), Image.Resampling.LANCZOS), (column * 800, row * 477 + 27))
            draw.text((column * 800 + 8, row * 477 + 7), f'{name} final50k | fixed86 | time{frame} | 2x downsample', fill='white')
    contact.save(OUT / 'native_inspection_fixed86.png')
    write_json(REPORT / 'inspection_provenance.json', {
        'crop_native_pixel_box': list(BOX), 'crop_resampling': 'none',
        'source_sha256': source, 'script_sha256': sha256(Path(__file__))})
    write_json(REPORT / 'artifact_manifest.json', {
        p.relative_to(ROOT).as_posix(): sha256(p) for p in OUT.rglob('*')
        if p.is_file() and '__pycache__' not in str(p)})


if __name__ == '__main__':
    main()
