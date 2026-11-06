"""Correct inherited sheet headings in new copies; retain frozen originals."""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from variants.staged_reconstruction.i10_tracking.io import ROOT, read, write, sha


def main():
    b = Path(__file__).parent; ev = read(b/'v1/evaluation.json'); rows = []
    out = ROOT/'output/staged_reconstruction/RubberDuck_temporal_memory/v1'
    for camera in [86,188]:
        for time in [2,100,200]:
            for kind in ['joint_comparison','decomposition']:
                original = out/f'c{camera:03d}_t{time:03d}_{kind}.png'
                assert sha(original) == ev['diagnostics']['images'][original.relative_to(ROOT).as_posix()]
                image = Image.open(original).convert('RGB'); before = np.array(image)
                draw = ImageDraw.Draw(image); draw.rectangle((0,0,image.width-1,33),fill='#171923')
                draw.text((6,8),f'L4 c{camera} t{time}: J6 parent / pure RGB / temporal surface memory; no independent fixed-time GT',fill='white')
                assert np.array_equal(before[34:],np.array(image)[34:])
                path = original.with_name(original.stem+'_review.png'); assert not path.exists(); image.save(path)
                rows.append(dict(original=original.relative_to(ROOT).as_posix(),original_sha256=sha(original),
                    corrected=path.relative_to(ROOT).as_posix(),sha256=sha(path),all_image_and_column_pixels_unchanged=True))
    write(b/'v1/title_review.json',dict(passed=True,reason='Frozen evaluator retained L2 fixed/revocable wording in sheet heading; actual columns and all native outputs were correctly labeled rgb_control/temporal_memory.',
        independent_display_only_correction=True,originals_retained=True,sheets=rows,source_sha256=sha(Path(__file__))))
    print('Twelve display-only titles corrected; native pixels and originals unchanged')


if __name__=='__main__':main()
