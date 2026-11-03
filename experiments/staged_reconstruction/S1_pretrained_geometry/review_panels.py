"""Show the sole accepted calibration; supplement frozen fixed-index figures."""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from variants.staged_reconstruction.prepare_data import ROOT, read, write, sha
from variants.staged_reconstruction.pretrained_depth.prepare import BASE, REPORT, RUNS, verify
from variants.staged_reconstruction.pretrained_depth.evaluate import heat


def main():
    verify(); evaluation = read(REPORT / 'evaluation.json'); figures, results = {}, {}
    out = ROOT / 'output/staged_reconstruction/S1_pretrained_geometry/v1'
    for case in ['diffuse', 'glossy']:
        meta = read(ROOT / f'data/derived/clean_static_synthetic/v2/{case}/train/cameras.json')
        selected = [i for i, r in enumerate(evaluation['results'][case]['frames']) if r['calibrated']]
        assert len(selected) == 1
        i = selected[0]; row = meta['frames'][i]
        with np.load(RUNS / case / f'view_{i:03d}.npz') as d:
            raw, depth, uv = d['raw_relative'], d['aligned_depth'], d['anchor_uv']
        with np.load(ROOT / f'data/derived/clean_static_synthetic/v2/{case}/evaluation_only/gt' / (Path(row['image']).stem+'.npz')) as gt:
            target = gt['static_depth']; visible = gt['observed_static_visible'] & (target > 0)
            rgb = (gt['observed_rgb']*255).round().astype('uint8')
        valid = depth > 0
        colors = rgb.copy(); colors[uv[:, 1], uv[:, 0]] = [20, 255, 30]
        tiles = [('RGB, train anchors green', colors),
                 ('Raw relative depth display', heat(raw-raw.min(), np.ones(raw.shape, bool), float(np.ptp(raw)))),
                 ('Metric depth 0..7', heat(depth, valid, 7)),
                 ('Static visible error 0..0.20', heat(abs(depth-target), valid&visible, .2))]
        canvas = Image.new('RGB', (256*4, 180), '#171923'); draw = ImageDraw.Draw(canvas)
        for j, (label, image) in enumerate(tiles):
            canvas.paste(Image.fromarray(image), (256*j, 36))
            draw.text((256*j+2, 2), f'{case} '+Path(row['image']).stem+' (only accepted view)', fill='white')
            draw.text((256*j+2, 18), label, fill='white')
        path = out / f'{case}_only_calibrated.png'; assert not path.exists()
        canvas.save(path); figures[path.relative_to(ROOT).as_posix()] = sha(path)
        record = evaluation['results'][case]['frames'][i]
        results[case] = {'image': row['image'], 'train_anchor_count': len(uv),
                         'valid_fraction_of_true_empty_in_accepted_view': record['empty_valid']/record['empty_total'],
                         'no_static_identity_or_accurate_geometry_from_local_anchor_acceptance': True}
    write(REPORT / 'supplemental_visuals.json', {'images': figures, 'results': results,
          'original_evaluation_and_gate_unchanged': True, 'source_sha256': sha(Path(__file__))})


if __name__ == '__main__': main()
