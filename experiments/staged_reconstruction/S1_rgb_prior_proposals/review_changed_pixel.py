"""Independent evaluator-only visual review of the sole accepted change."""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from variants.staged_reconstruction.prepare_data import ROOT, read, write, sha
from variants.staged_reconstruction.rgb_prior_proposals.prepare import REPORT, RUNS, verify


def main():
    verify(); evaluation = read(REPORT/'evaluation.json'); images, records = {}, {}
    out = ROOT/'output/staged_reconstruction/S1_rgb_prior_proposals/v1'
    for case in ['diffuse', 'glossy']:
        frames = evaluation['results'][case]['frames']; selected = [i for i, r in enumerate(frames) if r['accepted']]
        assert selected == [82]
        i = selected[0]; name = Path(frames[i]['image']).stem
        with np.load(RUNS/case/f'view_{i:03d}.npz') as d:
            accepted = d['accepted']; xy = d['uv'][accepted]; assert len(xy) == 1
            x, y = xy[0]; old, new = float(d['original'][accepted][0]), float(d['updated'][accepted][0])
            fit_old, fit_new = float(d['fit_old'][accepted][0]), float(d['fit_new'][accepted][0])
            held_old, held_new = d['held_old'][:, accepted][:, 0], d['held_new'][:, accepted][:, 0]
        with np.load(ROOT/f'data/derived/clean_static_synthetic/v2/{case}/evaluation_only/gt'/(name+'.npz')) as gt:
            rgb = Image.fromarray((gt['observed_rgb']*255).round().astype('uint8'))
            target = float(gt['static_depth'][y, x]); assert gt['observed_static_visible'][y, x]
        canvas = Image.new('RGB', (700, 220), '#171923'); draw = ImageDraw.Draw(canvas)
        marked = rgb.copy(); ImageDraw.Draw(marked).rectangle((x-3, y-3, x+3, y+3), outline='red')
        canvas.paste(marked, (0, 40)); canvas.paste(rgb.crop((x-16, y-16, x+17, y+17)).resize((132, 132), Image.Resampling.NEAREST), (265, 40))
        draw.rectangle((265+16*4, 40+16*4, 265+17*4-1, 40+17*4-1), outline='red')
        draw.text((3, 4), f'{case} {name}: sole changed pixel ({x},{y}), RGB crop x4', fill='white')
        lines = [f'D1 depth {old:.6f}', f'D4 depth {new:.6f}', f'GT evaluator only {target:.6f}',
            f'abs error {abs(old-target):.6f} -> {abs(new-target):.6f}',
            f'fit cost {fit_old:.8f} -> {fit_new:.8f}',
            f'held0 {held_old[0]:.6f} -> {held_new[0]:.6f}', f'held1 {held_old[1]:.6f} -> {held_new[1]:.6f}']
        for j, line in enumerate(lines): draw.text((410, 43+20*j), line, fill='white')
        draw.text((3, 194), 'RGB cost acceptance is not independent geometry correctness; original thresholds unchanged.', fill='white')
        path = out/f'{case}_only_changed_pixel.png'; assert not path.exists()
        canvas.save(path); images[path.relative_to(ROOT).as_posix()] = sha(path)
        records[case] = {'image': name, 'uv': [int(x), int(y)], 'old_error': abs(old-target), 'new_error': abs(new-target),
            'fit_increase_within_original_1e6_tolerance': fit_new-fit_old, 'GT_evaluator_only': True}
    write(REPORT/'changed_pixel_review.json', {'images': images, 'records': records, 'source_sha256': sha(Path(__file__)),
        'original_outputs_and_gates_unchanged': True})


if __name__ == '__main__': main()
