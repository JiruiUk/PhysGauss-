"""Clarify original all-held versus D1-intersection displays, preserve originals."""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from variants.staged_reconstruction.prepare_data import ROOT, read, write, sha
from variants.staged_reconstruction.pretrained_depth.prepare import RUNS
from variants.staged_reconstruction.pretrained_depth.evaluate import heat
from experiments.staged_reconstruction.S1_pretrained_geometry.oracle_diagnostic.run import DIAG, check
from experiments.staged_reconstruction.S1_pretrained_geometry.oracle_diagnostic.core import oracle_shape


def main():
    check(); figures = {}
    out = ROOT/'output/staged_reconstruction/S1_pretrained_geometry/oracle_diagnostic'
    for case in ['diffuse', 'glossy']:
        meta = read(ROOT/f'data/derived/clean_static_synthetic/v2/{case}/train/cameras.json'); panels = []
        for i in [0, 24, 72]:
            row = meta['frames'][i]
            with np.load(RUNS/case/f'view_{i:03d}.npz') as d: raw = d['raw_relative']
            with np.load(ROOT/f'data/derived/clean_static_synthetic/v2/{case}/evaluation_only/gt'/(Path(row['image']).stem+'.npz')) as gt:
                target = gt['static_depth']; visible = gt['observed_static_visible'] & (target > 0)
                rgb = (gt['observed_rgb']*255).round().astype('uint8')
            with np.load(ROOT/f'runs/pidg_staged_reconstruction/S1_depth_refinement/v1/{case}/view_{i:03d}.npz') as d: baseline = d['refined']
            oracle, held, _ = oracle_shape(raw, target, visible)
            selected = held & (oracle > 0); same = selected & (baseline > 0)
            tiles = [('Input RGB', rgb), ('Privileged ALL held error', heat(abs(oracle-target), selected, .1)),
                ('Privileged SAME D1-pixel error', heat(abs(oracle-target), same, .1)),
                ('D1 SAME-pixel error', heat(abs(baseline-target), same, .1))]
            panel = Image.new('RGB', (1024, 180), '#171923'); draw = ImageDraw.Draw(panel)
            for j, (label, image) in enumerate(tiles):
                panel.paste(Image.fromarray(image), (256*j, 36)); draw.text((256*j+2, 2), case+' '+Path(row['image']).stem, fill='white')
                draw.text((256*j+2, 18), label+' 0..0.10' if j else label, fill='white')
            panels.append(np.asarray(panel))
        path = out/f'{case}_matched_pixel_review.png'; assert not path.exists()
        Image.fromarray(np.concatenate(panels)).save(path); figures[path.relative_to(ROOT).as_posix()] = sha(path)
    write(DIAG/'figure_clarification.json', {'images': figures, 'original_figures_and_metrics_preserved': True,
        'clarification': 'Original oracle panel shows all held pixels; original D1 panel shows intersection. Added explicit all/intersection/intersection panels for direct comparison. Black is missing, not accuracy.',
        'source_sha256': sha(Path(__file__))})


if __name__ == '__main__': main()
