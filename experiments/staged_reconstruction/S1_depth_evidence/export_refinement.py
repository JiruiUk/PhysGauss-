"""Evaluation-only native-size depth diagnostics, no Gaussian model or training."""
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw
from audit import normal_values
from variants.staged_reconstruction.prepare_data import ROOT,read,write,sha
from variants.staged_reconstruction.depth_refine.infer import REPORT,RUNS,SOURCE,verify


def colors(x,valid,maximum,lo=0):
    mapped = np.clip((x-lo)/max(maximum-lo,1e-12),0,1)
    color = cv2.cvtColor(cv2.applyColorMap((mapped*255).astype('uint8'),cv2.COLORMAP_TURBO),cv2.COLOR_BGR2RGB)
    color[~valid] = 0
    return color


def main():
    verify(); assert not (REPORT/'exports.json').exists()
    evaluation = read(REPORT/'evaluation.json'); inputs = read(REPORT/'inference.json')
    for p,h in inputs['outputs'].items(): assert sha(ROOT/p) == h,p
    for p,h in read(SOURCE/'dataset.json')['files'].items(): assert sha(ROOT/p) == h,p
    out = ROOT/'output/staged_reconstruction/S1_depth_evidence/refinement_v1'; out.mkdir(parents=True,exist_ok=True)
    files = {}
    for case in ['diffuse','glossy']:
        directory = ROOT/f'data/derived/clean_static_synthetic/v2/{case}'; meta = read(directory/'train/cameras.json')
        depth_rows,normal_rows = [],[]
        for i in [0,24,72]:
            row = meta['frames'][i]; view,k = np.asarray(row['world_to_camera']),np.asarray(meta['K'])
            with np.load(RUNS/case/f'view_{i:03d}.npz') as d: original,refined = d['original'].copy(),d['refined'].copy()
            with np.load(directory/'evaluation_only/gt'/(Path(row['image']).stem+'.npz')) as g:
                gt,gt_normal = g['static_depth'].copy(),g['static_normal'].copy()
            supported = original > 0; on = supported&(gt > 0); lo,hi = float(gt[gt > 0].min()),float(gt.max())
            _,eligible = normal_values(original,view,k,.025); same_face = gt[1:-1,1:-1] > 0
            for ay,ax in [(slice(1,-1),slice(2,None)),(slice(1,-1),slice(None,-2)),(slice(2,None),slice(1,-1)),(slice(None,-2),slice(1,-1))]:
                same_face &= (gt[ay,ax] > 0)&((gt_normal[1:-1,1:-1]*gt_normal[ay,ax]).sum(2) > .99)
            fixed = eligible&same_face
            title = f'{case} train{i} '+Path(row['image']).stem
            height,width = gt.shape
            canvas = Image.new('RGB',(4*width,height+36),'#171923'); draw = ImageDraw.Draw(canvas)
            values = [(f'orig D {lo:.2f}..{hi:.2f}',colors(original,supported,hi,lo)),
                ('refined D same scale',colors(refined,supported,hi,lo)),
                ('orig abs error 0..0.10',colors(np.abs(original-gt),on,.1)),
                ('refined abs error 0..0.10',colors(np.abs(refined-gt),on,.1))]
            for j,(name,array) in enumerate(values):
                canvas.paste(Image.fromarray(array),(j*width,36)); draw.text((j*width+2,2),title,fill='white'); draw.text((j*width+2,17),name,fill='white')
            depth_rows.append(np.asarray(canvas))
            canvas = Image.new('RGB',(2*width,height+36),'#171923'); draw = ImageDraw.Draw(canvas)
            for j,(name,depth) in enumerate([('orig normal error 0..90deg',original),('refined normal error 0..90deg',refined)]):
                normal,_ = normal_values(depth,view,k,.025)
                angle = np.degrees(np.arccos(np.clip(np.abs((normal*gt_normal[1:-1,1:-1]).sum(2)),0,1)))
                image = np.zeros((height,width,3),np.uint8); image[1:-1,1:-1] = colors(angle,fixed,90)
                canvas.paste(Image.fromarray(image),(j*width,36)); draw.text((j*width+2,2),title,fill='white'); draw.text((j*width+2,17),name,fill='white')
            normal_rows.append(np.asarray(canvas))
        for name,rows in [('depth',depth_rows),('normal_error',normal_rows)]:
            path = out/f'{case}_{name}.png'; assert not path.exists()
            Image.fromarray(np.concatenate(rows)).save(path); files[path.relative_to(ROOT).as_posix()] = sha(path)
    verify(); write(REPORT/'exports.json',{'images':files,'evaluation_sha256':sha(REPORT/'evaluation.json'),
        'source_sha256':sha(Path(__file__)),'black_pixels_are_missing_evidence_not_empty_geometry':True,
        'Gaussian_render_or_training':False,'normal_pixels_fixed_to_original_eligible_and_GT_single_face':True})
    print('DEPTH_REFINEMENT_DIAGNOSTICS_EXPORTED')


if __name__ == '__main__': main()
