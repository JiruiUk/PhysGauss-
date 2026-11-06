"""Fixed RGB/raw/calibration/boundary panels, no new selection or optimization."""
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from matplotlib import colormaps
from variants.staged_reconstruction.i10_pseudo_depth import io
from variants.staged_reconstruction.i10_pseudo_depth.core import depth_at


def color(values,valid=None):
    good=np.isfinite(values) if valid is None else (valid&np.isfinite(values))
    scaled=np.zeros(values.shape,float)
    if good.any():
        lo,hi=np.quantile(values[good],[.02,.98]);scaled[good]=np.clip((values[good]-lo)/max(hi-lo,1e-8),0,1)
    result=(colormaps['viridis'](scaled)[...,:3]*255).astype(np.uint8);result[~good]=0
    return Image.fromarray(result)


def main():
    io.verify();p=io.read(io.BASE/'protocol.json');audit=io.read(io.REPORT/'audit.json');paths=[]
    io.OUT.mkdir(parents=True,exist_ok=True)
    supplement=[next((i for i,c in enumerate(audit['calibration']) if c['passed']),None)]
    supplement=[i for i in supplement if i is not None and i not in p['fixed_review_frames']]
    for index in p['fixed_review_frames']+supplement:
        with np.load(io.RUNS/'raw'/f'{index:03d}.npz') as f:raw=f['relative']
        c=audit['calibration'][index];observed=Image.fromarray(io.rgb(index));draw=ImageDraw.Draw(observed)
        fit=set(c['fit_ids'])
        for j,(x,y) in enumerate(c['uv']):draw.ellipse((x-2,y-2,x+2,y+2),outline='#20caff' if j in fit else '#ff50dd')
        metric=np.full(raw.shape,np.nan)
        if c['coefficients'] is not None:
            inverse=c['coefficients'][0]*raw.astype(float)+c['coefficients'][1]
            positive=np.isfinite(inverse)&(inverse>0);metric[positive]=1/inverse[positive]
        targets=Image.fromarray(io.rgb(index));d=ImageDraw.Draw(targets)
        for row in audit['eligible_fit_observations'].get(str(index),[]):
            x,y=row['uv'];d.ellipse((x-3,y-3,x+3,y+3),outline='#30ff50',width=2)
        tiles=[observed,color(raw),color(metric),targets]
        titles=['RGB: cyan=scale fit; magenta=scale check','Small relative inverse depth (per-image display scale)',
                'Affine depth: display only, NOT dense reliable supervision','Green=admitted even-frame sparse targets; none if unknown']
        canvas=Image.new('RGB',(800,540),'#151821');d=ImageDraw.Draw(canvas)
        for j,(tile,title) in enumerate(zip(tiles,titles)):
            x,y=(j%2)*400,35+(j//2)*250;canvas.paste(tile,(x,y+20));d.text((x+3,y+3),title,fill='white')
        d.text((5,5),f'L7-P train index{index:03d} | calibration {c["passed"]}: {c["reason"]} | no geometry GT',fill='white')
        path=io.OUT/f'prior_audit_{index:03d}.png';canvas.save(path);paths.append(path)
    canvas=Image.new('RGB',(1000,300),'#151821');d=ImageDraw.Draw(canvas)
    summary=audit['summary']
    lines=['L7-P full-denominator temporal coverage (counts, NOT surface area)',
        'Band                 early[0,40)     middle[40,80)     late[80,120)',
        'Calibrated views     '+str(summary['calibrated_views_by_third']),
        'Reliable track starts '+str(summary['reliable_tracks_by_third']),
        'Even targets          '+str(summary['auxiliary_observations_by_third']),
        'Original 730 tracks, 222 motion candidates, 4433 seed trajectories retained.',
        'Parent-compensated consistency and tracker visibility are proxies, not GT.',
        'Local gate passed: '+str(summary['local_evidence_gate_passed'])]
    for i,line in enumerate(lines):d.text((12,10+i*33),line,fill='white')
    path=io.OUT/'temporal_coverage.png';canvas.save(path);paths.append(path)
    io.write(io.REPORT/'figures.json',dict(images_sha256={v.relative_to(io.ROOT).as_posix():io.sha(v) for v in paths},
        no_new_Gaussian_training=True,supplemental_first_passing_calibration=supplement,source_sha256=io.sha(Path(__file__))))
    print('L7_FIXED_FIVE_REVIEW_PANELS')


if __name__=='__main__':main()
