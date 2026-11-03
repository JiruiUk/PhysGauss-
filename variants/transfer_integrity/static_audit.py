"""Independent training-RGB static-blue hypothesis check; no model imports."""
from pathlib import Path
import sys
import json
import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from output.short_clip_diagnostic.camera_math import w2c, intrinsic, project

REPORT = ROOT/'experiments/transfer_integrity'
OUT = ROOT/'output/transfer_integrity'
TRAIN = ROOT/'data/derived/kinematic_resplit/training'


def images(frame, core=True):
    with Image.open(TRAIN/(frame['file_path']+'.png')) as im:
        rgba = np.array(im.convert('RGBA').resize((400,225), Image.Resampling.BILINEAR))
    rgb = (rgba[...,:3].astype(float)*rgba[...,3:4]/255).round().astype('uint8')
    blue = (rgba[...,3]>250)&(rgb[...,2]>1.2*np.maximum(rgb[...,0],rgb[...,1]))&(rgb[...,2]>64)
    yellow = (rgba[...,3]>245)&(rgb[...,0]>64)&(rgb[...,1]>46)&(rgb[...,2]<.8*np.minimum(rgb[...,0],rgb[...,1]))
    return (rgb, ndimage.binary_erosion(blue, iterations=2), ndimage.binary_erosion(yellow, iterations=1)) if core else (rgb,blue,yellow)


def inside(mask, xy):
    q=np.rint(xy).astype(int)
    valid=np.isfinite(xy).all(1)&(q[:,0]>=0)&(q[:,0]<400)&(q[:,1]>=0)&(q[:,1]<225)
    result=np.zeros(len(q), bool)
    result[valid]=mask[q[valid,1],q[valid,0]]
    return result


def triangulate(observations, matrices):
    rows=[]
    for n,p in observations:
        a=matrices[n]
        rows.extend([p[0]*a[2]-a[0],p[1]*a[2]-a[1]])
    _,_,vh=np.linalg.svd(np.asarray(rows),full_matrices=False)
    return vh[-1,:3]/vh[-1,3]


def main():
    cv2.setNumThreads(2)
    OUT.mkdir(parents=True, exist_ok=True); REPORT.mkdir(parents=True, exist_ok=True)
    meta=json.loads((TRAIN/'transforms_train.json').read_text())
    frames=sorted(meta['frames'],key=lambda f:int(Path(f['file_path']).stem))
    ids=[int(Path(f['file_path']).stem) for f in frames]
    data={n:images(f) for n,f in zip(ids,frames)}
    views={n:w2c(f['transform_matrix'],'legacy') for n,f in zip(ids,frames)}
    k=intrinsic(meta['camera_angle_x'],'legacy')
    matrices={n:k@v[:3] for n,v in views.items()}
    centers={n:np.linalg.inv(v)[:3,3] for n,v in views.items()}
    active=[];complete=[]
    settings=dict(winSize=(21,21),maxLevel=3,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01))
    for index,n in enumerate(ids):
        gray=cv2.cvtColor(data[n][0],cv2.COLOR_RGB2GRAY)
        if index and active:
            p=np.float32([t[-1][1] for t in active]).reshape(-1,1,2)
            q,st,er=cv2.calcOpticalFlowPyrLK(previous,gray,p,None,**settings)
            back,bs,be=cv2.calcOpticalFlowPyrLK(gray,previous,q,None,**settings)
            good=(st[:,0]>0)&(bs[:,0]>0)&(er[:,0]<20)&(be[:,0]<20)&(np.linalg.norm(back[:,0]-p[:,0],axis=1)<.5)&inside(data[n][1],q[:,0])
            keep=[]
            for track,point,ok in zip(active,q[:,0],good):
                if ok:keep.append(track+[(n,point.astype(float))])
                else:complete.append(track)
            active=keep
        if index%8==0:
            mask=data[n][1].astype('uint8')*255
            for t in active:
                cv2.circle(mask,tuple(np.rint(t[-1][1]).astype(int)),5,0,-1)
            pp=cv2.goodFeaturesToTrack(gray,120,.003,4,mask=mask)
            if pp is not None:active.extend([[(n,p.astype(float))] for p in pp[:,0]])
        previous=gray
    complete+=active
    qualified=[];rejected=[]
    for track in complete:
        if len(track)<8:continue
        fit=track[::2];check=track[1::2]
        x=triangulate(fit,matrices)
        errors=[];depth=[];directions=[]
        for n,p in track:
            uv,z=project(x[None],views[n],k)
            errors.append(float(np.linalg.norm(uv[0]-p)));depth.append(z[0])
            direction=x-centers[n];directions.append(direction/np.linalg.norm(direction))
        direction=np.asarray(directions)
        angle=float(np.degrees(np.arccos(np.clip((direction@direction.T).min(),-1,1))))
        err=np.asarray(errors)[1::2]
        ok=np.isfinite(x).all() and min(depth)>.2 and angle>=2 and np.median(err)<=1.5 and np.quantile(err,.9)<=3
        row={'frames':[n for n,p in track], 'uv':[p.tolist() for n,p in track], 'xyz':x.tolist(),
             'held_median_px':float(np.median(err)), 'held_p90_px':float(np.quantile(err,.9)), 'ray_baseline_degrees':angle}
        (qualified if ok else rejected).append(row)
    thirds=[sum(any(lo<=n<=hi for n in r['frames']) for r in qualified) for lo,hi in [(1,67),(68,134),(135,200)]]
    passed=len(qualified)>=12 and min(thirds)>0
    result={'static_hypothesis_gate_passed':passed,'qualified_tracks':len(qualified),'tracks_with_at_least8_observations':len(qualified)+len(rejected),
            'time_third_track_support':thirds,'held_median_px':float(np.median([r['held_median_px'] for r in qualified])) if qualified else None,
            'camera_convention':'Actual trained loader, no fitting', 'stationarity_is_conditional_prior_not_physical_GT':True,
            'tracks':qualified,'rejected_track_count':len(rejected)}
    (REPORT/'static_image_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    selected=[2,50,86,134,172,200]
    panel=Image.new('RGB',(1200,500),'#181c24');draw=ImageDraw.Draw(panel)
    for index,n in enumerate(selected):
        if n not in data:continue
        im=Image.fromarray(data[n][0]);d=ImageDraw.Draw(im)
        for r in qualified:
            if n in r['frames']:
                x,y=r['uv'][r['frames'].index(n)];d.ellipse((x-2,y-2,x+2,y+2),fill='#ff5555')
        col,row=index%3,index//3
        draw.text((col*400+8,row*250+5),f'GT training frame {n}; qualified static tracks',fill='white')
        panel.paste(im,(col*400,row*250+25))
    panel.save(OUT/'static_image_audit.png')
    print(json.dumps({k:v for k,v in result.items() if k!='tracks'},indent=2),flush=True)
    if not passed:raise RuntimeError('Static-blue hypothesis not sufficiently supported; do not enable static losses')


if __name__=='__main__':main()
