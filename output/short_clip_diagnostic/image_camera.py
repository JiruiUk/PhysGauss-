"""Independent image tracking and depth-free camera checks on fixed clips."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import torch
import cv2
from PIL import Image,ImageDraw
from scipy import ndimage
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from output.short_clip_diagnostic.camera_math import w2c,intrinsic,fundamental,sampson
from variants.flow_comparison.geometry import forward_backward_valid
REPORT=ROOT/'experiments/short_clip_diagnostic';OUT=ROOT/'output/short_clip_diagnostic'
TRAIN=ROOT/'data/derived/kinematic_resplit/training';FLOW=ROOT/'data/derived/flow_comparison_train_only'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,allow_nan=False),encoding='utf-8')
HYPOTHESES=[(p,k) for p in ['legacy','metadata_cv'] for k in ['legacy','metadata']]


def image(frame):
    with Image.open(TRAIN/(frame['file_path']+'.png')) as im:rgba=np.array(im.convert('RGBA').resize((400,225),Image.Resampling.BILINEAR))
    rgb=(rgba[...,:3].astype(float)*(rgba[...,3:4]/255)).round().astype('uint8')
    blue=(rgba[...,3]>242)&(rgb[...,2]>1.2*np.maximum(rgb[...,0],rgb[...,1]))&(rgb[...,2]>64)
    return rgb,ndimage.binary_erosion(blue,iterations=2)


def at(mask,points):
    xy=np.rint(points).astype(int);valid=(xy[:,0]>=0)&(xy[:,0]<400)&(xy[:,1]>=0)&(xy[:,1]<225)
    out=np.zeros(len(points),bool);out[valid]=mask[xy[valid,1],xy[valid,0]]
    return out


def sample_flow(flow,points):
    return np.stack([ndimage.map_coordinates(flow[...,j],[points[:,1],points[:,0]],order=1,mode='constant') for j in range(2)],1)


def lk(first,second,mask,target):
    a=cv2.cvtColor(first,cv2.COLOR_RGB2GRAY);b=cv2.cvtColor(second,cv2.COLOR_RGB2GRAY)
    p=cv2.goodFeaturesToTrack(a,300,.003,4,mask=mask.astype('uint8')*255)
    if p is None:return np.empty((0,2)),np.empty((0,2))
    settings=dict(winSize=(21,21),maxLevel=3,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01))
    q,status,error=cv2.calcOpticalFlowPyrLK(a,b,p,None,**settings)
    back,bs,be=cv2.calcOpticalFlowPyrLK(b,a,q,None,**settings)
    pp=p[:,0];qq=q[:,0];valid=(status[:,0]>0)&(bs[:,0]>0)&(np.linalg.norm(back[:,0]-pp,axis=1)<=.5)&(error[:,0]<=20)&(be[:,0]<=20)
    valid&=np.isfinite(qq).all(1)&at(target,qq)
    return pp[valid].astype(float),qq[valid].astype(float)


def statistics(values):
    values=np.asarray(values);values=values[np.isfinite(values)]
    return {'n':len(values),'median':float(np.median(values)) if len(values) else None,'p90':float(np.quantile(values,.9)) if len(values) else None,
            'fraction_below1px':float(np.mean(values<1)) if len(values) else None}


def main():
    torch.set_num_threads(4);cv2.setNumThreads(2);OUT.mkdir(parents=True,exist_ok=True)
    files=[Path(__file__),ROOT/'output/short_clip_diagnostic/camera_math.py',REPORT/'plan.json',ROOT/'tests/test_clip_camera_geometry.py',ROOT/'experiments/short_clip_audit/results.json']
    parent=read(ROOT/'experiments/short_clip_audit/input_lock.json');lock={**parent,**{str(p.relative_to(ROOT)):sha(p) for p in files}}
    assert all(sha(ROOT/p)==v for p,v in lock.items());write(REPORT/'input_lock.json',lock)
    clips=read(ROOT/'experiments/short_clip_audit/results.json')['selected'];meta=read(TRAIN/'transforms_train.json');lookup={int(Path(f['file_path']).stem):f for f in meta['frames']}
    masks=np.load(ROOT/'output/surface_support/image_tracks.npz');component={int(n):c for n,c in zip(masks['frames'],masks['components'])}
    manifest=read(FLOW/'manifest.json');flowlookup={(r['source'],r['target']):r for r in manifest['pairs']}
    numbers=sorted({n for c in clips for n in c['frames']});images={n:image(lookup[n]) for n in numbers}
    pairs=sorted({(a,b) for c in clips for a,b in zip(c['frames'][:-1],c['frames'][1:])});rows=[];saved={};camera_values={}
    for n in numbers:
        camera_values[str(n)]={}
        for pose,k in HYPOTHESES:camera_values[str(n)][pose+'__'+k]={'w2c':w2c(lookup[n]['transform_matrix'],pose).tolist(),'K':intrinsic(meta['camera_angle_x'],k).tolist()}
    for source,target in pairs:
        first,blue0=images[source];second,blue1=images[target]
        with np.load(FLOW/flowlookup[(source,target)]['file']) as v:f=v['forward'].astype(np.float32);b=v['backward'].astype(np.float32)
        fb=forward_backward_valid(torch.tensor(f),torch.tensor(b)).numpy();y,x=np.where(blue0&fb);keep=(x%4==0)&(y%4==0);a=np.stack([x[keep],y[keep]],1).astype(float);q=a+f[y[keep],x[keep]]
        good=at(blue1,q);a=a[good];q=q[good]
        la,lq=lk(first,second,blue0,blue1)
        duck0=ndimage.binary_erosion(component[source]>=0,iterations=2);duck1=ndimage.binary_erosion(component[target]>=0,iterations=2)
        da,dq=lk(first,second,duck0,duck1)
        row={'source':source,'target':target,'blue_unimatch_count':len(a),'blue_LK_count':len(la),'duck_LK_count':len(da),
          'blue_LK_vs_raw_flow_EPE':statistics(np.linalg.norm((lq-la)-sample_flow(f,la),axis=1)),
          'duck_LK_vs_raw_flow_EPE':statistics(np.linalg.norm((dq-da)-sample_flow(f,da),axis=1)),'camera_scores':{}}
        for method,aa,qq in [('raw_blue',a,q),('LK_blue',la,lq)]:
            saved[f'{source}_{target}_{method}_source']=aa;saved[f'{source}_{target}_{method}_target']=qq
            row['camera_scores'][method]={}
            for pose,k in HYPOTHESES:
                key=pose+'__'+k;ca=camera_values[str(source)][key];cb=camera_values[str(target)][key]
                fm=fundamental(np.array(ca['w2c']),np.array(cb['w2c']),np.array(ca['K']),np.array(cb['K']))
                row['camera_scores'][method][key]=statistics(sampson(aa,qq,fm))
        saved[f'{source}_{target}_duck_LK_source']=da;saved[f'{source}_{target}_duck_LK_target']=dq
        rows.append(row);print('camera pair',source,target,'LK blue',len(la),'raw',len(a),row['camera_scores']['raw_blue'],flush=True)
    summaries=[]
    for clip in clips:
        keys=set(zip(clip['frames'][:-1],clip['frames'][1:]));rr=[r for r in rows if (r['source'],r['target']) in keys];scores={}
        for method in ['raw_blue','LK_blue']:
            scores[method]={}
            for pose,k in HYPOTHESES:
                key=pose+'__'+k;medians=[r['camera_scores'][method][key]['median'] for r in rr if r['camera_scores'][method][key]['n']>=8]
                scores[method][key]={'qualified_pairs_n_ge8':len(medians),'median_of_pair_medians_pixels':float(np.median(medians)) if medians else None}
        summaries.append({'clip':clip['name'],'frames':clip['frames'],'camera_scores':scores,
             'LK_vs_flow_median_of_pair_medians':{region:float(np.median([r[region+'_LK_vs_raw_flow_EPE']['median'] for r in rr if r[region+'_LK_vs_raw_flow_EPE']['n']>=8])) if any(r[region+'_LK_vs_raw_flow_EPE']['n']>=8 for r in rr) else None for region in ['blue','duck']}})
    np.savez_compressed(OUT/'image_correspondences.npz',**saved)
    write(REPORT/'camera_values.json',camera_values);write(REPORT/'image_camera.json',{'pairs':rows,'clips':summaries,'unique_pairs':len(pairs),
      'conditional_static_blue_assumption':True,'physical_camera_GT_claim':False,'LK_is_not_material_GT':True,'new_model_training_steps':0})
    assert all(sha(ROOT/p)==v for p,v in lock.items());print('IMAGE CAMERA CHECK COMPLETE',flush=True)


if __name__=='__main__':main()
