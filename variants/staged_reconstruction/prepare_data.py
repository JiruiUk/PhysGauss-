"""Training-only conditional static visual hull; no learned labels or GT claims."""
from pathlib import Path
import hashlib, json
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_erosion
from output.short_clip_diagnostic.camera_math import w2c, intrinsic, project
from .model import observation_masks, eligible_voxels

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT/'experiments/staged_reconstruction'
RUNS = ROOT/'runs/pidg_staged_reconstruction'
DATA = ROOT/'data/derived/kinematic_resplit/training'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')


def read(path): return json.loads(path.read_text(encoding='utf-8-sig'))


def image_data(frame, width=400):
    path = DATA/(frame['file_path']+'.png')
    with Image.open(path) as im:
        rgba = np.asarray(im.convert('RGBA').resize((width,width*9//16),Image.Resampling.BILINEAR),dtype=np.float32)/255
    pos, neg = observation_masks(rgba, max(2,width//100))
    return rgba[...,:3]*rgba[...,3:4], pos, neg


def main():
    REPORT.mkdir(parents=True,exist_ok=True); RUNS.mkdir(parents=True,exist_ok=True)
    protocol = read(REPORT/'S1_protocol.json')
    output = RUNS/protocol.get('initialization_file','static_initialization.npz')
    if output.exists():
        receipt = read(REPORT/'data_receipt.json')
        assert sha(output) == receipt['initialization_sha256']
        assert sha(REPORT/'S1_protocol.json') == receipt['protocol_sha256']
        return
    meta = read(DATA/'transforms_train.json'); width=400
    mode=protocol['camera_contract']
    k = intrinsic(meta['camera_angle_x'],'metadata' if mode=='metadata_cv' else 'legacy',width,width*9//16)
    audit_path=ROOT/protocol['world_bound_camera_audit']
    anchors=np.asarray(read(audit_path)['metadata' if mode=='metadata_cv' else 'legacy']['xyz'])
    low,high=anchors.min(0)-.5,anchors.max(0)+.5
    spacing=protocol['voxel_spacing']
    axes=[np.arange(a,b+spacing/2,spacing,dtype=np.float32) for a,b in zip(low,high)]
    shape=tuple(map(len,axes))
    points = np.stack(np.meshgrid(*axes,indexing='ij'),axis=-1).reshape(-1,3)
    positive=np.zeros(len(points),np.int32); negative=positive.copy(); seen=positive.copy()
    thirds=np.zeros((len(points),3),np.int32); colors=np.zeros((len(points),3),np.float64)
    hashes={}; panels=[]
    for index,frame in enumerate(meta['frames']):
        number=int(Path(frame['file_path']).stem); target,pos,neg=image_data(frame,width)
        hashes[frame['file_path']]=sha(DATA/(frame['file_path']+'.png'))
        uv,depth=project(points,w2c(frame['transform_matrix'],mode),k)
        good=(depth>.2)&np.isfinite(uv).all(-1)&(uv[:,0]>=0)&(uv[:,0]<width)&(uv[:,1]>=0)&(uv[:,1]<width*9//16)
        rows=np.flatnonzero(good); xy=uv[rows].astype(int)
        p=pos[xy[:,1],xy[:,0]]; n=neg[xy[:,1],xy[:,0]]
        positive[rows]+=p; negative[rows]+=n; seen[rows]+=1
        thirds[rows,min((number-1)*3//200,2)]+=p
        colors[rows]+=target[xy[:,1],xy[:,0]]*p[:,None]
        if number in [2,86,188]:
            panel=(target*255).astype(np.uint8); overlay=panel.copy()
            overlay[pos]=[30,210,130];overlay[neg]=[40,40,40]
            panels.append(np.concatenate([panel,overlay],axis=1))
        if (index+1)%30==0: print('STATIC_CARVING',index+1,'/',len(meta['frames']),flush=True)
    keep=eligible_voxels(positive,negative,seen,thirds)
    volume=keep.reshape(shape)
    shell=volume & ~binary_erosion(volume)
    rows=np.flatnonzero(shell.reshape(-1))
    assert len(rows)>=100, f'No supported static hull: {len(rows)} surface points'
    if len(rows)>protocol['max_static_points']:
        rows=np.sort(np.random.default_rng(protocol['seed']).choice(rows,protocol['max_static_points'],replace=False))
    rgb=colors[rows]/np.maximum(positive[rows,None],1)
    np.savez_compressed(output, xyz=points[rows],rgb=rgb.astype(np.float32),spacing=np.asarray(spacing),
                        support=positive[rows],third_support=thirds[rows],voxel_rows=rows)
    if panels: Image.fromarray(np.concatenate(panels,axis=0)).save(REPORT/'training_masks.png')
    write(REPORT/'data_receipt.json',{'initialization_sha256':sha(output),'protocol_sha256':sha(REPORT/'S1_protocol.json'),
        'train_metadata_sha256':sha(DATA/'transforms_train.json'),'training_image_sha256':hashes,
        'volume_voxels':int(keep.sum()),'surface_points':len(rows),'spacing':spacing,
        'world_bounds':[low.tolist(),high.tolist()],'grid_shape':shape,
        'world_bound_source_sha256':sha(audit_path),'initialization_file':str(output),
        'minimum_positive_views':int(positive[rows].min()),'training_only':True,
        'labels':'Conditional blue appearance candidates checked for multitime geometric support; not physical/material GT',
        'unknown_foreground_and_yellow_occlusion':'excluded from negative supervision',
        'test_images_used':False})
    print('STATIC_INITIALIZATION',len(rows),'points;',int(keep.sum()),'volume voxels',flush=True)


if __name__=='__main__':main()
