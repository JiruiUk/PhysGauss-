"""Rebuild S1's train-only voxel solid hypothesis without changing S1 inputs."""
import argparse
import numpy as np
from scipy.ndimage import distance_transform_edt
from ..prepare_data import ROOT,REPORT,RUNS,DATA,read,write,sha,image_data
from ..model import eligible_voxels
from output.short_clip_diagnostic.camera_math import w2c,intrinsic,project


def main():
    p=argparse.ArgumentParser();p.add_argument('--enable-static-volume-prior',action='store_true');args=p.parse_args()
    if not args.enable_static_volume_prior:raise SystemExit('Static volume prior is opt-in')
    from ..freeze import verify
    verify();report=REPORT/'S5';report.mkdir(parents=True,exist_ok=True)
    receipt=read(REPORT/'data_receipt.json');low,high=np.asarray(receipt['world_bounds']);spacing=receipt['spacing']
    axes=[np.arange(a,b+spacing/2,spacing,dtype=np.float32) for a,b in zip(low,high)];shape=tuple(map(len,axes))
    assert list(shape)==receipt['grid_shape']
    points=np.stack(np.meshgrid(*axes,indexing='ij'),axis=-1).reshape(-1,3)
    initial=np.load(RUNS/'static_metadata_initialization.npz',allow_pickle=False)
    assert np.allclose(points[initial['voxel_rows']],initial['xyz'],atol=1e-6)
    positive=np.zeros(len(points),np.int32);negative=positive.copy();seen=positive.copy();thirds=np.zeros((len(points),3),np.int32)
    meta=read(DATA/'transforms_train.json');K=intrinsic(meta['camera_angle_x'],'metadata',400,225)
    for index,frame in enumerate(meta['frames']):
        assert sha(DATA/(frame['file_path']+'.png'))==receipt['training_image_sha256'][frame['file_path']]
        _,pos,neg=image_data(frame,400);uv,z=project(points,w2c(frame['transform_matrix'],'metadata_cv'),K)
        valid=(z>.2)&np.isfinite(uv).all(1)&(uv[:,0]>=0)&(uv[:,0]<400)&(uv[:,1]>=0)&(uv[:,1]<225)
        rows=np.flatnonzero(valid);xy=uv[rows].astype(int);hit=pos[xy[:,1],xy[:,0]]
        positive[rows]+=hit;negative[rows]+=neg[xy[:,1],xy[:,0]];seen[rows]+=1
        number=int(frame['file_path'].split('/')[-1]);thirds[rows,min((number-1)*3//200,2)]+=hit
    volume=eligible_voxels(positive,negative,seen,thirds).reshape(shape)
    assert int(volume.sum())==receipt['volume_voxels']
    field=(distance_transform_edt(~volume)-distance_transform_edt(volume))*spacing
    path=RUNS/'S5_conditional_static_sdf.npz'
    np.savez_compressed(path,sdf=field.astype(np.float32),volume=volume,origin=low.astype(np.float32),spacing=spacing)
    write(report/'static_volume_receipt.json',{'file':str(path.relative_to(ROOT)).replace('\\','/'),'sha256':sha(path),
        'S1_initialization_receipt_sha256':sha(REPORT/'data_receipt.json'),'shape':shape,'solid_voxels':int(volume.sum()),
        'discretization_margin':2*spacing,'maximum_interior_distance':float(-field.min()),
        'training_only':True,'identical_S1_voxel_eligibility':True,'independent_geometry_GT':False,
        'contact_surface_certified':False,'permitted_use':'Conditional deep-interior nonpenetration prior on independently qualified visible tracks only',
        'not_permitted':'Physical contact-force/restitution/friction inference; unknown body identity or unseen surface as truth'})
    print('STATIC_VOLUME_PRIOR_PREPARED',int(volume.sum()),float(field.min()),flush=True)


if __name__=='__main__':main()
