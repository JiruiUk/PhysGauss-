"""Single revised S2 candidate, preserving the original frozen RGB-only attempt."""
import argparse
import numpy as np
from scipy.spatial import cKDTree
from ..prepare_data import ROOT,REPORT,RUNS,DATA,read,write,sha
from output.short_clip_diagnostic.camera_math import w2c,intrinsic,project

REC_REPORT=REPORT/'S2_recovery';REC_RUNS=RUNS/'S2_recovery'


def main():
    p=argparse.ArgumentParser();p.add_argument('--enable-recovery',action='store_true');args=p.parse_args()
    if not args.enable_recovery:raise SystemExit('S2 repair is opt-in')
    from ..stage2.freeze import verify
    verify();REC_REPORT.mkdir(parents=True,exist_ok=True);REC_RUNS.mkdir(parents=True,exist_ok=True)
    if (REC_REPORT/'S2_protocol.json').exists():raise SystemExit('Recovery protocol already prepared; do not overwrite it')
    protocol=read(REPORT/'S2_protocol.json');protocol.update({'version':2,'candidate':'S2_recovery_one_candidate',
        'loss':'Original RGB +0.5 visible-yellow RGB +2.0 conservative one-sided image coverage; no physics/flow/track/contact',
        'recovery_enabled':True,'coverage_weight':2.,'foreground_rgb_weight':.5,
        'repair_trigger':'RGB-only attempt misses dynamic objects despite lower global image loss; original attempt/source/checkpoint preserved',
        'identity_claim':'Qualified early component mapping only; otherwise permutation-invariant unknown union. No occlusion re-ID',
        'depth_prior':'Per-object front static-neighborhood depth minus0.25, with continuous uniform depth slab; not depth GT',
        'comparison_limit':'Depth prior and image objective both change; this is a repair, not a single-factor causal comparison'})
    write(REC_REPORT/'S2_protocol.json',protocol)
    meta=read(DATA/'transforms_train.json');frame=meta['frames'][0];view=w2c(frame['transform_matrix'],'metadata_cv')
    K=intrinsic(meta['camera_angle_x'],'metadata',400,225);inverse=np.linalg.inv(view)
    static=np.load(RUNS/'static_metadata_initialization.npz')['xyz'];uv,z=project(static,view,K)
    eligible=(z>0)&np.isfinite(uv).all(1);tree=cKDTree(uv[eligible]);depths=z[eligible]
    rng=np.random.default_rng(protocol['seed']);objects=[]
    for entry in read(REPORT/'S2_data_receipt.json')['objects']:
        with np.load(RUNS/entry['file']) as data:
            world=data['xyz']+data['center'];color=data['rgb']
        pixels,_=project(world,view,K);center_pixel=np.median(pixels,axis=0)
        distance,indices=tree.query(center_pixel,k=64);depth_prior=float(np.min(depths[indices])-.25)
        rays=np.c_[pixels,np.ones(len(pixels))]@np.linalg.inv(K).T
        camera_points=rays*(depth_prior+rng.uniform(-.12,.12,len(rays)))[:,None]
        updated=camera_points@inverse[:3,:3].T+inverse[:3,3];center=updated.mean(0)
        file=entry['file'];spacing=depth_prior/K[0,0]*.9
        np.savez_compressed(REC_RUNS/file,xyz=(updated-center).astype(np.float32),rgb=color,spacing=np.asarray(spacing),center=center.astype(np.float32))
        objects.append({'object_id':entry['object_id'],'file':file,'sha256':sha(REC_RUNS/file),'points':len(updated),
            'initial_depth_prior':depth_prior,'nearest_static_projection_median_px':float(np.median(distance))})
    write(REC_REPORT/'S2_data_receipt.json',{'objects':objects,'protocol_sha256':sha(REC_REPORT/'S2_protocol.json'),
        'source_attempt_receipt_sha256':sha(REPORT/'S2_data_receipt.json'),'source_identity_sha256':sha(REPORT/'S2_identity.json'),
        'monocular_depth_prior_not_GT':True,'no_test_images_used_in_initialization':True})
    print('S2_RECOVERY_PREPARED',objects,flush=True)


if __name__=='__main__':main()
