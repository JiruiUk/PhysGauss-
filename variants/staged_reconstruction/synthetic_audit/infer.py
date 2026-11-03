"""RGB-only geometric admission audit. This is not full Gaussian training."""
import argparse
import json
import msvcrt
from pathlib import Path
import cv2
import numpy as np
import torch
from ..prepare_data import ROOT, read, write, sha
from ..adaptive_static.geometry import camera_center,select_views,patch_features,sweep,fuse_voxels
from ..adaptive_static.prepare import checked_maps,world_points
from ..clean_static.evidence import accumulation
from .input import load

REPORT = ROOT/'experiments/staged_reconstruction/S1_clean_visibility/synthetic_v2'
RUNS = ROOT/'runs/pidg_staged_reconstruction/S1_clean_visibility/synthetic_v2'


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--enable-synthetic-audit',action='store_true'); args = parser.parse_args()
    if not args.enable_synthetic_audit: raise SystemExit('Synthetic RGB inference is opt-in')
    assert not (RUNS/'stop.request').exists(), 'Saved user stop; do not resume'
    protocol = read(REPORT/'protocol.json'); torch.set_num_threads(4); cv2.setNumThreads(2)
    RUNS.mkdir(parents=True,exist_ok=True)
    files = list(Path(__file__).parent.glob('*.py'))+[REPORT/'protocol.json']
    files += [ROOT/p for p in read(REPORT/'dataset.json')['files'] if '/train/' in p]
    for prefix in ['adaptive_static','clean_static']:
        files += list((Path(__file__).parent.parent/prefix).glob('*.py'))
    manifest = {p.relative_to(ROOT).as_posix():sha(p) for p in files}
    if (REPORT/'inference_inputs.json').exists(): assert read(REPORT/'inference_inputs.json') == manifest
    else: write(REPORT/'inference_inputs.json',manifest)
    for case in ['diffuse','glossy']:
        directory = RUNS/case; directory.mkdir(exist_ok=True)
        if (directory/'reconstruction.npz').exists(): continue
        values = load(ROOT/f'data/derived/clean_static_synthetic/v2/{case}/train')
        images,views,k = values['rgb'],values['views'],values['K']
        ids = np.rint(values['times']*200).astype(int).tolist()
        center = camera_center(views); radius = float(np.median(np.linalg.norm(np.linalg.inv(views)[:,:3,3]-center,axis=1)))
        near,far = radius*.45,radius*1.55
        tensor = torch.from_numpy(images).permute(0,3,1,2).cuda(); features,texture = patch_features(tensor,5); del tensor
        tensor_views = torch.tensor(views,dtype=torch.float32,device='cuda'); tensor_k = torch.tensor(k,dtype=torch.float32,device='cuda')
        depths = 1/torch.linspace(1/near,1/far,protocol['depth_samples'],device='cuda')
        maps,sources = [],[]
        for i in range(len(images)):
            selected = select_views(i,views,ids); assert len(selected) == 6
            path = directory/f'raw_{i:03d}.npz'; sources.append(selected)
            if path.exists():
                with np.load(path) as data: depth_map = data['depth'].copy()
            else:
                depth_map = np.zeros(images.shape[1:3],np.float32); yy,xx = np.nonzero(texture[i].cpu().numpy())
                uv_all = np.c_[xx,yy]
                for offset in range(0,len(uv_all),protocol['pixel_chunk']):
                    uv = torch.tensor(uv_all[offset:offset+protocol['pixel_chunk']],dtype=torch.float32,device='cuda')
                    _,z,good,_,_,_ = sweep(uv,features,texture,tensor_views,tensor_k,depths,i,selected,
                        protocol['uniqueness_margin'],protocol['photometric_cost_limit'])
                    xy = uv[good].long().cpu().numpy(); depth_map[xy[:,1],xy[:,0]] = z[good].cpu().numpy()
                np.savez_compressed(path,depth=depth_map)
            maps.append(depth_map)
            write(RUNS/'state.json',{'status':'RGB_geometry_inference','case':case,'completed':i+1,'total':len(images)})
            if i%12 == 0: print('SYNTHETIC_RGB_SCAN',case,i+1,len(images),flush=True)
            if (RUNS/'stop.request').exists(): return
        checked,support,records = checked_maps(maps,sources,views,k,ids,protocol)
        positions,colors,evidence = [],[],[]
        for i,(uv,sup) in enumerate(support):
            xy = uv.astype(int); positions.append(world_points(uv,checked[i][xy[:,1],xy[:,0]],views[i],k))
            colors.append(images[i,xy[:,1],xy[:,0]]); evidence.append(sup)
        xyz,col,sup = np.concatenate(positions),np.concatenate(colors),np.concatenate(evidence)
        if len(xyz):
            xyz,col,sup = fuse_voxels(xyz,col,sup,protocol['reservoir_voxel_spacing'])
            depths_np = np.asarray(checked)
            result = accumulation(xyz,views,k,depths_np,depths_np > 0,protocol,ids)
            reject = result['quarantine']
            np.savez_compressed(directory/'visibility_evidence.npz',**result)
        else: reject = np.zeros(0,bool)
        np.savez_compressed(directory/'reconstruction.npz',xyz=xyz,rgb=col,support=sup,quarantine=reject)
        write(REPORT/f'{case}_inference.json',{'RGB_only':True,'complete_Gaussian_training':False,
            'points':len(xyz),'quarantined':int(reject.sum()),'per_view':records,'sources':sources,
            'camera_derived_depth_range':[near,far],'input_manifest_sha256':sha(REPORT/'inference_inputs.json'),
            'reconstruction_sha256':sha(directory/'reconstruction.npz')})
        del features,texture; torch.cuda.empty_cache()
    assert all(sha(ROOT/p) == expected for p,expected in manifest.items())
    write(RUNS/'state.json',{'status':'inferred_pending_independent_GT_evaluation'})


if __name__ == '__main__':
    RUNS.mkdir(parents=True,exist_ok=True)
    with (RUNS/'inference.lock').open('a+b') as lock:
        lock.seek(0)
        if not lock.read(1): lock.write(b'0'); lock.flush()
        lock.seek(0); msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        try: main()
        finally: lock.seek(0); msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)
