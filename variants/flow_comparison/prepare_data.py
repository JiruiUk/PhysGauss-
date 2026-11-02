"""UniMatch flow for actual TRAIN pairs, camera compensation with frozen H3 depth."""
import hashlib,json,os,runpy,sys,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from .geometry import motion_target,forward_backward_valid,pixel_grid,sample
ROOT=Path(__file__).resolve().parents[2]
REPORT=ROOT/'experiments/flow_comparison';CACHE=ROOT/'data/derived/flow_comparison_train_only'
SOURCE=ROOT/'data/derived/kinematic_resplit/training'
TEACHER=ROOT/'runs/pidg_integrity_tuning/H3_combined/rubber_duck'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def prepare():
    CACHE.mkdir(parents=True,exist_ok=True);(CACHE/'depth').mkdir(exist_ok=True);(CACHE/'pairs').mkdir(exist_ok=True)
    lock=read(REPORT/'flow_model_lock.json');metadata=read(SOURCE/'transforms_train.json')
    frames=sorted(metadata['frames'],key=lambda f:int(Path(f['file_path']).stem));ids=[int(Path(f['file_path']).stem) for f in frames]
    if len(ids)!=120 or len(set(ids))!=120:raise ValueError('Expected120 training frames only')
    hashes={str((SOURCE/(f['file_path']+'.png')).relative_to(ROOT)):sha(SOURCE/(f['file_path']+'.png')) for f in frames}
    for p in [TEACHER/'point_cloud/iteration_50000/point_cloud.ply',TEACHER/'point_cloud/iteration_50000/material.pt',TEACHER/'deform/iteration_50000/deform.pth',SOURCE/'transforms_train.json']:
        hashes[str(p.relative_to(ROOT))]=sha(p)
    generation={'input_sha256':hashes,'model_lock':lock,
      'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),Path(__file__).with_name('geometry.py')]}}
    input_lock=CACHE/'input_lock.json'
    if input_lock.exists():
        if read(input_lock)!=generation:raise ValueError('Partial cache has different source/input lock')
    else:input_lock.write_text(json.dumps(generation,indent=2),encoding='utf8')
    if (CACHE/'manifest.json').exists():
        m=read(CACHE/'manifest.json')
        if m['input_sha256']!=hashes or m['model_lock']!=lock:raise ValueError('Cached targets have changed inputs')
        if any(sha(CACHE/k)!=v for k,v in m['output_sha256'].items()):raise ValueError('Cached flow/depth changed')
        print('FLOW CACHE VERIFIED',flush=True);return CACHE
    os.environ['PIDG_HASH_SOURCE_DIR']=str(ROOT/'third_party/PIDG/hashencoder');os.environ['PIDG_HASH_BUILD_DIR']=str(ROOT/'third_party/PIDG/tmp_build')
    sys.path.insert(0,str(ROOT/'variants/integrity_tuning/code'))
    from scene_PIDG.gaussian_model import GaussianModel
    from scene_PIDG.deform_model import DeformModel
    from scene_PIDG.cameras import Camera
    from gaussian_renderer import render
    from utils.graphics_utils import focal2fov,fov2focal
    from types import SimpleNamespace
    model_cfg=runpy.run_path(str(TEACHER/'model_config.py'))
    g=GaussianModel(3);g.load_ply(str(TEACHER/'point_cloud/iteration_50000/point_cloud.ply'))
    d=DeformModel(model_cfg['grid_args'],model_cfg['network_args'],reg_spatial_able=False,reg_temporal_able=False);d.load_weights(str(TEACHER),50000)
    for module in [d.grid4d,d.deform]:module.eval().requires_grad_(False)
    pipe=SimpleNamespace(compute_cov3D_python=False,convert_SHs_python=False,debug=False)
    background=torch.zeros(3,device='cuda');camera_data={};images={};start=time.monotonic()
    with torch.no_grad():
        for index,frame in enumerate(frames):
            fid=int(Path(frame['file_path']).stem);name=f'{fid:04d}'
            with Image.open(SOURCE/(frame['file_path']+'.png')) as im:rgba=np.array(im.convert('RGBA'))
            rgb=(rgba[...,:3].astype(np.float32)*rgba[...,3:4]/255).astype(np.uint8)
            tensor=torch.from_numpy(rgb.copy()).permute(2,0,1).float()[None].cuda()
            low=F.interpolate(tensor/255,size=(225,400),mode='bilinear',align_corners=False)[0]
            images[fid]=F.interpolate(tensor,size=(448,800),mode='bilinear',align_corners=False).cpu()
            m=np.linalg.inv(np.array(frame['transform_matrix']));R=-m[:3,:3].T;R[:,0]=-R[:,0];T=-m[:3,3]
            fovx=metadata['camera_angle_x'];fovy=focal2fov(fov2focal(fovx,1600),900)
            cam=Camera(fid,R,T,fovx,fovy,low,None,name,fid,data_device='cuda',fid=(fid-1)/199)
            target=CACHE/'depth'/f'{name}.npz'
            if not target.exists():
                out=d.step(g.get_xyz.detach(),cam.fid[None].expand(len(g.get_xyz),1),fixed_attention=False)
                package=render(cam,g,pipe,background,out['d_xyz'],out['d_rotation'],out['d_scaling'])
                a=package['alpha'][0];depth=package['depth'][0]/a.clamp_min(.01)
                gt_alpha=F.interpolate(torch.from_numpy(rgba[...,3].copy()).float()[None,None].cuda()/255,size=(225,400),mode='bilinear',align_corners=False)[0,0]
                valid=(a>.95)&(gt_alpha>.95)&torch.isfinite(depth)&(depth>0)
                np.savez_compressed(target,depth=depth.cpu().numpy(),valid=valid.cpu().numpy(),w2c=cam.w2c.cpu().numpy(),K=cam.k.cpu().numpy())
            camera_data[fid]=dict(np.load(target,allow_pickle=False))
            if index%20==0:print('TRAIN DEPTH',index,'elapsed',round(time.monotonic()-start,1),flush=True)
    del g,d;torch.cuda.empty_cache()
    sys.path.append(lock['code_path'])
    from unimatch.unimatch import UniMatch
    flow_net=UniMatch(num_scales=2,upsample_factor=4).cuda().eval().requires_grad_(False)
    weights=torch.load(lock['weights_path'],map_location='cpu',weights_only=True);flow_net.load_state_dict(weights.get('model',weights),strict=True)
    pairs=[]
    with torch.no_grad():
        for a,b in zip(ids[:-1],ids[1:]):
            path=CACHE/'pairs'/f'{a:04d}_{b:04d}.npz'
            if not path.exists():
                prediction=flow_net(images[a].cuda(),images[b].cuda(),attn_type='swin',attn_splits_list=[2,8],corr_radius_list=[-1,4],prop_radius_list=[-1,1],pred_bidir_flow=True)['flow_preds'][-1]
                prediction=F.interpolate(prediction,size=(225,400),mode='bilinear',align_corners=False)
                prediction[:,0]*=400/800;prediction[:,1]*=225/448
                forward,backward=prediction.permute(0,2,3,1)
                ca={k:torch.as_tensor(v,device='cuda') for k,v in camera_data[a].items()};cb={k:torch.as_tensor(v,device='cuda') for k,v in camera_data[b].items()}
                motion,inside=motion_target(forward,cb['depth'],ca['w2c'],ca['K'],cb['w2c'],cb['K'])
                validity=inside&forward_backward_valid(forward,backward)&ca['valid']&(sample(cb['valid'].float(),pixel_grid(225,400,forward)+forward)>.99)
                motion=torch.where(validity[...,None],motion,torch.zeros_like(motion))
                if not torch.isfinite(motion).all():raise ValueError('Nonfinite motion target')
                if validity.sum()<100:raise ValueError(f'Insufficient confidence for train pair {a}->{b}')
                np.savez_compressed(path,forward=forward.cpu().numpy(),backward=backward.cpu().numpy(),motion=motion.cpu().numpy(),valid=validity.cpu().numpy())
            values=np.load(path,allow_pickle=False)
            pairs.append({'source':a,'target':b,'dt':(b-a)/199,'file':str(path.relative_to(CACHE)),'valid_pixels':int(values['valid'].sum())})
            if len(pairs)%15==0:print('TRAIN FLOW',len(pairs),'of119 elapsed',round(time.monotonic()-start,1),flush=True)
    manifest={'train_frame_ids':ids,'pairs':pairs,'image_inputs':'120 training RGB/RGBA only','heldout_RGB_used':False,
      'depth_source':'frozen train-only H3 final50k, alpha-normalized renderer depth, not GT',
      'motion_target':'backward camera compensation sampled at UniMatch forward correspondences; FB/opacity/in-bounds validity',
      'flow_inference_size':[800,448],'supervision_size':[400,225],'model_lock':lock,'input_sha256':hashes,
      'output_sha256':{str(p.relative_to(CACHE)):sha(p) for folder in ['depth','pairs'] for p in (CACHE/folder).glob('*.npz')},
      'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),Path(__file__).with_name('geometry.py')]}}
    (CACHE/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8');(REPORT/'flow_data_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    print('TRAIN-ONLY FLOW PREPARATION COMPLETE',len(pairs),flush=True)
    return CACHE


if __name__=='__main__':prepare()
