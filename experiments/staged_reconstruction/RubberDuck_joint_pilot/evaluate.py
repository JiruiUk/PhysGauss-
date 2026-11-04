"""Read-only native RGB validation and original-trajectory/fixed-view evidence."""
import argparse
from pathlib import Path
import random
import subprocess
import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image,ImageDraw
import torch
from variants.staged_reconstruction.rubber_joint import runtime as r
from variants.staged_reconstruction.prepare_data import ROOT,DATA,read,sha
from variants.staged_reconstruction.train_static import camera,render_static,PIPE

LABELS=['parent6k','frozen_static','joint_depth_trust']


def load_models():
    models={};audits={};initials=[];finals=[]
    renderer,parent,_,_,parent_state=r.load('frozen_static');models['parent6k']=parent
    for arm in LABELS[1:]:
        renderer,scene,opts,_,_=r.load(arm)
        path=r.RUNS/'formal'/arm/'checkpoint.pt';state=r.restore(path,scene,opts)
        assert state['iteration']==2000 and len(state['sample_trace'])==2000
        initial=torch.load(path.with_name('initial.pt'),map_location='cpu',weights_only=False)
        for key in ['model','camera_stack','rng_python','rng_numpy','rng_torch','rng_cuda','physics_graphs']:
            r.assert_equivalent(initial[key],parent_state[key],0,0)
        r.assert_equivalent(initial['optimizers'][1],parent_state['optimizer'],0,0)
        sampler=random.Random();sampler.setstate(initial['rng_python']);stack=list(initial['camera_stack']);trace=[]
        for _ in range(2000):
            if not stack:stack=sampler.sample(range(120),120)
            trace.append(stack.pop())
        assert state['sample_trace']==trace and state['camera_stack']==stack
        assert state['rng_python']==sampler.getstate()
        for name,value in parent_state['model'].items():
            if name.startswith('physics.') or (name.startswith('static.') and
                (arm=='frozen_static' or name not in ['static._opacity','static._features_dc','static._features_rest'])):
                assert torch.equal(value,state['model'][name]),(arm,name)
        r.assert_equivalent(state['physics_graphs'],parent_state['physics_graphs'],0,0)
        physics_group=state['optimizers'][1]['param_groups'][-1]
        for key in physics_group['params']:
            r.assert_equivalent(state['optimizers'][1]['state'].get(key),parent_state['optimizer']['state'].get(key),0,0)
        static_steps={g['name']:int(state['optimizers'][0]['state'][g['params'][0]]['step'])
            for g in state['optimizers'][0]['param_groups']}
        for name,step in static_steps.items():
            assert step==(14000 if arm=='joint_depth_trust' and name in ['_opacity','_features_dc','_features_rest'] else 12000)
        audits[arm]={'checkpoint_sha256':sha(path),'updates':2000,'static_Adam_steps':static_steps,
            'sampler_replay_exact':True,'frozen_static_geometry_physics_and_physics_Adam_graphs_exact':True}
        initials.append(initial);finals.append(state);models[arm]=scene
    r.assert_equivalent(initials[0],initials[1],0,0)
    assert finals[0]['sample_trace']==finals[1]['sample_trace']
    return renderer,models,audits


def rgb(package):return (package['render'].detach().clamp(0,1).permute(1,2,0).cpu().numpy()*255).round().astype('uint8')


def alpha(package):
    a=package['alpha'].detach().squeeze().clamp(0,1).cpu().numpy()
    return np.repeat((a*255).round().astype('uint8')[...,None],3,2)


def depth(package,lo,hi):
    a=package['alpha'].detach().squeeze().cpu().numpy()
    D=package['depth'].detach().squeeze().cpu().numpy()/np.maximum(a,1e-8)
    mapped=np.clip((D-lo)/max(hi-lo,1e-8),0,1)
    color=cv2.cvtColor(cv2.applyColorMap((mapped*255).astype('uint8'),cv2.COLORMAP_TURBO),cv2.COLOR_BGR2RGB)
    color[(a<=.5)|~np.isfinite(D)|(D<=0)]=0
    return color


def sheet(rows,heading,cell=(400,225)):
    cols=len(rows[0][1]);w,h=cell
    canvas=Image.new('RGB',(cols*w,len(rows)*(h+28)+34),'#171923');draw=ImageDraw.Draw(canvas)
    draw.text((6,8),heading,fill='white')
    for row,(label,values) in enumerate(rows):
        for col,(name,array) in enumerate(values):
            x,y=col*w,34+row*(h+28)
            canvas.paste(Image.fromarray(array).resize((w,h),Image.Resampling.LANCZOS),(x,y+28))
            draw.text((x+4,y+4),label+' '+name,fill='white')
    return canvas


@torch.no_grad()
def validation(renderer,models):
    from utils.loss_utils import ssim
    meta=read(ROOT/'data/derived/kinematic_resplit/transforms_val.json');assert len(meta['frames'])==30
    rows=[]
    for index,frame in enumerate(meta['frames']):
        assert not r.stopped()
        target=r.frame_target(frame);cam=camera(frame,meta,1600);row={'frame':frame['file_path']}
        for name,scene in models.items():
            package=scene.render(renderer,cam,frame['time']);pred=package['render'].clamp(0,1)
            mse=(pred-target).square().mean();absolute=(pred-target).abs().mean();structural=ssim(pred,target)
            row[name]={'PSNR':float(-10*torch.log10(mse.clamp_min(1e-12))),
                'SSIM':float(structural),'L1':float(absolute),'loss':float(.8*absolute+.2*(1-structural)),
                'full_image_alpha_mean':float(package['alpha'].mean())}
        rows.append(row);print('J4_VALIDATION',index+1,30,flush=True)
    means={n:{k:float(np.mean([row[n][k] for row in rows])) for k in rows[0][n]} for n in LABELS}
    passed=means['joint_depth_trust']['PSNR']>=means['frozen_static']['PSNR']-.3
    return {'frames':rows,'means':means,'native_width':1600,'all30_full_RGB':True,
        'semantic_masks_used':False,'independent_geometry_GT':False,'PSNR_nonregression_gate':passed,
        'manual_geometry_cleanup_required':True,'test_RGB_not_read':True}


@torch.no_grad()
def diagnostics(renderer,models):
    meta=read(DATA/'transforms_train.json');byid={int(Path(f['file_path']).stem):f for f in meta['frames']}
    images={};camera_receipts={};support=[]
    for camid in [86,188]:
        frame=byid[camid];cam=camera(frame,meta,1600)
        camera_receipts[str(camid)]={'frame':frame['file_path'],'camera_transform':frame['transform_matrix'],
            'FoVx':cam.FoVx,'FoVy':cam.FoVy,'size':[1600,900]}
        statics={n:render_static(renderer,cam,s.static) for n,s in models.items()}
        p=statics['parent6k'];a=p['alpha'].squeeze();D=p['depth'].squeeze()/a.clamp_min(1e-8)
        valid=(a>.5)&torch.isfinite(D)&(D>0);lo,hi=[float(torch.quantile(D[valid],q)) for q in [0.,1.]]
        for number in [2,100,200]:
            t=(number-1)/199;rows=[];values=[]
            for name,scene in models.items():
                static=statics[name];dynamic=renderer(cam,scene.dynamic_at(t),PIPE,scene.centers.new_zeros(3),0.,0.,0.)
                joint=scene.render(renderer,cam,t)
                arrays=[('staticRGB',rgb(static)),('dynamicRGB',rgb(dynamic)),('jointRGB',rgb(joint)),
                    ('staticAlpha',alpha(static)),('jointAlpha',alpha(joint)),
                    ('staticD-alpha',depth(static,lo,hi)),('jointD-alpha',depth(joint,lo,hi))]
                rows.append((name,arrays));values.append((name,[('jointRGB',rgb(joint))]))
                if number in [2,200]:
                    for label,array in arrays:
                        path=r.OUT/f'c{camid:03d}_t{number:03d}_{name}_{label}_native.png';Image.fromarray(array).save(path)
                        images[path.relative_to(ROOT).as_posix()]=sha(path)
                if number==2:
                    mask=(static['alpha'].squeeze()>.5);expected=static['depth'].squeeze()/static['alpha'].squeeze().clamp_min(1e-8)
                    support.append({'camera':camid,'model':name,'pixels':mask.numel(),'alpha_gt05':int(mask.sum()),
                        'parent_support':int(valid.sum()),'parent_support_now_missing':int((valid&~mask).sum()),
                        'D_alpha_parent_support_MAE_proxy':float((expected[valid]-D[valid]).abs().mean()),
                        'not_independent_geometry_accuracy':True})
            path=r.OUT/f'c{camid:03d}_t{number:03d}_decomposition.png'
            sheet(rows,f'Rubber Duck c{camid} t{number}; fixed view: NO independent GT; black depth = missing; D range {lo:.3f}..{hi:.3f}').save(path)
            images[path.relative_to(ROOT).as_posix()]=sha(path)
            path=r.OUT/f'c{camid:03d}_t{number:03d}_joint_comparison.png'
            sheet([('',[(name,arr[0][1]) for name,arr in values])],f'Rubber Duck camera{camid} time{number}: parent / frozen control / joint',cell=(640,360)).save(path)
            images[path.relative_to(ROOT).as_posix()]=sha(path)
    for index in [0,24,48,72,96,119]:
        frame=meta['frames'][index];cam=camera(frame,meta,1600);target=r.frame_target(frame)
        observed=(target.permute(1,2,0).cpu().numpy()*255).round().astype('uint8')
        rows=[('observed', [('training trajectory RGB',observed)])]
        for name,scene in models.items():rows.append((name,[('joint RGB',rgb(scene.render(renderer,cam,frame['time'])))]))
        path=r.OUT/f'observed_train_index{index:03d}.png';sheet(rows,frame['file_path']+' observed trajectory; no identity/geometry GT',cell=(800,450)).save(path)
        images[path.relative_to(ROOT).as_posix()]=sha(path)
    return {'images':images,'fixed_cameras':camera_receipts,'static_support_proxies':support,
        'missing_black_depth_is_not_zero_error':True}


@torch.no_grad()
def plys(models):
    from scene_PIDG.gaussian_model import GaussianModel
    from plyfile import PlyData
    result={}
    for name,scene in models.items():
        for label,bank in [('static',scene.static),('canonical_dynamic_local',scene.pools[0])]:
            native=GaussianModel(3);native.active_sh_degree=3
            for key,p in bank.named_parameters():setattr(native,key,p)
            native.dynamic_mask=torch.full((len(bank.get_xyz),),label!='static',dtype=torch.bool,device='cuda')
            path=r.OUT/f'{name}_{label}.ply';assert not path.exists();native.save_ply(str(path))
            vertex=PlyData.read(path)['vertex'];xyz=np.stack([vertex[k] for k in ['x','y','z']],1)
            assert np.array_equal(xyz,bank.get_xyz.cpu().numpy())
            assert all(np.isfinite(vertex[k]).all() for k in vertex.data.dtype.names)
            result[name+'/'+label]={'path':path.relative_to(ROOT).as_posix(),'sha256':sha(path),'vertices':len(xyz),
                'checkpoint_position_exact':True,'all_fields_finite':True,'coordinates':'world' if label=='static' else 'local canonical; add checkpoint centers after deformation'}
        for number in [2,100,200]:
            at=scene.dynamic_at((number-1)/199);xyz=at.get_xyz.cpu().numpy();path=r.OUT/f'{name}_dynamic_world_t{number:03d}.ply'
            vertices=np.empty(len(xyz),dtype=[('x','f4'),('y','f4'),('z','f4')])
            for i,key in enumerate(['x','y','z']):vertices[key]=xyz[:,i]
            from plyfile import PlyElement
            PlyData([PlyElement.describe(vertices,'vertex')]).write(path)
            actual=PlyData.read(path)['vertex'];assert all(np.array_equal(actual[k],xyz[:,i]) for i,k in enumerate(['x','y','z']))
            result[name+'/dynamic_world/'+str(number)]={'path':path.relative_to(ROOT).as_posix(),'sha256':sha(path),
                'vertices':len(xyz),'checkpoint_deformed_position_exact':True,'coordinates':'world; XYZ-only diagnostic, not full Gaussian PLY'}
    return result


def encode(path,frames,count=200):
    assert not path.exists();iterator=iter(frames);first=next(iterator);height,width=np.asarray(first).shape[:2]
    with path.with_suffix('.encode.log').open('wb') as log:
        process=subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-nostdin','-n','-f','rawvideo',
            '-pix_fmt','rgb24','-s',f'{width}x{height}','-r','20','-i','pipe:0','-an','-c:v','libx264',
            '-threads','4','-profile:v','main','-pix_fmt','yuv420p','-crf','18','-movflags','+faststart',str(path)],
            stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=log,creationflags=subprocess.CREATE_NO_WINDOW)
        actual=0
        try:
            process.stdin.write(np.asarray(first).tobytes());actual+=1
            for frame in iterator:process.stdin.write(np.asarray(frame).tobytes());actual+=1
        finally:process.stdin.close();code=process.wait(timeout=60)
        assert code==0 and actual==count
    cap=cv2.VideoCapture(str(path));decoded=0;contacts=[]
    while True:
        ok,value=cap.read()
        if not ok:break
        if decoded in [1,99,199]:contacts.append(cv2.cvtColor(value,cv2.COLOR_BGR2RGB))
        decoded+=1
    cap.release();assert decoded==count
    contact=path.with_suffix('.decoded.png');Image.fromarray(np.concatenate(contacts)).save(contact)
    return {'path':path.relative_to(ROOT).as_posix(),'sha256':sha(path),'decoded_frames':decoded,
        'codec':'H264 Main yuv420p faststart','browser_playback_not_claimed':True,
        'decoded_contact':contact.relative_to(ROOT).as_posix(),'decoded_contact_sha256':sha(contact)}


@torch.no_grad()
def videos(renderer,models):
    meta=read(DATA/'transforms_train.json');byid={int(Path(f['file_path']).stem):f for f in meta['frames']};result={}
    for camid in [86,188]:
        cam=camera(byid[camid],meta,1600)
        static={n:rgb(render_static(renderer,cam,s.static)) for n,s in models.items()}
        def frames():
            for number in range(1,201):
                assert not r.stopped();t=(number-1)/199;rows=[];joint_images=[];dynamic_images=[];alpha_images=[]
                for name,scene in models.items():
                    joint=scene.render(renderer,cam,t);dynamic=renderer(cam,scene.dynamic_at(t),PIPE,scene.centers.new_zeros(3),0.,0.,0.)
                    joint_images.append((name,rgb(joint)));dynamic_images.append((name,rgb(dynamic)));alpha_images.append((name,alpha(joint)))
                rows=[('jointRGB',joint_images),('dynamicRGB',dynamic_images),('jointAlpha',alpha_images),('staticRGB',list(static.items()))]
                if number%25==0:print('J4_VIDEO',camid,number,200,flush=True)
                yield np.asarray(sheet(rows,f'Rubber Duck fixed{camid} time{number}; no independent cross-time GT',cell=(480,270)))
        result[str(camid)]=encode(r.OUT/f'fixed_c{camid:03d}_parent_control_joint.mp4',frames())
    return result


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--enable-rubber-joint-evaluation',action='store_true')
    parser.add_argument('--phase',choices=['evaluate','video'],default='evaluate');args=parser.parse_args()
    if not args.enable_rubber_joint_evaluation:raise SystemExit('Opt-in independent J4 evaluation')
    assert not r.stopped();r.verify();torch.set_num_threads(4);cv2.setNumThreads(2);r.OUT.mkdir(parents=True,exist_ok=True)
    with r.lock('evaluation.lock'):
        renderer,models,audits=load_models()
        if args.phase=='evaluate':
            assert not (r.REPORT/'evaluation.json').exists()
            result={'checkpoint_audits':audits,'validation':validation(renderer,models),
                'diagnostics':diagnostics(renderer,models),'plys':plys(models),'source_sha256':sha(Path(__file__))}
            r.verify();r.atomic_json(r.REPORT/'evaluation.json',result)
            print('J4_VALIDATION_AND_NATIVE_IMAGES_DONE',result['validation']['means'],flush=True)
        else:
            assert (r.REPORT/'evaluation.json').exists() and not (r.REPORT/'videos.json').exists()
            result=videos(renderer,models);r.verify();r.atomic_json(r.REPORT/'videos.json',{'videos':result,'source_sha256':sha(Path(__file__))})
            print('J4_TWO_FULL200_VIDEOS_DECODED',flush=True)

if __name__=='__main__':main()
