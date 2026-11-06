"""Read-only L5 native evidence; training reprojection is never called GT."""
import argparse
import random
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch
from variants.staged_reconstruction.i10_surface_support import runtime as a
from variants.staged_reconstruction.i10_track_train.core import observations
from variants.staged_reconstruction.i10_tracking.io import ROOT,DATA,read,write,sha
from variants.staged_reconstruction.train_static import camera,render_static,PIPE
from experiments.staged_reconstruction.RubberDuck_joint_pilot import evaluate as helper

r=a.r
LABELS=['J6_parent','temporal_control','surface_support']


def load_models():
    p=read(a.BASE/'protocol.json');source=torch.load(ROOT/p['parent'],map_location='cpu',weights_only=False)
    renderer,parent,opts,_,_=r.load('temporal_control');models={'J6_parent':parent};audits={};initials=[];finals=[]
    for arm in p['arms']:
        renderer,scene,opts,_,_=r.load(arm)
        path=a.RUNS/'formal'/arm/'checkpoint.pt';state=r.restore(path,scene,opts)
        initial=torch.load(path.with_name('initial.pt'),map_location='cpu',weights_only=False)
        for key in ['model','optimizers','bindings','physics_graphs','rng_python','rng_numpy','rng_torch','rng_cuda','camera_stack']:
            r.assert_equivalent(initial[key],source[key],0,0)
        assert state['iteration']==p['updates']==2000 and not state['color_static_reference']
        for name,value in initial['model'].items():
            if name.startswith(('static.','physics.')) or name=='centers':assert torch.equal(value,state['model'][name]),(arm,name)
        r.assert_equivalent(initial['physics_graphs'],state['physics_graphs'],0,0)
        r.assert_equivalent(initial['optimizers'][0],state['optimizers'][0],0,0)
        assert initial['bindings']==state['bindings']
        for before,after in zip(initial['optimizers'][1]['param_groups'],state['optimizers'][1]['param_groups']):
            assert before==after
            for key in before['params']:
                x=initial['optimizers'][1]['state'].get(key);y=state['optimizers'][1]['state'].get(key)
                if before['name']=='PIDG_velocity_stress':r.assert_equivalent(x,y,0,0)
                else:assert int(y['step'])==int(x['step'])+2000
        rng=random.Random();rng.setstate(initial['rng_python']);stack=list(initial['camera_stack']);trace=[]
        for _ in range(2000):
            if not stack:stack=rng.sample(range(120),120)
            trace.append(stack.pop())
        assert state['sample_trace']==trace and state['camera_stack']==stack and state['rng_python']==rng.getstate()
        audits[arm]=dict(checkpoint_sha256=sha(path),complete_parent_states_exact=True,
            static_model_Adam_physics_model_Adam_and_graphs_frozen=True,all2000_sampler_replay=True)
        initials.append(initial);finals.append(state);models[arm]=scene
    r.assert_equivalent(initials[0],initials[1],0,0)
    assert finals[0]['sample_trace']==finals[1]['sample_trace']
    return renderer,models,audits


@torch.no_grad()
def validation(renderer,models):
    from utils.loss_utils import ssim
    meta=read(ROOT/'data/derived/kinematic_resplit/transforms_val.json');assert len(meta['frames'])==30
    rows=[]
    for index,frame in enumerate(meta['frames']):
        assert not r.stopped();cam=camera(frame,meta,1600);target=r.frame_target(frame)
        row=dict(frame=frame['file_path'])
        for name,scene in models.items():
            pkg=scene.render(renderer,cam,frame['time']);pred=pkg['render'].clamp(0,1)
            row[name]=dict(PSNR=float(-10*torch.log10((pred-target).square().mean().clamp_min(1e-12))),
                SSIM=float(ssim(pred,target)),L1=float((pred-target).abs().mean()))
        rows.append(row);print('L5_VALIDATION',index+1,30,flush=True)
    means={n:{k:float(np.mean([v[n][k] for v in rows])) for k in ['PSNR','SSIM','L1']} for n in LABELS}
    return dict(frames=rows,means=means,all30_native1600=True,semantic_masks_used=False,test_not_read=True,
        PSNR_nonregression_gate=means['surface_support']['PSNR']>=means['temporal_control']['PSNR']-.3,
        parent_nonregression_gate=means['surface_support']['PSNR']>=means['J6_parent']['PSNR']-.3,
        independent_geometry_GT=False)


@torch.no_grad()
def diagnostics(renderer,models):
    meta=read(DATA/'transforms_train.json');byid={int(Path(f['file_path']).stem):f for f in meta['frames']}
    images={};receipts={};proxies=[]
    for cid in [86,188]:
        cam=camera(byid[cid],meta,1600);receipts[str(cid)]=dict(frame=byid[cid]['file_path'],transform=byid[cid]['transform_matrix'],size=[1600,900])
        static={n:render_static(renderer,cam,s.static) for n,s in models.items()}
        p=static['J6_parent'];valid=p['alpha'].squeeze()>.5;D=p['depth'].squeeze()/p['alpha'].squeeze().clamp_min(1e-8)
        lo,hi=[float(torch.quantile(D[valid],q)) for q in [0.,1.]]
        for n,pkg in static.items():
            assert all(torch.equal(pkg[k],static['J6_parent'][k]) for k in ['render','alpha','depth'])
        for number in [2,100,200]:
            t=(number-1)/199;rows=[];comparison=[]
            for name,scene in models.items():
                s=static[name];d=renderer(cam,scene.dynamic_at(t),PIPE,scene.centers.new_zeros(3),0.,0.,0.)
                joint=scene.render(renderer,cam,t)
                arrays=[('staticRGB',helper.rgb(s)),('dynamicRGB',helper.rgb(d)),('jointRGB',helper.rgb(joint)),
                    ('staticAlpha',helper.alpha(s)),('dynamicAlpha',helper.alpha(d)),('jointAlpha',helper.alpha(joint)),
                    ('staticD-alpha',helper.depth(s,lo,hi)),('dynamicD-alpha',helper.depth(d,lo,hi)),('jointD-alpha',helper.depth(joint,lo,hi))]
                rows.append((name,arrays));comparison.append((name,helper.rgb(joint)))
                for label,array in arrays:
                    path=a.OUT/f'c{cid:03d}_t{number:03d}_{name}_{label}_native.png';Image.fromarray(array).save(path)
                    images[path.relative_to(ROOT).as_posix()]=sha(path)
                alpha=d['alpha'].squeeze();pred=d['render'];dark=(pred.max(0).values<.05)&(alpha>.5)
                proxies.append(dict(camera=cid,time=number,model=name,pixels=alpha.numel(),dynamic_alpha_gt05=int((alpha>.5).sum()),
                    dynamic_dark_high_alpha=int(dark.sum()),dark_definition='all rendered dynamic RGB channels <.05 and rendered alpha>.5; evaluation only, not training class',not_geometry_GT=True))
            path=a.OUT/f'c{cid:03d}_t{number:03d}_decomposition.png'
            helper.sheet(rows,f'L5 c{cid} t{number}: parent / temporal control / surface support; no independent fixed-view GT').save(path);images[path.relative_to(ROOT).as_posix()]=sha(path)
            path=a.OUT/f'c{cid:03d}_t{number:03d}_joint_comparison.png'
            helper.sheet([('joint RGB',comparison)],f'L5 c{cid} t{number}: J6 parent / temporal control / surface support',cell=(640,360)).save(path);images[path.relative_to(ROOT).as_posix()]=sha(path)
    for index in [0,24,48,72,96,119]:
        frame=meta['frames'][index];cam=camera(frame,meta,1600);target=r.frame_target(frame)
        rows=[('observed',[('train trajectory RGB',(target.permute(1,2,0).cpu().numpy()*255).round().astype('uint8'))])]
        rows += [(n,[('joint RGB',helper.rgb(s.render(renderer,cam,frame['time'])))]) for n,s in models.items()]
        path=a.OUT/f'observed_train_index{index:03d}.png';helper.sheet(rows,'Observed trajectory; no geometry/identity GT',cell=(800,450)).save(path);images[path.relative_to(ROOT).as_posix()]=sha(path)
    return dict(images=images,fixed_cameras=receipts,static_RGB_alpha_D_exact=True,high_alpha_proxies=proxies,
        black_missing_is_not_zero_error=True,D_div_alpha_is_rendered_proxy=True)


@torch.no_grad()
def track_diagnostic(models):
    from variants.staged_reconstruction.i10_tracking.core import project_world
    result={}
    for name,scene in models.items():
        total=missing=0;fit=[];held=[]
        for memory in a.MEMORIES:
            frames=memory['frames'];row=memory['binding']['row']
            xyz=np.array([scene.dynamic_at(a.META['frames'][f]['time']).get_xyz[row].cpu().numpy() for f in frames])
            uv,z=project_world(xyz[:,None],a.ASSET['views'][frames],a.ASSET['k'])
            error=np.linalg.norm(uv[:,0]-memory['uv'],axis=-1);good=np.isfinite(error)&(z[:,0]>.01)
            total+=len(frames);missing+=int((~good).sum())
            fit += [float(e) for f,e,g in zip(frames,error,good) if not f%2 and g]
            held += [float(e) for f,e,g in zip(frames,error,good) if f%2 and g]
        result[name]=dict(total=total,missing=missing,fit_mean_px=float(np.mean(fit)),held_mean_px=float(np.mean(held)))
    return dict(all_original_bound_memories_unfiltered=True,held_RGB_trained=True,independent_3D_GT=False,models=result)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--enable-surface-evaluation',action='store_true')
    parser.add_argument('--phase',choices=['evaluate','video'],default='evaluate');args=parser.parse_args()
    if not args.enable_surface_evaluation:raise SystemExit('Evaluation opt-in')
    a.install(True);r.verify();assert not r.stopped();torch.set_num_threads(4);cv2.setNumThreads(2);a.OUT.mkdir(parents=True,exist_ok=True)
    with r.lock('evaluation.lock'):
        renderer,models,audits=load_models()
        if args.phase=='evaluate':
            assert not (a.REPORT/'evaluation.json').exists()
            result=dict(checkpoint_audits=audits,validation=validation(renderer,models),diagnostics=diagnostics(renderer,models),
                training_track_diagnostic=track_diagnostic(models),plys=helper.plys(models),source_sha256=sha(Path(__file__)))
            r.verify();write(a.REPORT/'evaluation.json',result);print('L5_EVALUATION_COMPLETE',result['validation']['means'],flush=True)
        else:
            assert not (a.REPORT/'videos.json').exists() and (a.REPORT/'evaluation.json').exists()
            result=helper.videos(renderer,models);r.verify();write(a.REPORT/'videos.json',dict(videos=result,source_sha256=sha(Path(__file__))))

if __name__=='__main__':main()
