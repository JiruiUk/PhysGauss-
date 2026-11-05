"""Read-only native J6 evidence, with exact two-source and two-Adam audits."""
import argparse
import random
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image
from variants.staged_reconstruction.i10_joint import runtime as adapter
from variants.staged_reconstruction.prepare_data import ROOT,DATA,read,sha
from variants.staged_reconstruction.train_static import camera,render_static,PIPE
from experiments.staged_reconstruction.RubberDuck_joint_pilot import evaluate as original

r=adapter.r
LABELS=['I10_J4_initial','frozen_static','joint_depth_trust']
rgb=original.rgb;alpha=original.alpha;depth=original.depth;sheet=original.sheet


def load_models():
    models={};audits={};initials=[];finals=[]
    renderer,parent,opts,_,parent_state=r.load('frozen_static')
    expected=r.cpu_values(r.snapshot(parent,opts,0,parent_state['camera_stack'],[]))
    models[LABELS[0]]=parent
    p=read(r.BASE/'protocol.json')
    i10=torch.load(ROOT/p['parent'],map_location='cpu',weights_only=False)
    j4=torch.load(ROOT/p['static_parent'],map_location='cpu',weights_only=False)
    r.assert_equivalent(expected['optimizers'][1],i10['optimizer'],0,0)
    r.assert_equivalent(expected['optimizers'][0],j4['optimizers'][0],0,0)
    for name,value in expected['model'].items():
        source=j4 if name.startswith('static.') else i10
        assert torch.equal(value,source['model'][name]),name
    for arm in LABELS[1:]:
        renderer,scene,opts,_,_=r.load(arm)
        path=r.RUNS/'formal'/arm/'checkpoint.pt';state=r.restore(path,scene,opts)
        assert state['iteration']==2000 and len(state['sample_trace'])==2000
        initial=torch.load(path.with_name('initial.pt'),map_location='cpu',weights_only=False)
        r.assert_equivalent(initial,expected,0,0)
        sampler=random.Random();sampler.setstate(initial['rng_python']);stack=list(initial['camera_stack']);trace=[]
        for _ in range(2000):
            if not stack:stack=sampler.sample(range(120),120)
            trace.append(stack.pop())
        assert state['sample_trace']==trace and state['camera_stack']==stack
        assert state['rng_python']==sampler.getstate()
        for name,value in expected['model'].items():
            if name.startswith('physics.') or (name.startswith('static.') and
                (arm=='frozen_static' or name not in ['static._opacity','static._features_dc','static._features_rest'])):
                assert torch.equal(value,state['model'][name]),(arm,name)
        r.assert_equivalent(state['physics_graphs'],expected['physics_graphs'],0,0)
        group=state['optimizers'][1]['param_groups'][-1]
        assert group['name']=='PIDG_velocity_stress'
        for key in group['params']:
            r.assert_equivalent(state['optimizers'][1]['state'].get(key),i10['optimizer']['state'].get(key),0,0)
        dynamic_steps={}
        for group in state['optimizers'][1]['param_groups']:
            values=[]
            for key in group['params']:
                if key not in state['optimizers'][1]['state']:continue
                before=int(i10['optimizer']['state'][key]['step'])
                after=int(state['optimizers'][1]['state'][key]['step'])
                assert after==before+(0 if group['name']=='PIDG_velocity_stress' else 2000)
                values.append(after)
            dynamic_steps[group['name']]=sorted(set(values))
        static_steps={}
        for group in state['optimizers'][0]['param_groups']:
            key=group['params'][0];name=group['name']
            before=int(j4['optimizers'][0]['state'][key]['step'])
            after=int(state['optimizers'][0]['state'][key]['step'])
            assert after==before+(2000 if arm=='joint_depth_trust' and name in ['_opacity','_features_dc','_features_rest'] else 0)
            static_steps[name]=after
        audits[arm]={'checkpoint_sha256':sha(path),'updates':2000,'static_Adam_steps':static_steps,
            'dynamic_Adam_steps':dynamic_steps,'sampler_replay_exact':True,
            'initial_complete_components_and_Adam_exact':True,'frozen_geometry_physics_Adam_graphs_exact':True}
        initials.append(initial);finals.append(state);models[arm]=scene
    r.assert_equivalent(initials[0],initials[1],0,0)
    assert finals[0]['sample_trace']==finals[1]['sample_trace']
    return renderer,models,audits


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--enable-i10-joint-evaluation',action='store_true')
    p.add_argument('--phase',choices=['evaluate','video'],default='evaluate');args=p.parse_args()
    if not args.enable_i10_joint_evaluation:raise SystemExit('J6 evaluation default off')
    adapter.install(True);original.LABELS=LABELS
    assert not r.stopped();r.verify();torch.set_num_threads(4);cv2.setNumThreads(2)
    r.OUT.mkdir(parents=True,exist_ok=True)
    with r.lock('evaluation.lock'):
        renderer,models,audits=load_models()
        if args.phase=='evaluate':
            assert not (r.REPORT/'evaluation.json').exists()
            result={'stage':'J6','component_parents':read(r.BASE/'protocol.json')['parent_state_policy'],
                'checkpoint_audits':audits,'validation':original.validation(renderer,models),
                'diagnostics':diagnostics(renderer,models),'plys':original.plys(models),
                'source_sha256':sha(Path(__file__))}
            r.verify();r.atomic_json(r.REPORT/'evaluation.json',result)
            print('J6_NATIVE_EVALUATION_DONE',result['validation']['means'],flush=True)
        else:
            assert (r.REPORT/'evaluation.json').exists() and not (r.REPORT/'videos.json').exists()
            result=original.videos(renderer,models);r.verify()
            r.atomic_json(r.REPORT/'videos.json',{'videos':result,'source_sha256':sha(Path(__file__))})
            print('J6_TWO_FULL200_VIDEOS_DECODED',flush=True)


@torch.no_grad()
def diagnostics(renderer,models):
    meta=read(DATA/'transforms_train.json');byid={int(Path(f['file_path']).stem):f for f in meta['frames']}
    images={};camera_receipts={};support=[]
    for camid in [86,188]:
        frame=byid[camid];cam=camera(frame,meta,1600)
        camera_receipts[str(camid)]={'frame':frame['file_path'],'camera_transform':frame['transform_matrix'],
            'FoVx':cam.FoVx,'FoVy':cam.FoVy,'size':[1600,900]}
        statics={n:render_static(renderer,cam,s.static) for n,s in models.items()}
        p=statics['I10_J4_initial'];a=p['alpha'].squeeze();D=p['depth'].squeeze()/a.clamp_min(1e-8)
        valid=(a>.5)&torch.isfinite(D)&(D>0);lo,hi=[float(torch.quantile(D[valid],q)) for q in [0.,1.]]
        for number in [2,100,200]:
            t=(number-1)/199;rows=[];values=[]
            for name,scene in models.items():
                static=statics[name];dynamic=renderer(cam,scene.dynamic_at(t),PIPE,scene.centers.new_zeros(3),0.,0.,0.)
                joint=scene.render(renderer,cam,t)
                arrays=[('staticRGB',rgb(static)),('dynamicRGB',rgb(dynamic)),('jointRGB',rgb(joint)),
                    ('staticAlpha',alpha(static)),('dynamicAlpha',alpha(dynamic)),('jointAlpha',alpha(joint)),
                    ('staticD-alpha',depth(static,lo,hi)),('dynamicD-alpha',depth(dynamic,lo,hi)),('jointD-alpha',depth(joint,lo,hi))]
                rows.append((name,arrays));values.append((name,[('jointRGB',rgb(joint))]))
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
            sheet([('',[(name,arr[0][1]) for name,arr in values])],f'Rubber Duck camera{camid} time{number}: I10+J4 initial / frozen control / joint',cell=(640,360)).save(path)
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





if __name__=='__main__':main()
