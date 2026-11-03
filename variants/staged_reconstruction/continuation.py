"""Opt-in S4--S6 continuation; baseline and earlier frozen trainers are untouched."""
import argparse
import json
import os
import random
import time
from pathlib import Path
import numpy as np
import torch
from .prepare_data import ROOT,REPORT,RUNS,DATA,read,write,sha
from .train_static import setup,camera
from .stage3.train import reconstruction,rng_restore
from .stage4.bind import load_physics_state
from .stage4.constraints import fixed_row_track_loss
from .stage5.nonpenetration import sample_sdf,nonpenetration
from .state_recovery.runtime import graph_state,restore_graphs


def parent_gate(stage):
    previous=f'S{int(stage[1:])-1}';path=RUNS/previous/'formal/checkpoint.pt'
    receipt=REPORT/previous/'admission.json'
    if not receipt.exists():raise RuntimeError(f'{previous} reviewed admission is required')
    value=read(receipt)
    if not value.get('admitted') or not value.get('native_images_reviewed'):
        raise RuntimeError(f'{previous} failed admission')
    assert sha(path)==value['checkpoint_sha256']
    bindings=read(REPORT/'S4/bindings.json')
    if not bindings['bindings']:raise RuntimeError('No qualified fixed-row tracks; do not fabricate S4--S6 training')
    assert bindings['parent_sha256']==sha(RUNS/'S3/formal/checkpoint.pt')
    return path,receipt,bindings


def stage_protocol(stage,parent,receipt):
    return {'stage':stage,'enabled_by_default':False,'seed':20261100+int(stage[1:]),
        'parent_sha256':sha(parent),'parent_admission_sha256':sha(receipt),
        'updates':4000,'width':1600,'checkpoint_interval':500,'validation_interval':2000,
        'final_preselected':True,'physics_every':10,'physics_samples_per_object':16,
        'kinematic_weight':.01,'CMR_weight':.00001,'shared_gradient_cap':.1,'time_step':1/199,
        'tracks_every':5,'tracks_per_update':4,'track_weight':.1,
        'bindings_sha256':sha(REPORT/'S4/bindings.json'),
        'nonpenetration_enabled':stage in ['S5','S6'],'nonpenetration_weight':.01,
        'sdf_receipt_sha256':sha(REPORT/'S5/static_volume_receipt.json') if stage in ['S5','S6'] else None,
        'nonpenetration_scope':'Only independently image-qualified fixed rows at their observed times; conditional static interior, not contact truth',
        'static_geometry_frozen':True,'static_appearance_trainable':stage=='S6',
        'joint_learning_rate_multiplier':.25 if stage=='S6' else 1.,
        'time_units':'Sequence normalized; acquisition seconds unavailable',
        'contact_force':False,'friction':False,'restitution':False,'constitutive':False,
        'unknown_force':False,'reidentification_through_occlusion':False,
        'validation_dynamic_PSNR_min':20.,'validation_dynamic_alpha_min':.85,
        'validation_static_PSNR_drop_from_S1_max':1.}


def optimizer_for(scene,bootstrap,parent,stage):
    opt=scene.optimizer_from(bootstrap)
    if any(g.get('name')=='static_appearance' for g in parent['optimizer']['param_groups']):
        params=[scene.static._features_dc,scene.static._features_rest]
        for p in params:p.requires_grad_(True)
        opt.add_param_group({'params':params,'lr':.000125,'name':'static_appearance'})
    opt.load_state_dict(parent['optimizer'])
    if stage=='S6' and not any(g['name']=='static_appearance' for g in opt.param_groups):
        params=[scene.static._features_dc,scene.static._features_rest]
        for p in params:p.requires_grad_(True)
        opt.add_param_group({'params':params,'lr':.000125,'name':'static_appearance'})
        for g in opt.param_groups:g['lr']*=.25
    return opt


def save(directory,scene,opt,step,stack,protocol):
    temp=directory/'checkpoint.tmp';path=directory/'checkpoint.pt'
    torch.save({'model':scene.state_dict(),'optimizer':opt.state_dict(),'iteration':step,'camera_stack':stack,
        'rng_python':random.getstate(),'rng_numpy':np.random.get_state(),'rng_torch':torch.get_rng_state(),
        'rng_cuda':torch.cuda.get_rng_state_all(),'physics_graphs':graph_state(scene),'protocol_sha256':sha(REPORT/protocol['stage']/'protocol.json'),
        'parent_sha256':protocol['parent_sha256'],'stage':protocol['stage'],
        'physical_time_units':protocol['time_units']},temp)
    os.replace(temp,path);write(directory/'checkpoint.json',{'sha256':sha(path),'iteration':step,'full_state':True})


def freeze(stage):
    paths=[Path(__file__)]+list((Path(__file__).parent/'stage4').glob('*.py'))+list((Path(__file__).parent/'stage5').glob('*.py'))
    paths += list((Path(__file__).parent/'state_recovery').glob('*.py'))
    paths += [REPORT/stage/'protocol.json',REPORT/'S4/bindings.json',ROOT/'tests/test_staged_continuation.py',
        ROOT/'tests/test_staged_track_constraints.py',ROOT/'tests/test_staged_binding.py',ROOT/'tests/test_staged_nonpenetration.py']
    if stage in ['S5','S6']:paths+=[REPORT/'S5/static_volume_receipt.json',RUNS/'S5_conditional_static_sdf.npz']
    values={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in paths}
    receipt=REPORT/stage/'source_freeze.json'
    if receipt.exists():assert read(receipt)==values
    else:write(receipt,values)
    from .state_recovery.train import freeze as freeze3
    freeze3(read(REPORT/'S3/protocol.json'))


def evaluate(renderer,scene,stage,step,split='val'):
    from .stage2 import train as parent
    old=parent.REPORT;parent.REPORT=REPORT/stage
    try:value=parent.evaluate(renderer,scene,step,split,1600,False)
    finally:parent.REPORT=old
    value['stage']=stage;write(REPORT/stage/f'{stage}_{split}_{step:05d}.json',value)
    return value


def interior_loss(scene,batch,volume,margin,enabled=False):
    if not enabled:return None
    from variants.flow_comparison.temporal import position
    points=[]
    for track in batch:
        k=track['local_early_object'];row=track['local_row']
        for n in track['frames']:
            points.append(position(scene.deforms[k],scene.objects[k].get_xyz[row:row+1],(n-1)/199)+scene.centers[k])
    if not points:raise ValueError('No independently qualified points for conditional nonpenetration')
    xyz=torch.cat(points);distance=sample_sdf(xyz,*volume)
    return nonpenetration(distance,torch.ones(len(xyz),dtype=torch.bool,device=xyz.device),margin,True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['S4','S5','S6'],required=True)
    p.add_argument('--enable-continuation',action='store_true');p.add_argument('--smoke',action='store_true')
    p.add_argument('--stop-after',type=int);p.add_argument('--evaluate-only',action='store_true');args=p.parse_args()
    if not args.enable_continuation:raise SystemExit('Continuation is opt-in; pass --enable-continuation')
    stage=args.stage;parent_path,receipt,bindings=parent_gate(stage)
    report=REPORT/stage;report.mkdir(parents=True,exist_ok=True)
    protocol=stage_protocol(stage,parent_path,receipt);file=report/'protocol.json'
    if file.exists():assert read(file)==protocol
    else:write(file,protocol)
    freeze(stage)
    directory=RUNS/stage/('_smoke' if args.smoke else 'formal');directory.mkdir(parents=True,exist_ok=True)
    import msvcrt
    lock=open(directory/'training.lock','a+b');lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0)
    try:msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    except OSError:raise SystemExit(f'{stage} already running')
    try:
        torch.set_num_threads(4);random.seed(protocol['seed']);np.random.seed(protocol['seed']);torch.manual_seed(protocol['seed'])
        renderer=setup();scene,parent,bootstrap=load_physics_state(parent_path)
        assert bindings['topology']==[len(o.get_xyz) for o in scene.objects]
        opt=optimizer_for(scene,bootstrap,parent,stage);saved=directory/'checkpoint.pt'
        if saved.exists():
            assert sha(saved)==read(directory/'checkpoint.json')['sha256']
            state=torch.load(saved,map_location='cpu',weights_only=False)
            assert state['protocol_sha256']==sha(file) and state['parent_sha256']==sha(parent_path)
            scene.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);restore_graphs(scene,state['physics_graphs']);rng_restore(state)
            start=state['iteration'];stack=state['camera_stack'];print(stage,'FULL_STATE_RESTORED',start,flush=True)
        else:
            restore_graphs(scene,parent['physics_graphs'])
            rng_restore(parent);start=0;stack=parent['camera_stack']
            save(directory,scene,opt,start,stack,protocol)
        if args.evaluate_only:
            evaluate(renderer,scene,stage,start,'val');evaluate(renderer,scene,stage,start,'test');return
        if not args.smoke:assert read(report/'smoke_verification.json')['passed']
        meta=read(DATA/'transforms_train.json');cache={};cams={};track_cams={}
        byid={int(Path(f['file_path']).stem):f for f in meta['frames']}
        for item in bindings['bindings']:
            for n in item['frames']:
                if n not in track_cams:track_cams[n]=camera(byid[n],meta,400)
        ledger=np.load(RUNS/'S2_observations.npz',allow_pickle=False)
        observations={int(n):(torch.tensor(c),o.tolist()) for n,c,o in zip(ledger['frame_ids'],ledger['components'],ledger['early_owner_to_component'])}
        volume=None;margin=0.
        if protocol['nonpenetration_enabled']:
            sdf_receipt=read(REPORT/'S5/static_volume_receipt.json');path=ROOT/sdf_receipt['file']
            assert sha(path)==sdf_receipt['sha256']
            with np.load(path,allow_pickle=False) as v:
                volume=(torch.tensor(v['sdf'],device='cuda'),v['origin'],float(v['spacing']))
            margin=sdf_receipt['discretization_margin']
        frozen={k:v.clone() for k,v in scene.static.state_dict().items() if not (stage=='S6' and k.startswith('_features'))}
        from .stage2.train import image
        from variants.momentum_guard.balance import balance
        began=time.monotonic()
        for step in range(start+1,protocol['updates']+1):
            if not stack:stack=random.sample(range(len(meta['frames'])),len(meta['frames']))
            index=stack.pop();frame=meta['frames'][index]
            if index not in cache:
                rgba=image(frame,1600);cache[index]=torch.tensor(rgba[...,:3]*rgba[...,3:4]).permute(2,0,1);cams[index]=camera(frame,meta,1600)
            gt=cache[index].cuda();opt.zero_grad(set_to_none=True)
            rgb=reconstruction(scene.render(renderer,cams[index],frame['time']),gt,scene,cams[index],frame,observations)
            loss=rgb;details={}
            if step%protocol['tracks_every']==0:
                tracks=bindings['bindings'];begin=((step//protocol['tracks_every']-1)*protocol['tracks_per_update'])%len(tracks)
                batch=[tracks[(begin+j)%len(tracks)] for j in range(min(len(tracks),protocol['tracks_per_update']))]
                track=fixed_row_track_loss(scene,batch,track_cams,True);extra=protocol['track_weight']*track
                details['track_loss']=float(track.detach())
                if protocol['nonpenetration_enabled']:
                    penetration,info=interior_loss(scene,batch,volume,margin,True)
                    extra=extra+protocol['nonpenetration_weight']*penetration
                    details['conditional_interior']=info;details['nonpenetration_loss']=float(penetration.detach())
                groups={'grid':list(scene.grids.parameters()),'deformation':list(scene.networks.parameters()),'canonical':[o._xyz for o in scene.objects]}
                guarded,guard=balance(rgb,extra,groups,[],protocol['shared_gradient_cap'],1.)
                loss=loss+guarded;details['track_interior_guard']=guard
            if step%protocol['physics_every']==0:
                terms,rows=scene.physics.loss(scene,frame['time'],step,16,protocol['time_step'])
                physical=protocol['kinematic_weight']*terms['kinematic']+protocol['CMR_weight']*terms['CMR']
                groups={'grid':list(scene.grids.parameters()),'deformation':list(scene.networks.parameters()),'canonical':[o._xyz for o in scene.objects]}
                guarded,guard=balance(rgb,physical,groups,list(scene.physics.parameters()),protocol['shared_gradient_cap'],1.)
                loss=loss+guarded;details['physics']={'terms':{k:float(v.detach()) for k,v in terms.items()},'objects':rows,'guard':guard}
            assert bool(loss.isfinite());loss.backward()
            assert all(p.grad is None or p.grad.isfinite().all() for p in scene.parameters())
            opt.step();scene.constrain()
            if details:
                with (directory/'objectives.jsonl').open('a') as s:s.write(json.dumps({'step':step,**details})+'\n')
            if step%200==0:
                assert all(torch.equal(v,scene.static.state_dict()[k]) for k,v in frozen.items())
                print(stage,'TRAIN',step,float(loss.detach()),'elapsed',round(time.monotonic()-began,1),flush=True)
                if not args.smoke:write(RUNS/'state.json',{'stage':stage,'status':'training','iteration':step,'pid':os.getpid()})
            if step%500==0:save(directory,scene,opt,step,stack,protocol)
            if not args.smoke and step%2000==0:evaluate(renderer,scene,stage,step)
            if (RUNS/'stop.request').exists() or args.stop_after and step>=args.stop_after:
                save(directory,scene,opt,step,stack,protocol)
                if not args.smoke:write(RUNS/'state.json',{'stage':stage,'status':'saved_stopped','iteration':step})
                return
        save(directory,scene,opt,protocol['updates'],stack,protocol)
        evaluate(renderer,scene,stage,protocol['updates'],'val');evaluate(renderer,scene,stage,protocol['updates'],'test');freeze(stage)
        if not args.smoke:write(RUNS/'state.json',{'stage':stage,'status':'evaluated_pending_review','iteration':protocol['updates']})
    except Exception as error:
        write(directory/f'failure_{time.time_ns()}.json',{'error':repr(error),'attempted_step':locals().get('step'),'checkpoint_preserved':True})
        if not args.smoke:write(RUNS/'state.json',{'stage':stage,'status':'failed','error':str(error)})
        raise
    finally:lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()


if __name__=='__main__':main()
