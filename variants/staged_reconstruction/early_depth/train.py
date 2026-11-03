"""Original S3-U phases from exact step zero, optional early ordinal prior."""
import argparse
import json
import random
import shutil
import time
import numpy as np
import torch
from ..prepare_data import ROOT, REPORT, DATA, read, write, sha
from ..train_static import setup, camera
from ..stage2.train import image
from ..union.train import reconstruction
from ..union.runtime import save, restore
from ..state_recovery.runtime import restore_graphs
from ..ordinal_depth.train import extra_loss
from .prepare import EXP, WORK, INITIAL, INITIAL_SHA


def load_initial():
    from ..union.prepare import verify_sources
    verify_sources()
    assert sha(INITIAL)==INITIAL_SHA
    renderer=setup()
    from ..union.model import UnionScene
    base=read(REPORT/'S3_union/protocol.json')
    scene=UnionScene(base); opt=scene.optimizer()
    initial=torch.load(INITIAL,map_location='cpu',weights_only=False)
    assert initial['iteration']==0 and not initial['optimizer']['state']
    scene.load_state_dict(initial['model']); opt.load_state_dict(initial['optimizer'])
    restore_graphs(scene,initial['physics_graphs'])
    random.setstate(initial['rng_python']); np.random.set_state(initial['rng_numpy'])
    torch.set_rng_state(initial['rng_torch']); torch.cuda.set_rng_state_all(initial['rng_cuda'])
    return renderer,scene,opt,initial,base


def schedule(total,warmup=16000):
    if total<1: raise ValueError('Training updates start at1')
    phase='warmup' if total<=warmup else 'physics'
    step=total if phase=='warmup' else total-warmup
    return phase,step,1600 if phase=='physics' or step>12000 else 800


def step_loss(scene,renderer,opt,frame,meta,cache,total,plan,base,enabled,depth_cache):
    phase,step,width=schedule(total,plan['warmup_updates'])
    key=(frame['_index'],width)
    if key not in cache:
        rgba=image(frame,width)
        cache[key]=(rgba,torch.tensor(rgba[...,:3]*rgba[...,3:4]).permute(2,0,1),camera(frame,meta,width))
    rgba,cpu_gt,cam=cache[key]
    opt.zero_grad(set_to_none=True)
    package=scene.render(renderer,cam,frame['time'])
    rgb=reconstruction(package,cpu_gt.cuda(),rgba,scene,cam)
    loss=rgb
    record={'global_step':total,'phase':phase,'phase_step':step,'width':width,'frame_index':frame['_index'],'rgb':float(rgb.detach())}
    if phase=='physics' and step%base['physics_every']==0:
        from variants.momentum_guard.balance import balance
        terms,rows=scene.physics.loss(scene,frame['time'],step,base['physics_samples_global'],base['time_step'])
        physical=base['kinematic_weight']*terms['kinematic']+base['CMR_weight']*terms['CMR']
        groups={'grid':list(scene.grids.parameters()),'deformation':list(scene.networks.parameters()),'canonical':[scene.pools[0]._xyz]}
        guarded,guard=balance(rgb,physical,groups,list(scene.physics.parameters()),base['shared_gradient_cap'],1.)
        loss=loss+guarded
        record['physics']={'terms':{k:float(v.detach()) for k,v in terms.items()},'diagnostics':rows,'guard':guard}
    if enabled:
        depth=extra_loss(scene,renderer,frame,meta,total,plan,depth_cache)
        weight=plan['ordinal_weight']*min(1.,total/plan['ordinal_ramp'])
        loss=loss+weight*depth
        record.update(ordinal=float(depth.detach()),ordinal_weight=weight)
    assert loss.isfinite()
    return loss,record


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--enable-early-depth-experiment',action='store_true')
    p.add_argument('--arm',choices=['control','ordinal'],default='control')
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--stop-after',type=int)
    args=p.parse_args()
    if not args.enable_early_depth_experiment: raise SystemExit('Early-depth experiment is opt-in')
    torch.set_num_threads(4)
    plan,protocol_hash=read(EXP/'protocol.json'),sha(EXP/'protocol.json')
    if not args.smoke: assert read(EXP/'preflight.json')['passed']
    directory=WORK/('_smoke' if args.smoke else 'formal')/args.arm
    directory.mkdir(parents=True,exist_ok=True)
    import msvcrt
    lock=open(WORK/'training.lock','a+b'); lock.seek(0); lock.write(b'0'); lock.flush(); lock.seek(0)
    try: msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    except OSError: raise SystemExit('Early-depth training already running')
    try:
        renderer,scene,opt,initial,base=load_initial()
        if (directory/'checkpoint.pt').exists():
            state=restore(directory/'checkpoint.pt',scene,opt,protocol_hash)
            assert state['phase']==args.arm and state['parent_sha256']==INITIAL_SHA
            start,stack=state['iteration'],list(state['camera_stack'])
            write(directory/f'resume_{start:05d}.json',{'iteration':start,'full_state_restored':True})
            if start>=plan['warmup_updates']:
                admission=read(EXP/'warmup_admission.json')
                assert admission['passed'] and admission['native_images_reviewed'], 'Warmup visual review is required before physics'
        else:
            start,stack=0,list(initial['camera_stack'])
            save(directory,scene,opt,0,stack,args.arm,protocol_hash,INITIAL_SHA)
            shutil.copyfile(directory/'checkpoint.pt',directory/'initial.pt')
        frozen={k:v.detach().clone() for k,v in scene.static.state_dict().items()}
        meta=read(DATA/'transforms_train.json')
        for i,f in enumerate(meta['frames']): f['_index']=i
        cache,depth_cache,began={},{},time.monotonic()
        target=plan['smoke_updates'] if args.smoke else plan['actual_updates']
        for total in range(start+1,target+1):
            if (WORK/'stop.request').exists():
                save(directory,scene,opt,total-1,stack,args.arm,protocol_hash,INITIAL_SHA); return
            if not stack: stack=random.sample(range(120),120)
            index=stack.pop()
            loss,record=step_loss(scene,renderer,opt,meta['frames'][index],meta,cache,total,plan,base,args.arm=='ordinal',depth_cache)
            loss.backward()
            assert all(v.grad is None or v.grad.isfinite().all() for v in scene.parameters())
            opt.step(); scene.constrain()
            with (directory/'steps.jsonl').open('a',encoding='utf-8') as stream: stream.write(json.dumps(record)+'\n')
            if total in [1,40,80,100] or total%200==0:
                assert all(torch.equal(v,scene.static.state_dict()[k]) for k,v in frozen.items())
                print('EARLY_DEPTH_TRAIN',args.arm,total,record['phase'],float(loss.detach()),record.get('ordinal'),'elapsed',round(time.monotonic()-began,1),flush=True)
                write(EXP/'progress.json',{'mode':'smoke' if args.smoke else 'formal','arm':args.arm,'actual_updates':total,'target':target,'acceptance':'pending'})
            if total in [40,80,100] or total%500==0 or args.stop_after and total>=args.stop_after:
                save(directory,scene,opt,total,stack,args.arm,protocol_hash,INITIAL_SHA)
            if args.stop_after and total>=args.stop_after: return
            if total==plan['warmup_updates']:
                from .validation import validate_warmup
                admission=validate_warmup(renderer,scene,base)
                write(EXP/'warmup_admission.json',admission)
                if not admission['passed']: raise RuntimeError('Original warmup numerical guardrails failed; physics not admitted')
                # Save and return for actual native-image review, as original
                # S3-U requires. Do not automatically admit physics on numbers.
                save(directory,scene,opt,total,stack,args.arm,protocol_hash,INITIAL_SHA)
                print('EARLY_WARMUP_PENDING_VISUAL_REVIEW',flush=True)
                return
        save(directory,scene,opt,target,stack,args.arm,protocol_hash,INITIAL_SHA)
    except Exception as error:
        write(directory/'failure.json',{'error':repr(error),'attempted_update':locals().get('total')}); raise
    finally:
        lock.seek(0); msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1); lock.close()


if __name__=='__main__': main()
