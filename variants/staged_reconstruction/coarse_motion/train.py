"""Original S3-U optimization with an optional small shared motion representation."""
import argparse
import json
import random
import shutil
import time
import numpy as np
import torch
from ..prepare_data import DATA,read,write,sha
from ..early_depth.train import load_initial,schedule,step_loss as original_step
from ..union.runtime import save,restore
from .prepare import EXP,WORK,INITIAL_SHA
from .core import install


def step_loss(scene,renderer,opt,frame,meta,cache,total,plan,base,enabled):
    loss,row=original_step(scene,renderer,opt,frame,meta,cache,total,plan,base,False,{})
    row['shared_motion']=enabled
    row['control_count']=plan['control_count'] if enabled else None
    return loss,row


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--enable-coarse-motion-experiment',action='store_true')
    parser.add_argument('--arm',choices=['control','coarse'],default='control')
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--stop-after',type=int)
    args=parser.parse_args()
    if not args.enable_coarse_motion_experiment: raise SystemExit('Coarse-motion experiment is opt-in')
    torch.set_num_threads(4)
    plan=read(EXP/'protocol.json'); protocol_hash=sha(EXP/'protocol.json')
    if not args.smoke:
        from .freeze import verify
        assert read(EXP/'preflight.json')['passed']; verify()
    directory=WORK/('_smoke' if args.smoke else 'formal')/args.arm
    directory.mkdir(parents=True,exist_ok=True)
    import msvcrt
    lock=open(WORK/'training.lock','a+b'); lock.seek(0); lock.write(b'0'); lock.flush(); lock.seek(0)
    try: msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    except OSError: raise SystemExit('Coarse-motion training already running')
    try:
        renderer,scene,opt,initial,base=load_initial()
        enabled=args.arm=='coarse' and plan['shared_motion_enabled']
        adapter=install(scene,plan,enabled)
        if (directory/'checkpoint.pt').exists():
            state=restore(directory/'checkpoint.pt',scene,opt,protocol_hash)
            assert state['phase']==args.arm and state['parent_sha256']==INITIAL_SHA
            start,stack=state['iteration'],list(state['camera_stack'])
            write(directory/f'resume_{start:05d}.json',{'iteration':start,'full_state_restored':True})
            if start>=plan['warmup_updates']:
                admission=read(EXP/'warmup_admission.json')
                assert admission['passed'] and admission['native_images_reviewed'],'Warmup admission required'
        else:
            start,stack=0,list(initial['camera_stack'])
            save(directory,scene,opt,0,stack,args.arm,protocol_hash,INITIAL_SHA)
            shutil.copyfile(directory/'checkpoint.pt',directory/'initial.pt')
        frozen={key:value.detach().clone() for key,value in scene.static.state_dict().items()}
        meta=read(DATA/'transforms_train.json')
        for i,frame in enumerate(meta['frames']): frame['_index']=i
        cache={}; began=time.monotonic(); target=plan['smoke_updates'] if args.smoke else plan['actual_updates']
        for total in range(start+1,target+1):
            if (WORK/'stop.request').exists():
                save(directory,scene,opt,total-1,stack,args.arm,protocol_hash,INITIAL_SHA); return
            if not stack: stack=random.sample(range(120),120)
            index=stack.pop()
            loss,row=step_loss(scene,renderer,opt,meta['frames'][index],meta,cache,total,plan,base,enabled)
            loss.backward()
            assert all(p.grad is None or p.grad.isfinite().all() for p in scene.parameters())
            opt.step(); scene.constrain()
            with (directory/'steps.jsonl').open('a',encoding='utf-8') as stream: stream.write(json.dumps(row)+'\n')
            if total in [1,40,80,100] or total%200==0:
                assert all(torch.equal(value,scene.static.state_dict()[key]) for key,value in frozen.items())
                print('COARSE_MOTION_TRAIN',args.arm,total,row['phase'],float(loss.detach()),row['shared_motion'],'elapsed',round(time.monotonic()-began,1),flush=True)
                write(EXP/'progress.json',{'mode':'smoke' if args.smoke else 'formal','arm':args.arm,'actual_updates':total,'target':target,'acceptance':'pending'})
            if total in [40,80,100] or total%500==0 or args.stop_after and total>=args.stop_after:
                save(directory,scene,opt,total,stack,args.arm,protocol_hash,INITIAL_SHA)
            if args.stop_after and total>=args.stop_after: return
            if total==plan['warmup_updates']:
                from .validation import validate_warmup
                admission=validate_warmup(renderer,scene,base); write(EXP/'warmup_admission.json',admission)
                if not admission['passed']: raise RuntimeError('Original warmup numerical guardrails failed; physics not admitted')
                save(directory,scene,opt,total,stack,args.arm,protocol_hash,INITIAL_SHA)
                print('COARSE_MOTION_WARMUP_PENDING_VISUAL_REVIEW',flush=True); return
        save(directory,scene,opt,target,stack,args.arm,protocol_hash,INITIAL_SHA)
    except Exception as error:
        write(directory/'failure.json',{'error':repr(error),'attempted_update':locals().get('total')}); raise
    finally:
        lock.seek(0); msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1); lock.close()


if __name__=='__main__': main()
