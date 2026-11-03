"""S3 continuation with a strict reviewed S2 gate and full optimizer/RNG recovery."""
import argparse
import json
import math
import os
import random
import time
from pathlib import Path
import numpy as np
import torch
from ..prepare_data import ROOT,REPORT,RUNS,DATA,read,write,sha
from ..train_static import setup,camera
from ..stage2.train import image,objective,evaluate as image_evaluate
from ..geometry_recovery.model import BoundedScene,GEOM_REPORT as REC_REPORT,GEOM_RUNS as REC_RUNS
from ..recovery.coverage import coverage_loss
from .physics import PhysicsField

STAGE_REPORT=REPORT/'S3';STAGE_RUNS=RUNS/'S3'


def parent_admission():
    path=REC_REPORT/'admission.json'
    if not path.exists():raise RuntimeError('S2 reconstruction has not been reviewed/admitted; S3 training is gated')
    admitted=read(path)
    if not admitted.get('admitted') or not admitted.get('native_images_reviewed'):
        raise RuntimeError('S2 reconstruction failed admission; repair S2 before S3')
    parent=REC_RUNS/'S2_objects/checkpoint.pt'
    assert sha(parent)==admitted['checkpoint_sha256']
    return parent,admitted,sha(path)


class PhysicsScene(BoundedScene):
    def __init__(self,parent_state):
        super().__init__(read(REC_REPORT/'S2_protocol.json'))
        self.load_state_dict(parent_state['model'])
        self.physics=PhysicsField(len(self.objects))

    def optimizer_from(self,parent_state):
        opt=super().optimizer();opt.load_state_dict(parent_state['optimizer'])
        opt.add_param_group({'params':self.physics.parameters(),'lr':.001,'name':'PIDG_velocity_stress'})
        return opt


def rng_restore(s):
    random.setstate(s['rng_python']);np.random.set_state(s['rng_numpy']);torch.set_rng_state(s['rng_torch']);torch.cuda.set_rng_state_all(s['rng_cuda'])


def save(directory,scene,opt,step,stack,protocol_hash,parent_hash):
    tmp=directory/'checkpoint.tmp';file=directory/'checkpoint.pt'
    torch.save({'model':scene.state_dict(),'optimizer':opt.state_dict(),'iteration':step,'camera_stack':stack,
        'rng_python':random.getstate(),'rng_numpy':np.random.get_state(),'rng_torch':torch.get_rng_state(),
        'rng_cuda':torch.cuda.get_rng_state_all(),'protocol_sha256':protocol_hash,'parent_sha256':parent_hash,
        'stage':'S3','physical_time_units':'sequence normalized, acquisition seconds unknown'},tmp)
    os.replace(tmp,file);write(directory/'checkpoint.json',{'sha256':sha(file),'iteration':step,'full_state':True})


def reconstruction(package,gt,scene,cam,frame,observations):
    rgb=objective(package['render'],gt);r,g,b=gt
    mask=(r>.25)&(g>.18)&(b<.8*torch.minimum(r,g))
    foreground=(package['render'][:,mask]-gt[:,mask]).abs().mean() if mask.any() else gt.sum()*0
    label,mapping=observations[int(Path(frame['file_path']).stem)]
    cover=coverage_loss(package['means3D_deform'][len(scene.static.get_xyz):],scene.object_ids,cam,label,mapping)
    static=(b>r*1.3+.02)&(b>g*1.05)
    static_rgb=(package['render'][:,static]-gt[:,static]).abs().mean() if static.any() else gt.sum()*0
    return rgb+.5*foreground+.5*static_rgb+2.*cover


def evaluate(renderer,scene,step,split='val'):
    # Isolate report and file prefix while reusing the already audited numeric
    # image/alpha evaluator. The frozen S2 files are not edited or overwritten.
    from ..stage2 import train as parent
    old=parent.REPORT;parent.REPORT=STAGE_REPORT
    try:value=image_evaluate(renderer,scene,step,split,1600,False)
    finally:parent.REPORT=old
    value['stage']='S3';write(STAGE_REPORT/f'S3_{split}_{step:05d}.json',value)
    return value


def freeze(protocol):
    files=list(Path(__file__).parent.glob('*.py'))+[STAGE_REPORT/'protocol.json',ROOT/'tests/test_staged_physics.py']
    files += [ROOT/p for p in ['variants/flow_comparison/local_momentum.py','variants/coherent_momentum/operator.py',
        'variants/flow_comparison/temporal.py','variants/momentum_guard/balance.py']]
    value={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in files};file=STAGE_REPORT/'source_freeze.json'
    if file.exists():assert read(file)==value
    else:write(file,value)
    from ..geometry_recovery.freeze import verify
    verify()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--enable-physics-stage',action='store_true')
    parser.add_argument('--smoke',action='store_true');parser.add_argument('--stop-after',type=int);parser.add_argument('--evaluate-only',action='store_true');args=parser.parse_args()
    if not args.enable_physics_stage:raise SystemExit('S3 is opt-in; pass --enable-physics-stage')
    source,admission,admission_hash=parent_admission();STAGE_REPORT.mkdir(parents=True,exist_ok=True);STAGE_RUNS.mkdir(parents=True,exist_ok=True)
    protocol={'stage':'S3','enabled_by_default':False,'seed':20261105,'updates':6000,'width':1600,
        'parent_sha256':sha(source),'parent_admission_sha256':admission_hash,'CMR':True,'static_frozen':True,
        'velocity_stress':'Original PIDG VelocityNet independently per object; no free particle index embedding',
        'momentum':'Material velocity time difference + current-world quadratic WLS stress divergence; same-object full-rank neighborhoods',
        'time_step':1/199,'time_units':'Sequence normalized; actual acquisition seconds unavailable',
        'kinematic_weight':.01,'CMR_weight':.00001,'physics_every':10,'shared_gradient_cap':.1,
        'physics_samples_per_object':16,'rate_coefficient_fixed':1.,'density_fixed':1.,'contact':False,
        'dense_LPFM':False,'sparse_tracks':False,'constitutive':False,'unknown_body_force':False,
        'checkpoint_interval':500,'validation_interval':2000,'final_preselected':True,
        'reconstruction':'Accepted S2 recovery RGB/foreground/coverage unchanged; preserve unknown image identities',
        'evaluation_limit':'Physics residual and velocity consistency are internal diagnostics, not physical GT or calibrated stress'}
    path=STAGE_REPORT/'protocol.json'
    if path.exists():assert read(path)==protocol
    else:write(path,protocol)
    freeze(protocol);protocol_hash=sha(path)
    torch.set_num_threads(4);random.seed(protocol['seed']);np.random.seed(protocol['seed']);torch.manual_seed(protocol['seed'])
    renderer=setup();parent_state=torch.load(source,map_location='cpu',weights_only=False);scene=PhysicsScene(parent_state);opt=scene.optimizer_from(parent_state)
    directory=STAGE_RUNS/('_smoke' if args.smoke else 'formal');directory.mkdir(parents=True,exist_ok=True)
    import msvcrt
    lock=open(directory/'training.lock','a+b');lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0)
    try:msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    except OSError:raise SystemExit('S3 already running; no duplicate launched')
    try:
        saved=directory/'checkpoint.pt'
        if saved.exists():
            assert sha(saved)==read(directory/'checkpoint.json')['sha256']
            state=torch.load(saved,map_location='cpu',weights_only=False);assert state['protocol_sha256']==protocol_hash
            scene.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);rng_restore(state);start=state['iteration'];stack=state['camera_stack']
            print('S3_FULL_STATE_RESTORED',start,flush=True)
        else:
            rng_restore(parent_state);start=0;stack=parent_state['camera_stack']
            save(directory,scene,opt,start,stack,protocol_hash,protocol['parent_sha256'])
        if args.evaluate_only:evaluate(renderer,scene,start,'val');evaluate(renderer,scene,start,'test');return
        if not args.smoke:assert read(STAGE_REPORT/'smoke_verification.json')['passed']
        meta=read(DATA/'transforms_train.json');ledger=np.load(RUNS/'S2_observations.npz',allow_pickle=False)
        observations={int(n):(torch.tensor(c),o.tolist()) for n,c,o in zip(ledger['frame_ids'],ledger['components'],ledger['early_owner_to_component'])}
        cache={};cameras={};frozen={k:v.clone() for k,v in scene.static.state_dict().items()};began=time.monotonic();completed=start
        from variants.momentum_guard.balance import balance
        for step in range(start+1,protocol['updates']+1):
            if not stack:stack=random.sample(range(len(meta['frames'])),len(meta['frames']))
            index=stack.pop();frame=meta['frames'][index]
            if index not in cache:
                rgba=image(frame,protocol['width']);cache[index]=torch.tensor(rgba[...,:3]*rgba[...,3:4]).permute(2,0,1)
                cameras[index]=camera(frame,meta,protocol['width'])
            cam=cameras[index];gt=cache[index].cuda();opt.zero_grad(set_to_none=True)
            package=scene.render(renderer,cam,frame['time']);rgb=reconstruction(package,gt,scene,cam,frame,observations);loss=rgb
            if step%protocol['physics_every']==0:
                terms,rows=scene.physics.loss(scene,frame['time'],step,protocol['physics_samples_per_object'],protocol['time_step'])
                physical=protocol['kinematic_weight']*terms['kinematic']+protocol['CMR_weight']*terms['CMR']
                groups={'grid':list(scene.grids.parameters()),'deformation':list(scene.networks.parameters()),'canonical':[o._xyz for o in scene.objects]}
                guarded,guard=balance(rgb,physical,groups,list(scene.physics.parameters()),protocol['shared_gradient_cap'],1.)
                loss=loss+guarded
                with (directory/'physics.jsonl').open('a') as stream:stream.write(json.dumps({'step':step,'frame':frame['file_path'],
                    'terms':{k:float(v.detach()) for k,v in terms.items()},'objects':rows,'gradient_guard':guard})+'\n')
            assert bool(loss.isfinite());loss.backward()
            assert all(p.grad is None or p.grad.isfinite().all() for p in scene.parameters())
            opt.step();scene.constrain();completed=step
            if step%200==0:
                assert all(torch.equal(v,scene.static.state_dict()[k]) for k,v in frozen.items())
                print('S3_TRAIN',step,float(loss.detach()),'elapsed',round(time.monotonic()-began,1),flush=True)
                if not args.smoke:write(RUNS/'state.json',{'stage':'S3','status':'training','iteration':step,'pid':os.getpid(),'active_run':str(directory)})
            if step%protocol['checkpoint_interval']==0:save(directory,scene,opt,step,stack,protocol_hash,protocol['parent_sha256'])
            if not args.smoke and step%protocol['validation_interval']==0:evaluate(renderer,scene,step)
            if (RUNS/'stop.request').exists() or args.stop_after and step>=args.stop_after:
                save(directory,scene,opt,step,stack,protocol_hash,protocol['parent_sha256'])
                if not args.smoke:write(RUNS/'state.json',{'stage':'S3','status':'saved_stopped','iteration':step})
                return
        save(directory,scene,opt,completed,stack,protocol_hash,protocol['parent_sha256'])
        evaluate(renderer,scene,completed,'val');evaluate(renderer,scene,completed,'test');freeze(protocol)
        if not args.smoke:write(RUNS/'state.json',{'stage':'S3','status':'evaluated_pending_review','iteration':completed})
    except Exception as error:
        # A failed step may already have popped a camera or partially changed
        # optimizer state. Preserve the last atomic full checkpoint for replay.
        write(directory/'failure.json',{'error':repr(error),'last_completed_in_memory':locals().get('completed'),
            'attempted_iteration':locals().get('step'),'recovery':'Replay from existing atomic checkpoint; never relabel partial state'})
        if not args.smoke:write(RUNS/'state.json',{'stage':'S3','status':'failed','error':str(error),'iteration':locals().get('completed',0)})
        raise
    finally:
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()


if __name__=='__main__':main()
