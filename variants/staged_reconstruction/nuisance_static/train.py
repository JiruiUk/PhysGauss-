"""Opt-in matched full-RGB synthetic Gaussian training with complete recovery."""
import argparse
import json
import msvcrt
import os
import random
import time
import cv2
import numpy as np
import torch
from torch import nn
from ..prepare_data import ROOT,read,write,sha
from ..adaptive_static.model import AdaptiveGaussian
from ..train_static import setup,optimizer_for,render_static
from ..synthetic_audit.input import load
from .prepare import REPORT,RUNS,verify
from .core import NuisancePlane,optimizer_for_nuisance
from .runtime import calibrated_camera,layers,objective


def restore_rng(state):
    random.setstate(state['rng_python']); np.random.set_state(state['rng_numpy'])
    torch.set_rng_state(state['rng_torch']); torch.cuda.set_rng_state_all(state['rng_cuda'])


def save(directory,static,opt,bank,opts,step,stack,visits,arm,case):
    state = {'model':static.state_dict(),'optimizer':opt.state_dict(),'iteration':step,
        'nuisance':bank.state_dict() if bank is not None else None,
        'nuisance_optimizers':[o.state_dict() for o in opts], 'visits':visits,
        'camera_stack':stack,'rng_python':random.getstate(),'rng_numpy':np.random.get_state(),
        'rng_torch':torch.get_rng_state(),'rng_cuda':torch.cuda.get_rng_state_all(),
        'arm':arm,'case':case,'protocol_sha256':sha(REPORT.parent/'protocol.json'),
        'source_manifest_sha256':sha(REPORT/'source_freeze.json'),'physical_dynamic_parameters':0,
        'point_ids_fixed':True,'primitive_count_is_not_body_count':True}
    torch.save(state,directory/'checkpoint.tmp'); os.replace(directory/'checkpoint.tmp',directory/'checkpoint.pt')
    write(directory/'checkpoint.json',{'iteration':step,'sha256':sha(directory/'checkpoint.pt')})


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--enable-nuisance',action='store_true')
    parser.add_argument('--arm',choices=['full_RGB_control','full_RGB_nuisance'],default='full_RGB_control')
    parser.add_argument('--case',choices=['diffuse','glossy'],default='diffuse')
    parser.add_argument('--smoke',action='store_true'); parser.add_argument('--stop-after',type=int)
    args = parser.parse_args()
    if not args.enable_nuisance: raise SystemExit('Nuisance reconstruction is opt-in')
    verify(); assert read(REPORT/'test_receipt.json')['passed']
    assert read(REPORT/'native_check.json')['passed']
    if not args.smoke: assert read(REPORT/'smoke_verification.json')['passed']
    assert not (RUNS/'stop.request').exists(), 'Saved user stop; do not resume'
    lock = (RUNS/'experiment.lock').open('a+b'); lock.seek(0)
    if not lock.read(1): lock.write(b'0'); lock.flush()
    lock.seek(0); msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    directory = RUNS/('smoke' if args.smoke else 'formal')/args.case/args.arm; directory.mkdir(parents=True,exist_ok=True)
    try:
        torch.set_num_threads(4); cv2.setNumThreads(2); protocol = read(REPORT.parent/'protocol.json')
        renderer = setup(); values = load(ROOT/protocol['synthetic_data']/args.case/'train')
        images,views,k = values['rgb'],values['views'],values['K']; height,width = images.shape[1:3]
        static = AdaptiveGaussian(RUNS/args.case/'initialization.npz'); opt = optimizer_for(static)
        enabled = args.arm == 'full_RGB_nuisance'; bank,opts = None,[]
        if enabled:
            bank = nn.ModuleList([NuisancePlane(view,k,images[i],count=protocol['nuisance_initial_points_per_view'],seed=20261103+i)
                                  for i,view in enumerate(views)])
            opts = [optimizer_for_nuisance(m) for m in bank]
        cameras = [calibrated_camera(v,k,width,height) for v in views]
        targets = torch.from_numpy(images).permute(0,3,1,2).cuda()
        with np.load(RUNS/args.case/'support.npz') as data: masks = torch.from_numpy(data['reliable']).cuda()
        if not (directory/'checkpoint.pt').exists():
            random.seed(20261103); np.random.seed(20261103); torch.manual_seed(20261103)
            save(directory,static,opt,bank,opts,0,[],[0]*len(views),args.arm,args.case)
            (directory/'initial.pt').write_bytes((directory/'checkpoint.pt').read_bytes())
        assert sha(directory/'checkpoint.pt') == read(directory/'checkpoint.json')['sha256']
        state = torch.load(directory/'checkpoint.pt',map_location='cpu',weights_only=False)
        assert state['arm'] == args.arm and state['case'] == args.case
        assert state['source_manifest_sha256'] == sha(REPORT/'source_freeze.json')
        assert state['protocol_sha256'] == sha(REPORT.parent/'protocol.json')
        static.load_payload(state['model']); opt = optimizer_for(static); opt.load_state_dict(state['optimizer'])
        if enabled:
            bank.load_state_dict(state['nuisance']); assert len(opts) == len(state['nuisance_optimizers'])
            for o,s in zip(opts,state['nuisance_optimizers']): o.load_state_dict(s)
        else: assert state['nuisance'] is None and not state['nuisance_optimizers']
        restore_rng(state); stack,visits = state['camera_stack'],state['visits']; began = time.monotonic()
        target = 80 if args.smoke else protocol['synthetic_updates_per_arm_case']
        write(RUNS/'state.json',{'status':'training','case':args.case,'arm':args.arm,'iteration':state['iteration'],
              'pid':os.getpid(),'smoke':args.smoke})
        for step in range(state['iteration']+1,target+1):
            if not stack: stack = random.sample(range(len(views)),len(views))
            index = stack.pop(); opt.zero_grad(set_to_none=True)
            if enabled: opts[index].zero_grad(set_to_none=True)
            package = layers(renderer,cameras[index],static,bank[index] if enabled else None,enabled)
            loss = objective(package,targets[index],masks[index],static,protocol)
            assert torch.isfinite(loss); loss.backward()
            assert all(p.grad is None or p.grad.isfinite().all() for p in static.parameters())
            if enabled: assert all(p.grad is not None and p.grad.isfinite().all() for p in bank[index].parameters())
            opt.step(); static.constrain()
            if enabled:
                opts[index].step(); bank[index].constrain(); opts[index].zero_grad(set_to_none=True)
            visits[index] += 1
            with (directory/'sampling.jsonl').open('a') as log: log.write(json.dumps({'step':step,'view':index})+'\n')
            if step%20 == 0:
                with (directory/'losses.jsonl').open('a') as log: log.write(json.dumps({'step':step,'loss':float(loss.detach())})+'\n')
            if step%200 == 0:
                print('NUISANCE_TRAIN',args.case,args.arm,step,float(loss.detach()),round(time.monotonic()-began,1),flush=True)
                write(RUNS/'state.json',{'status':'training','case':args.case,'arm':args.arm,'iteration':step,'pid':os.getpid(),'smoke':args.smoke})
            if step%1000 == 0: save(directory,static,opt,bank,opts,step,stack,visits,args.arm,args.case)
            if (RUNS/'stop.request').exists() or args.stop_after and step >= args.stop_after:
                save(directory,static,opt,bank,opts,step,stack,visits,args.arm,args.case)
                write(RUNS/'state.json',{'status':'saved_stopped','case':args.case,'arm':args.arm,'iteration':step}); return
        save(directory,static,opt,bank,opts,target,stack,visits,args.arm,args.case); verify()
        write(RUNS/'state.json',{'status':'trained_pending_evaluation','case':args.case,'arm':args.arm,'iteration':target})
    except Exception as error:
        write(RUNS/'state.json',{'status':'failed','case':args.case,'arm':args.arm,'error':str(error)}); raise
    finally:
        lock.seek(0); msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1); lock.close()


if __name__ == '__main__': main()
