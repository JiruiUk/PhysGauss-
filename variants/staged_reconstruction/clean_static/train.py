"""Opt-in matched full-state G1 continuation; no growth or new objective."""
import argparse
import json
import msvcrt
import os
import random
import time
import cv2
import numpy as np
import torch
import torch.nn.functional as functional
from ..prepare_data import ROOT, DATA, read, write, sha
from ..adaptive_static.model import AdaptiveGaussian, replace_rows
from ..adaptive_static.prepare import RUNS as PREVIOUS_RUNS
from ..adaptive_static.train import refresh, confidence_hash
from ..train_static import setup, camera, render_static, optimizer_for
from ..automatic_static.discover import rgb
from ..cross_temporal_static.core import weighted_objective
from .prepare import REPORT, RUNS, verify


def rng_restore(state):
    random.setstate(state['rng_python']); np.random.set_state(state['rng_numpy'])
    torch.set_rng_state(state['rng_torch']); torch.cuda.set_rng_state_all(state['rng_cuda'])


def initialize(model, arm, protocol):
    parent = torch.load(ROOT/protocol['parent'], map_location='cpu', weights_only=False)
    assert parent['iteration'] == 12000
    model.load_payload(parent['model']); opt = optimizer_for(model); opt.load_state_dict(parent['optimizer'])
    with np.load(RUNS/'parent_evidence.npz') as data: rejected = data['quarantine'].copy()
    if arm == 'visibility': replace_rows(model, opt, ~rejected)
    geometry_path = (RUNS if arm == 'visibility' else PREVIOUS_RUNS)/'geometric_confidence.npz'
    with np.load(geometry_path) as data: geometry = {n:data[n].copy() for n in data.files}
    confidence = {n:np.minimum(v, geometry[n]) for n,v in parent['confidence'].items()}
    removed = parent['model']['point_ids'][torch.from_numpy(rejected)].tolist() if arm == 'visibility' else []
    rng_restore(parent)
    return opt, parent['camera_stack'].copy(), confidence, geometry, removed, parent['confidence_history']


def save(directory, model, opt, step, stack, confidence, history, arm, removed, parent_history):
    state = {'model':model.state_dict(), 'optimizer':opt.state_dict(), 'iteration':step,
        'camera_stack':stack, 'confidence':confidence, 'confidence_history':history,
        'parent_confidence_history':parent_history, 'confidence_sha256':confidence_hash(confidence),
        'arm':arm, 'removed_ids':removed, 'rng_python':random.getstate(), 'rng_numpy':np.random.get_state(),
        'rng_torch':torch.get_rng_state(), 'rng_cuda':torch.cuda.get_rng_state_all(),
        'protocol_sha256':sha(REPORT/'protocol.json'), 'source_manifest_sha256':sha(REPORT/'source_freeze.json'),
        'parent_sha256':read(REPORT/'protocol.json')['parent_sha256'], 'static_only':True, 'dynamic_parameters':0}
    temporary = directory/'checkpoint.tmp'; torch.save(state,temporary); os.replace(temporary,directory/'checkpoint.pt')
    write(directory/'checkpoint.json', {'iteration':step,'sha256':sha(directory/'checkpoint.pt'),
        'confidence_sha256':state['confidence_sha256']})


def restore(directory, model, arm):
    assert sha(directory/'checkpoint.pt') == read(directory/'checkpoint.json')['sha256']
    state = torch.load(directory/'checkpoint.pt',map_location='cpu',weights_only=False)
    assert state['arm'] == arm and state['protocol_sha256'] == sha(REPORT/'protocol.json')
    assert state['source_manifest_sha256'] == sha(REPORT/'source_freeze.json')
    assert state['confidence_sha256'] == confidence_hash(state['confidence'])
    model.load_payload(state['model']); opt = optimizer_for(model); opt.load_state_dict(state['optimizer'])
    rng_restore(state)
    return opt, state


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--enable-clean-static',action='store_true')
    parser.add_argument('--arm',choices=['control','visibility'],default='control')
    parser.add_argument('--smoke',action='store_true'); parser.add_argument('--stop-after',type=int)
    args = parser.parse_args()
    if not args.enable_clean_static: raise SystemExit('Clean static reconstruction is opt-in')
    verify(); assert read(REPORT/'candidate_review.json')['accepted_for_bounded_pilot']
    assert read(REPORT/'test_receipt.json')['passed']
    if not args.smoke: assert read(REPORT/'smoke_verification.json')['passed']
    assert not (RUNS/'stop.request').exists(), 'Saved user stop; do not resume'
    lock = (RUNS/'experiment.lock').open('a+b'); lock.seek(0)
    if not lock.read(1): lock.write(b'0'); lock.flush()
    lock.seek(0); msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    directory = RUNS/('smoke' if args.smoke else 'formal')/args.arm; directory.mkdir(parents=True,exist_ok=True)
    try:
        protocol = read(REPORT/'protocol.json'); torch.set_num_threads(4); cv2.setNumThreads(2)
        renderer = setup(); model = AdaptiveGaussian(PREVIOUS_RUNS/'initialization.npz')
        opt, stack, confidence, geometry, removed, parent_history = initialize(model,args.arm,protocol)
        meta = read(DATA/'transforms_train.json')
        frames = sorted(meta['frames'],key=lambda f:int(f['file_path'].replace('\\','/').split('/')[-1]))
        with np.load(PREVIOUS_RUNS/'geometric_confidence.npz') as original:
            pool = [f for f in frames if (original[str(int(f['file_path'].replace('\\','/').split('/')[-1]))] >= .5).sum() >= 64]
        assert len(pool) == 120
        contract = {'frames':[f['file_path'] for f in pool], 'common_original_G1_pool':True,
                    'parent_sampler_continued':True,'no_new_random_seed':True}
        if (directory/'sampling_contract.json').exists(): assert read(directory/'sampling_contract.json') == contract
        else: write(directory/'sampling_contract.json',contract)
        if not (directory/'checkpoint.pt').exists():
            save(directory,model,opt,12000,stack,confidence,[],args.arm,removed,parent_history)
            (directory/'initial.pt').write_bytes((directory/'checkpoint.pt').read_bytes())
        opt,state = restore(directory,model,args.arm)
        step0,stack,confidence,history = state['iteration'],state['camera_stack'],state['confidence'],state['confidence_history']
        # Check native adapter without consuming any restored random state.
        from scene_PIDG.gaussian_model import GaussianModel
        native = GaussianModel(3); native.active_sh_degree = 3
        for name,parameter in model.named_parameters(): setattr(native,name,parameter)
        with torch.no_grad():
            cam = camera(pool[0],meta,800); a = render_static(renderer,cam,model); b = render_static(renderer,cam,native)
            assert torch.equal(a['render'],b['render']) and torch.equal(a['alpha'],b['alpha'])
        del a,b,native; rng_restore(state)
        write(REPORT/f'native_parity_{args.arm}.json',{'passed':True,'native_original_GaussianModel_exact':True})
        target = 12080 if args.smoke else 12000+protocol['continuation_updates']
        interval = 10 if args.smoke else protocol['confidence_interval']
        targets = {str(int(f['file_path'].replace('\\','/').split('/')[-1])):rgb(f,400) for f in frames}
        cache,cameras,masks = {},{},{}; began = time.monotonic(); width = 1600
        write(RUNS/'state.json',{'status':'training','arm':args.arm,'iteration':step0,'pid':os.getpid(),'smoke':args.smoke})
        for step in range(step0+1,target+1):
            if not stack: stack = random.sample(range(len(pool)),len(pool))
            index = stack.pop(); frame = pool[index]; n = str(int(frame['file_path'].replace('\\','/').split('/')[-1]))
            if index not in cache:
                cache[index] = torch.from_numpy(rgb(frame,width)).permute(2,0,1); cameras[index] = camera(frame,meta,width)
            if index not in masks:
                masks[index] = functional.interpolate(torch.from_numpy(confidence[n])[None,None].cuda(),size=(900,1600),mode='nearest')[0,0]
            weight = masks[index]; opt.zero_grad(set_to_none=True)
            package = render_static(renderer,cameras[index],model)
            loss = weighted_objective(package['render'],package['alpha'],cache[index].cuda(),weight,protocol['admission_threshold'])
            loss = loss+.01*(model._xyz-model.initial_xyz).square().mean()
            assert torch.isfinite(loss); loss.backward()
            assert all(p.grad is None or p.grad.isfinite().all() for p in model.parameters())
            opt.step(); model.constrain()
            with (directory/'sampling.jsonl').open('a') as log:
                log.write(json.dumps({'step':step,'frame_index':index,'frame':n,'supervised_pixels':int((weight >= .5).sum())})+'\n')
            if step % 20 == 0:
                with (directory/'losses.jsonl').open('a') as log: log.write(json.dumps({'step':step,'loss':float(loss.detach()),'width':width})+'\n')
            if step % interval == 0:
                event = refresh(renderer,model,frames,meta,confidence,geometry,targets,protocol,step); history.append(event); masks.clear()
                write(directory/'confidence_history.json',history)
                print('CLEAN_STATIC_REFRESH',args.arm,step,event['positive_pixels'],flush=True)
            if step % 200 == 0:
                print('CLEAN_STATIC_TRAIN',args.arm,step,float(loss.detach()),'elapsed',round(time.monotonic()-began,1),flush=True)
                write(RUNS/'state.json',{'status':'training','arm':args.arm,'iteration':step,'pid':os.getpid(),'smoke':args.smoke})
            if step % 1000 == 0: save(directory,model,opt,step,stack,confidence,history,args.arm,removed,parent_history)
            if (RUNS/'stop.request').exists() or args.stop_after and step >= args.stop_after:
                save(directory,model,opt,step,stack,confidence,history,args.arm,removed,parent_history)
                write(RUNS/'state.json',{'status':'saved_stopped','arm':args.arm,'iteration':step,'smoke':args.smoke}); return
        save(directory,model,opt,target,stack,confidence,history,args.arm,removed,parent_history); verify()
        write(RUNS/'state.json',{'status':'trained_pending_evaluation','arm':args.arm,'iteration':target,'smoke':args.smoke})
    except Exception as error:
        write(RUNS/'state.json',{'status':'failed','arm':args.arm,'error':str(error),'pid':os.getpid()}); raise
    finally:
        lock.seek(0); msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1); lock.close()


if __name__ == '__main__': main()
