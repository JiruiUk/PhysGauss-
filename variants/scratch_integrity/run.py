"""Matched cold-start training, training-only eligibility, complete-state arms."""
from pathlib import Path
from datetime import datetime,timedelta,timezone
import argparse
import os
import sys
import shutil
import subprocess
import time
import json
import psutil
import torch
from . import prefix as common
from .prefix import ROOT,REPORT,RUNS,PREFIX,DATA,RECIPE,plan,step
from .prepare import prepare as prepare_code
from variants.metric_tuning import run as engine
from variants.association_training.run import tensor_equal

NAMES=['S0_scratch_control','S1_scratch_transfer']


def configuration(on,binding):
    p=plan();c=engine.read_json(RECIPE/'metric_config.json');c.update(p['terms'])
    c.update(transfer_integrity_enabled=on,transfer_data=binding['path'],transfer_data_sha256=binding['sha256'],
             transfer_source_sha256=binding['source_checkpoint_sha256'],transfer_start=binding['source_iteration'],
             transfer_interval=p['interval'],transfer_ramp=p['ramp_updates'],static_samples=p['static_points_per_term'],
             track_samples=p['tracked_points_per_term'],record_transfer_sampling=True)
    return c


def snapshot(boundary):
    directory=RUNS/f'common_candidates/{boundary}/rubber_duck';directory.mkdir(parents=True,exist_ok=True)
    checkpoint=PREFIX/'checkpoints/resume.pt';digest=engine.sha256(checkpoint)
    if (directory/'checkpoint_proof.json').exists():
        assert engine.sha256(directory/'checkpoints/resume.pt')==digest
        return directory
    for name in ['metric_config.json','model_config.py','metric_history.jsonl','metric_best.json']:
        shutil.copyfile(PREFIX/name,directory/name)
    (directory/'checkpoints').mkdir(exist_ok=True)
    for p in checkpoint.parent.glob('resume.*'):shutil.copyfile(p,directory/'checkpoints'/p.name)
    assert engine.sha256(directory/'checkpoints/resume.pt')==digest
    engine.write_json(directory/'checkpoint_proof.json',{'iteration':boundary,'sha256':digest,'source':str(PREFIX),'immutable_copy':True})
    return directory


def branch(name,on,binding,smoke=False):
    directory=RUNS/('_smoke' if smoke else 'pilot')/name/'rubber_duck';c=configuration(on,binding)
    if (directory/'branch.json').exists():
        assert engine.read_json(directory/'metric_config.json')==c
        return directory
    assert not (directory/'checkpoints/resume.pt').exists(),'Unreceipted branch state'
    directory.mkdir(parents=True,exist_ok=True);engine.write_json(directory/'metric_config.json',c)
    source=Path(binding['source_directory']);shutil.copyfile(source/'model_config.py',directory/'model_config.py')
    (directory/'checkpoints').mkdir(exist_ok=True)
    for p in (source/'checkpoints').glob('resume.*'):shutil.copyfile(p,directory/'checkpoints'/p.name)
    digest=engine.sha256(directory/'checkpoints/resume.pt');assert digest==binding['source_checkpoint_sha256']
    side=directory/'checkpoints/resume.metrics.json';state=engine.read_json(side)
    assert state['config']==engine.read_json(source/'metric_config.json')
    state.update(config=c,transfer_data_sha256=binding['sha256']);engine.write_json(side,state)
    engine.write_json(directory/'branch.json',{'source':str(PREFIX),'destination':str(directory),'iteration':binding['source_iteration'],
                     'immutable_binding_source':str(source),'identical_full_checkpoint_sha256':digest,
                     'model_optimizer_RNG_sampler_copied_byte_identically':True,'source_config':engine.read_json(source/'metric_config.json'),
                     'branch_config':c,'pretrained_model_used':False})
    engine.write_json(directory/'reproduction.json',{'id':name,'commands':[],'status':'initialized','target_iteration':50000,
                     'seed':plan()['seed'],'shared_fresh_prefix':binding['source_iteration'],'pretrained_model_used':False})
    return directory


def freeze(repo):
    paths=[*Path(__file__).parent.glob('*.py'),ROOT/'tests/test_scratch_integrity.py',REPORT/'plan.json',REPORT/'image_evidence_lock.json']
    paths += [ROOT/p for p in ['variants/color_repair/runtime.py','variants/coverage_tuning/runtime.py',
              'variants/metric_tuning/runtime.py','variants/kinematic_resplit/checkpoint.py','reproduction/scripts/training_checkpoint.py']]
    value={'own_and_inherited':{p.relative_to(ROOT).as_posix():engine.sha256(p) for p in paths},
           'fork':engine.runtime_hashes(repo),'C1_parent':engine.runtime_hashes(ROOT/'variants/color_repair/code'),
           'image_evidence':engine.read_json(REPORT/'image_evidence_lock.json')}
    assert all(engine.sha256(ROOT/p)==h for p,h in value['image_evidence'].items())
    p=REPORT/'training_freeze.json'
    if p.exists():assert engine.read_json(p)==value,'Frozen source/image evidence changed'
    else:engine.write_json(p,value)


def preflight(repo,binding,deadline):
    saved=REPORT/'preflight.json'
    if saved.exists():
        assert engine.read_json(saved)['binding_sha256']==binding['sha256'];return
    env=engine.env_for(RUNS)
    process=subprocess.run([sys.executable,'-u','-m','variants.scratch_integrity.probe'],cwd=ROOT,env=env,capture_output=True,text=True)
    (REPORT/'optimizer_probe.log').write_text(process.stdout+process.stderr,encoding='utf-8')
    assert process.returncode==0,'Cold-state real gradient/update probe failed'
    start=binding['source_iteration'];target=start+10;original=ROOT/'variants/color_repair/code'
    directories=[]
    for name,code in [('flags_off',repo),('C1_original_path',original),('C1_original_repeat',original)]:
        d=branch(name,False,binding,True);directories.append(d)
        if step(d)<target:assert common.train(d,code,target,deadline)==75
        assert step(d)==target
    states=[torch.load(d/'checkpoints/resume.pt',map_location='cpu',weights_only=False) for d in directories]
    differences={};noise={}
    for key in ['gaussian_parameters','gaussian_buffers','velocity_net','grid4d','deform']:
        def stats(s):
            delta=torch.cat([(v.double()-states[1][key][k].double()).flatten() for k,v in s[key].items()
                             if v.numel() and not (key=='gaussian_buffers' and k=='max_radii2D')])
            return {'max_abs':float(delta.abs().max()),'RMS':float(delta.square().mean().sqrt())}
        differences[key]=stats(states[0]);noise[key]=stats(states[2])
        assert differences[key]['RMS']<=3*noise[key]['RMS']+1e-7,(key,differences[key],noise[key])
    # This screen-space maximum is only consumed by densify_and_prune, which
    # has ended. Preserve and bound its discrete rounding discrepancy separately.
    import runpy
    assert start>=runpy.run_path(str(RECIPE/'model_config.py'))['densify_until_iter']
    assert not configuration(False,binding)['coverage_density']
    radius=states[0]['gaussian_buffers']['max_radii2D']-states[1]['gaussian_buffers']['max_radii2D']
    radius_audit={'different_elements':int((radius!=0).sum()),'max_abs':float(radius.abs().max()),
                  'used_by_future_optimizer_or_renderer':False,'densification_has_ended':True}
    assert radius_audit['different_elements']<=1 and radius_audit['max_abs']<=1
    native=engine.read_json(REPORT/'flags_off_native_render_audit.json')
    assert native['script_sha256']==engine.sha256(REPORT/'parity_render.py')
    for name,proof in native['states'].items():
        assert proof['sha256']==engine.sha256(RUNS/'_smoke'/name/'rubber_duck/checkpoints/resume.pt')
    for channel in ['render','alpha']:
        repeat=max(native['results'][n]['aggregate_RMS'][channel] for n in ['C1_original_repeat','C1_original_repeat2','C1_original_repeat3'])
        assert native['results']['flags_off']['aggregate_RMS'][channel]<=3*repeat+1e-7
    for key in ['rng_python','rng_numpy','rng_torch','rng_cuda','viewpoint_stack']:
        assert tensor_equal(states[0][key],states[1][key]) and tensor_equal(states[2][key],states[1][key]),key
    enabled=branch('enabled_resume',True,binding,True)
    if step(enabled)<target:assert common.train(enabled,repo,target,deadline)==75
    if step(enabled)<start+20:assert common.train(enabled,repo,start+20,deadline)==75
    assert step(enabled)==start+20
    assert len((enabled/'transfer_loss.jsonl').read_text().splitlines())==2
    assert f'FULL_CHECKPOINT_RESTORED completed={target} next={target+1}' in (enabled/'train.log').read_text()
    assert not (directories[0]/'transfer_loss.jsonl').exists()
    engine.write_json(saved,{'status':'passed','binding_sha256':binding['sha256'],'native_resolution':[1600,900],
        'off_updates':10,'enabled_updates':20,'resume_boundary':[target,start+20],'actual_gradient_optimizer_probe':True,
        'off_vs_original_parameter_difference':differences,'unchanged_original_repeat_difference':noise,'RNG_sampler_exact':True,
        'post_densification_screen_radius_audit':radius_audit,'native_render_repeat_audit_sha256':engine.sha256(REPORT/'flags_off_native_render_audit.json'),
        'numerical_gate':'RMS <= 3x unchanged original-repeat RMS + 1e-7; exact RNG/sampler; max_radii2D bounded separately after its last use; native outputs within original repeat envelope; no bitwise CUDA determinism claim'})


def wait_prefix(deadline):
    state=engine.read_json(RUNS/'controller.json')
    if state.get('status')!='common_prefix':return
    pid=state['pid']
    while state.get('status')=='common_prefix':
        assert psutil.pid_exists(pid),'Common-prefix process stopped unexpectedly; inspect prefix.log'
        assert datetime.now(timezone.utc)<deadline,'Common prefix wait budget exhausted'
        time.sleep(5);state=engine.read_json(RUNS/'controller.json')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--preflight-only',action='store_true');parser.add_argument('--hours',type=float,default=12)
    args=parser.parse_args();torch.set_num_threads(4);repo=prepare_code()
    deadline=datetime.now(timezone.utc)+timedelta(hours=args.hours)
    wait_prefix(deadline);common.wire(repo);freeze(repo)
    state={'status':'binding','pid':os.getpid(),'seed':plan()['seed'],'started_utc':engine.now()}
    try:
        binding=None
        for n in plan()['common_prefix_candidates']:
            if step()>n:continue
            if step()<n:common.run(n,args.hours)
            snapshot(n)
            process=subprocess.run([sys.executable,'-u','-m','variants.scratch_integrity.data'],cwd=ROOT,env=engine.env_for(RUNS),capture_output=True,text=True)
            (REPORT/f'binding_{n}.log').write_text(process.stdout+process.stderr,encoding='utf-8')
            if process.returncode==42:
                assert (REPORT/f'binding_{n}/gate_rejected.json').exists();continue
            assert process.returncode==0,f'Binding failed unexpectedly at {n}'
            binding=engine.read_json(REPORT/'data_binding.json');break
        assert binding is not None,'No predeclared prefix passed the original reliability gates'
        common.wire(repo);state.update(status='preflight',activation=binding['source_iteration']);engine.write_json(RUNS/'controller.json',state)
        freeze(repo);preflight(repo,binding,deadline);freeze(repo)
        if args.preflight_only:state['status']='preflight_passed';return
        arms=[branch(n,on,binding) for n,on in zip(NAMES,[False,True])]
        configs=[engine.read_json(d/'metric_config.json') for d in arms]
        assert {k for k in configs[0] if configs[0][k]!=configs[1][k]}=={'transfer_integrity_enabled'}
        engine.write_json(REPORT/'branch_lock.json',{'activation':binding['source_iteration'],'binding':binding,
                          'common_sha256':binding['source_checkpoint_sha256'],'only_main_flag_differs':True,
                          'initialization':engine.read_json(REPORT/'initialization.json')})
        state['status']='training';engine.write_json(RUNS/'controller.json',state)
        for d in arms:
            if step(d)<50000:assert common.train(d,repo,50000,deadline)==75
            assert step(d)==50000
            freeze(repo)
        samples=[[json.loads(x) for x in (d/'camera_sampling.jsonl').read_text().splitlines()] for d in arms]
        assert samples[0]==samples[1] and len(samples[0])==50000-binding['source_iteration']
        assert [x['iteration'] for x in samples[0]]==list(range(binding['source_iteration']+1,50001))
        assert len((arms[1]/'transfer_loss.jsonl').read_text().splitlines())==(50000-binding['source_iteration'])//plan()['interval']
        state['status']='trained_pending_evaluation'
    except Exception as error:
        state.update(status='failed',error=str(error));raise
    finally:
        state['updated_utc']=engine.now();engine.write_json(RUNS/'controller.json',state)


if __name__=='__main__':main()
