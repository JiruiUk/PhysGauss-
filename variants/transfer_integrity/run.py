"""Matched full-state continuation with bounded preflight and immutable inputs."""
from datetime import datetime,timedelta,timezone
from pathlib import Path
import argparse
import os
import shutil
import sys
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from variants.metric_tuning import run as engine
from variants.association_training.run import tensor_equal
from .prepare import prepare as prepare_code
from .data import prepare as prepare_data

RUNS=ROOT/'runs/pidg_transfer_integrity'
REPORT=ROOT/'experiments/transfer_integrity'
SOURCE=ROOT/'runs/pidg_color_repair/C1_repaired/rubber_duck'
DATA=ROOT/'data/derived/kinematic_resplit/training'
NAMES=['U0_C1_control','U1_transfer_geometry']


def config(on,binding):
    plan=engine.read_json(REPORT/'plan.json')
    c=engine.read_json(SOURCE/'metric_config.json')
    c.update(plan['terms'])
    c.update(transfer_integrity_enabled=on,transfer_data=binding['path'],transfer_data_sha256=binding['sha256'],
             transfer_source_sha256=binding['source_checkpoint_sha256'],transfer_interval=plan['interval'],
             transfer_ramp=plan['ramp_updates'],static_samples=plan['static_points_per_term'],
             track_samples=plan['tracked_points_per_term'],record_transfer_sampling=True)
    return c


def branch(name,target,binding,smoke=False):
    directory=RUNS/('_smoke' if smoke else 'pilot')/name/'rubber_duck'
    c=config(name==NAMES[1],binding)
    if (directory/'branch.json').exists():
        assert engine.read_json(directory/'metric_config.json')==c
        return directory
    directory.mkdir(parents=True,exist_ok=True)
    engine.write_json(directory/'metric_config.json',c);shutil.copyfile(SOURCE/'model_config.py',directory/'model_config.py')
    common=RUNS/'branch_states'/str(target)/'resume.pt'
    if not common.exists():
        original=torch.load(SOURCE/'checkpoints/resume.pt',map_location='cpu',weights_only=False)
        changed=dict(original,opt=dict(original['opt'],iterations=target+1),
                     testing_iterations=[target] if smoke else [50500,51000,51500,52000],saving_iterations=[target])
        common.parent.mkdir(parents=True,exist_ok=True);torch.save(changed,common)
        restored=torch.load(common,map_location='cpu',weights_only=False)
        assert all(tensor_equal(v,restored[k]) for k,v in original.items() if k not in ['opt','testing_iterations','saving_iterations'])
        assert {k for k in original['opt'] if original['opt'][k]!=restored['opt'][k]}=={'iterations'}
        engine.write_json(common.with_suffix('.proof.json'),{'source_sha256':binding['source_checkpoint_sha256'],
                         'modified_checkpoint_sha256':engine.sha256(common),'model_optimizer_RNG_camera_tensor_identical':True,
                         'changed_only':['opt.iterations','testing_iterations','saving_iterations']})
    checkpoint=directory/'checkpoints/resume.pt';checkpoint.parent.mkdir(exist_ok=True);shutil.copyfile(common,checkpoint)
    digest=engine.sha256(checkpoint);receipt=engine.read_json(SOURCE/'checkpoints/resume.json')
    receipt.update(path=str(checkpoint),sha256=digest,size_bytes=checkpoint.stat().st_size)
    engine.write_json(checkpoint.with_suffix('.json'),receipt)
    side=engine.read_json(SOURCE/'checkpoints/resume.metrics.json');side.update(config=c,checkpoint_sha256=digest,transfer_data_sha256=binding['sha256'])
    engine.write_json(checkpoint.with_suffix('.metrics.json'),side)
    engine.write_json(directory/'branch.json',{'source':str(SOURCE),'destination':str(directory),'iteration':50000,
                     'identical_full_checkpoint_sha256':digest,'source_checkpoint_sha256':binding['source_checkpoint_sha256']})
    engine.write_json(directory/'reproduction.json',{'id':name,'commands':[],'status':'initialized','target_iteration':target,'native_width':1600})
    return directory


def train(directory,repo,target,deadline,stop=None):
    command=[*engine.command_prefix(repo),'train_pidg.py','-s',str(DATA),'-m',str(directory),
             '--conf',str(directory/'model_config.py'),'--load2gpu_on_the_fly','--resolution','1600',
             '--save_image_dir',str(directory/'diagnostics'),'--iterations',str(target+1),'--checkpoint_interval','500',
             '--resume_checkpoint',str(directory/'checkpoints/resume.pt')]
    env=engine.env_for(directory);env['PIDG_TUNING_STOP_AFTER']=str(stop or target)
    return engine.execute(command,directory,'smoke' if target<50100 else 'long',env,deadline)


def freeze(repo):
    paths=[*Path(__file__).parent.glob('*.py'),ROOT/'tests/test_transfer_integrity.py',REPORT/'plan.json',REPORT/'data_binding.json',
           RUNS/'training_data.npz',SOURCE/'checkpoints/resume.pt',SOURCE/'metric_config.json',SOURCE/'model_config.py']
    own={str(p.relative_to(ROOT)):engine.sha256(p) for p in paths}
    inputs=engine.read_json(REPORT/'input_lock.json')
    assert all(engine.sha256(ROOT/p)==h for p,h in inputs.items()),'Prepared training inputs differ'
    value={'own':own,'input':inputs,'runtime':engine.runtime_hashes(repo)}
    p=REPORT/'training_freeze.json'
    if p.exists():assert engine.read_json(p)==value,'Frozen training source/input changed'
    else:engine.write_json(p,value)
    return value


def preflight(repo,binding,deadline):
    if (REPORT/'preflight.json').exists():
        assert engine.read_json(REPORT/'preflight.json')['status']=='passed'
        return
    import subprocess
    env=engine.env_for(RUNS)
    process=subprocess.run([sys.executable,'-u','-m','variants.transfer_integrity.probe'],cwd=ROOT,env=env,capture_output=True,text=True)
    (REPORT/'optimizer_probe.log').write_text(process.stdout+process.stderr,encoding='utf-8')
    assert process.returncode==0,'Real gradient/optimizer probe failed'
    off=branch(NAMES[0],50010,binding,True)
    reference=branch('C1_original_path',50010,binding,True)
    repeat=branch('C1_original_repeat',50010,binding,True)
    for directory,code in [(off,repo),(reference,ROOT/'variants/color_repair/code'),(repeat,ROOT/'variants/color_repair/code')]:
        completed=engine.read_json(directory/'checkpoints/resume.json')['iteration']
        if completed<50010:assert train(directory,code,50010,deadline)==75
        assert engine.read_json(directory/'checkpoints/resume.json')['iteration']==50010
    a=torch.load(off/'checkpoints/resume.pt',map_location='cpu',weights_only=False)
    b=torch.load(reference/'checkpoints/resume.pt',map_location='cpu',weights_only=False)
    c=torch.load(repeat/'checkpoints/resume.pt',map_location='cpu',weights_only=False)
    parity={};repeat_noise={}
    for key in ['gaussian_parameters','gaussian_buffers','velocity_net','grid4d','deform']:
        def stats(candidate):
            delta=torch.cat([(v.double()-b[key][k].double()).flatten() for k,v in candidate[key].items() if torch.is_tensor(v) and v.numel()])
            return {'max_abs':float(delta.abs().max()),'RMS':float(delta.square().mean().sqrt())}
        parity[key]=stats(a);repeat_noise[key]=stats(c)
        # Hash-grid/CUDA backward atomics are not bitwise deterministic. Compare
        # to an actual repeat of the unchanged C1 path, without hiding raw error.
        assert parity[key]['RMS']<=3*repeat_noise[key]['RMS']+1e-7,(key,parity[key],repeat_noise[key])
    for key in ['rng_python','rng_numpy','rng_torch','rng_cuda','viewpoint_stack']:
        assert tensor_equal(a[key],b[key]) and tensor_equal(c[key],b[key]),key
    def validation(directory):
        import json,re
        lines=re.findall(r'VAL_METRICS (\{[^\n]+)',(directory/'train.log').read_text())
        return json.JSONDecoder().raw_decode(lines[-1])[0]['metrics']
    metrics=[validation(directory) for directory in [off,reference,repeat]]
    tolerance={'PSNR':1e-4,'SSIM':1e-6,'LPIPS':1e-6,'MS-SSIM':1e-6}
    assert all(abs(metrics[0][k]-metrics[1][k])<=v and abs(metrics[2][k]-metrics[1][k])<=v for k,v in tolerance.items())
    on=branch(NAMES[1],50020,binding,True)
    completed=engine.read_json(on/'checkpoints/resume.json')['iteration']
    if completed<50010:assert train(on,repo,50020,deadline,50010)==75
    if engine.read_json(on/'checkpoints/resume.json')['iteration']<50020:assert train(on,repo,50020,deadline)==75
    assert 'FULL_CHECKPOINT_RESTORED completed=50010 next=50011' in (on/'train.log').read_text()
    assert len((on/'transfer_loss.jsonl').read_text().splitlines())==2
    assert not (off/'transfer_loss.jsonl').exists()
    engine.write_json(REPORT/'preflight.json',{'status':'passed','flags_off_steps':10,'on_steps':20,'resume_boundary':[50010,50020],
                     'original_C1_path_parameter_difference':parity,'original_path_repeat_difference':repeat_noise,
                     'numerical_gate':'RMS <= 3x independently repeated unchanged-C1 difference + 1e-7; exact RNG/sampler; tight validation tolerance',
                     'validation_metrics_off_original_repeat':metrics,'validation_tolerance':tolerance,
                     'native_width':1600,'actual_gradient_optimizer_probe':True,
                     'binding_sha256':binding['sha256']})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--preflight-only',action='store_true');parser.add_argument('--hours',type=float,default=4)
    args=parser.parse_args();torch.set_num_threads(4)
    RUNS.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    repo=prepare_code();binding=prepare_data()
    engine.RUNS,engine.REPORTS,engine.repo_global=RUNS,REPORT,repo
    deadline=datetime.now(timezone.utc)+timedelta(hours=args.hours)
    state={'status':'preflight','pid':os.getpid(),'started_utc':engine.now()}
    engine.write_json(RUNS/'controller.json',state)
    try:
        preflight(repo,binding,deadline)
        if args.preflight_only:
            state['status']='preflight_passed';return
        freeze(repo)
        for name in NAMES:branch(name,52000,binding)
        a,b=[RUNS/'pilot'/n/'rubber_duck' for n in NAMES]
        ca,cb=[engine.read_json(d/'metric_config.json') for d in [a,b]]
        assert {k for k in ca if ca[k]!=cb[k]}=={'transfer_integrity_enabled'}
        state['status']='training';engine.write_json(RUNS/'controller.json',state)
        for d in [a,b]:
            if engine.read_json(d/'checkpoints/resume.json')['iteration']<52000:assert train(d,repo,52000,deadline)==75
            assert engine.read_json(d/'checkpoints/resume.json')['iteration']==52000
        sampled=[[engine.json.loads(line) for line in (d/'camera_sampling.jsonl').read_text().splitlines()] for d in [a,b]]
        assert sampled[0]==sampled[1] and len(sampled[0])==2000
        freeze(repo);state['status']='trained_pending_evaluation'
    except Exception as error:
        state.update(status='failed',error=str(error));raise
    finally:
        state['updated_utc']=engine.now();engine.write_json(RUNS/'controller.json',state)


if __name__=='__main__':main()
