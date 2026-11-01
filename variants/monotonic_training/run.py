"""Complete H3+monotonic-clock joint training, preflight, locked tests and export."""
import argparse
from datetime import datetime,timedelta,timezone
import html
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from variants.metric_tuning import run as engine
from variants.kinematic_resplit import run as base
from .prepare import prepare

HERE=Path(__file__).parent;REPORTS=ROOT/'experiments/monotonic_training';RUNS=ROOT/'runs/pidg_monotonic_training'
NAME='J1_monotonic_kinematic';PREFIX=ROOT/'runs/pidg_color_repair/shared_prefix/rubber_duck'
REFERENCE=ROOT/'runs/pidg_integrity_tuning/H3_combined/rubber_duck'


def config(smoke=False,off=False):
    c=engine.read_json(REFERENCE/'metric_config.json')
    plan=engine.read_json(REPORTS/'plan.json')
    for k in ('velocity_clock_mode','kinematic_weight','kinematic_start','kinematic_ramp','kinematic_interval','kinematic_samples','kinematic_dt','kinematic_huber_delta'):c[k]=plan[k]
    text=(PREFIX/'model_config.py').read_text(encoding='utf8')
    if smoke:
        c.update(engineering_smoke=True,kinematic_start=20,kinematic_ramp=10,kinematic_interval=5,kinematic_samples=64,
          integrity_coverage_start=20,integrity_strain_start=30,integrity_ramp=10,integrity_strain_interval=5,
          integrity_screen_iteration=100,mask_start=20,mask_ramp=10,audit_gradients=True,audit_parameters=True)
        text+='\nwarm_up = 0\ndensify_until_iter = 0\n'
    if off:c.update(enabled=False,kinematic_weight=0.,velocity_clock_mode='legacy',integrity_enabled=False)
    return c,text


def initialize(smoke=False,off=False):
    d=RUNS/('_smoke' if smoke else '')/('flags_off' if off else NAME)/'rubber_duck';c,text=config(smoke,off)
    d.mkdir(parents=True,exist_ok=True)
    if (d/'metric_config.json').exists():
        if engine.read_json(d/'metric_config.json')!=c or (d/'model_config.py').read_text(encoding='utf8')!=text:raise ValueError('Existing configuration changed')
        return d
    engine.write_json(d/'metric_config.json',c);engine.write_text_atomic(d/'model_config.py',text)
    engine.write_json(d/'reproduction.json',{'id':NAME if not off else 'flags_off','commands':[],'status':'initialized',
      'train':120,'validation':30,'test':50,'target_iterations':20 if off else 100 if smoke else 50000,
      'resolution_width':400 if smoke else 1600,'seed':0,'selection':'validation only','shared_prefix':str(PREFIX) if not smoke else None})
    return d


def freeze(repo):
    helpers=[]
    for name in ['metric_tuning','kinematic_tuning','kinematic_resplit','v3_isotropic','v3_iso_scratch','coverage_tuning','color_repair','integrity_tuning','monotonic_training']:
        helpers+=list((ROOT/'variants'/name).glob('*.py'))
    helpers+=[ROOT/'tests/test_monotonic_training.py',REPORTS/'plan.json',ROOT/'scripts/pidg_variant_entrypoint.py',ROOT/'reproduction/scripts/training_checkpoint.py']
    helpers+=[PREFIX/p for p in ['model_config.py','metric_config.json','checkpoints/resume.json']]
    helpers+=[base.DATA/'training'/p for p in ['transforms_train.json','transforms_test.json','loader_role.json','points3d.ply']]
    return {'code':engine.runtime_hashes(repo),'inherited_code':engine.runtime_hashes(ROOT/'variants/integrity_tuning/code'),
      'helpers_inputs':{str(p.relative_to(ROOT)):engine.sha256(p) for p in helpers}}


def assert_frozen(repo):
    if engine.read_json(REPORTS/'source_freeze.json')!=freeze(repo):raise ValueError('Frozen code or training data changed')


def wire(repo):
    engine.RUNS,engine.REPORTS,engine.repo_global,engine.assert_frozen=RUNS,REPORTS,repo,assert_frozen
    base.RUNS,base.REPORTS,base.assert_frozen=RUNS,REPORTS,assert_frozen


def step(d):
    return engine.read_json(d/'checkpoints/resume.json')['iteration'] if (d/'checkpoints/resume.json').exists() else 0


def branch(d):
    receipt=d/'branch.json'
    source_manifest=engine.read_json(PREFIX/'checkpoints/resume.json')
    if receipt.exists():
        if engine.read_json(receipt)['identical_full_checkpoint_sha256']!=source_manifest['sha256']:raise ValueError('Prefix changed')
        return
    if (d/'checkpoints/resume.pt').exists():raise ValueError('Unreceipted checkpoint')
    if source_manifest['iteration']!=3000 or engine.sha256(PREFIX/'checkpoints/resume.pt')!=source_manifest['sha256']:raise ValueError('Invalid shared3k state')
    if (PREFIX/'model_config.py').read_text(encoding='utf8')!=(d/'model_config.py').read_text(encoding='utf8'):raise ValueError('Optimization recipe changed')
    for folder in ['point_cloud','deform','checkpoints']:shutil.copytree(PREFIX/folder,d/folder,dirs_exist_ok=True)
    for name in ['metric_best.json','metric_history.jsonl']:shutil.copyfile(PREFIX/name,d/name)
    source_config=engine.read_json(PREFIX/'metric_config.json');c=engine.read_json(d/'metric_config.json')
    side=d/'checkpoints/resume.metrics.json';state=engine.read_json(side)
    if state['config']!=source_config:raise ValueError('Wrong source sidecar')
    state['config']=c;engine.write_json(side,state)
    engine.write_json(receipt,{'source':str(PREFIX),'destination':str(d),'boundary':3000,
      'identical_full_checkpoint_sha256':source_manifest['sha256'],'source_config':source_config,'branch_config':c,
      'changed_flags':{k:[source_config.get(k),c.get(k)] for k in set(source_config)|set(c) if source_config.get(k)!=c.get(k)},
      'tensor_optimizer_rng_unchanged_at_branch':True})


def preflight(repo,deadline):
    receipt=REPORTS/'preflight.json'
    if receipt.exists() and engine.read_json(receipt).get('status')=='passed':return
    env=engine.env_for(RUNS);env['PIDG_MONOTONIC_GPU_TEST']='1'
    check=subprocess.run([sys.executable,'-m','pytest','tests/test_monotonic_training.py','tests/test_kinematic_tuning.py','-q','--basetemp','tmp/monotonic_gpu_pytest'],cwd=ROOT,env=env,capture_output=True)
    (REPORTS/'tests.log').write_bytes(check.stdout+check.stderr)
    if check.returncode:raise ValueError('Unit/GPU physics checks failed')
    d=initialize(True)
    if step(d)<70:
        if base.train(d,repo,deadline,True,limit=70)!=75:raise ValueError('Smoke70 save boundary failed')
    if step(d)<100:
        if base.train(d,repo,deadline,True)!=0:raise ValueError('Smoke100 failed')
    if b'FULL_CHECKPOINT_RESTORED completed=70 next=71' not in (d/'train.log').read_bytes():raise ValueError('Resume not exercised')
    gradients=[json.loads(s) for s in (d/'kinematic_parameter_gradient.jsonl').read_text().splitlines()]
    if {g['branch'] for g in gradients}!={'grid','deformation','velocity'} or any(not g['finite'] or g['gradient_l1']<=0 for g in gradients):raise ValueError('Joint gradient audit failed')
    off=initialize(True,True)
    if step(off)<20 and base.train(off,repo,deadline,True,off=True)!=0:raise ValueError('Flags-off smoke failed')
    manifest=engine.read_json(HERE/'source_manifest.json')
    changed=sorted(k for k in manifest['parent'] if manifest['parent'][k]!=manifest['fork'][k])
    if changed!=['motion_utils/pidg.py','scene_PIDG/gaussian_model.py','train_pidg.py']:raise ValueError('Unexpected PIDG changes')
    engine.write_json(receipt,{'status':'passed','smoke_steps':100,'resumed_from':70,'flags_off_smoke_steps':20,
      'legacy_forward_and_gradient_parity':True,'gpu_momentum_second_derivatives':True,'joint_gradients':gradients,
      'renderer_dataset_and_camera_sampler_unchanged':True,'completed_utc':engine.now()})


def monitor(d,status,error=None):
    h=[json.loads(s) for s in (d/'metric_history.jsonl').read_text().splitlines()] if (d/'metric_history.jsonl').exists() else []
    s=engine.tail(d/'train.log',20000)
    matches=re.findall(r'Training progress[^\r\n]*?\|\s*(\d+)/(\d+)',s)
    iteration=max(step(d),int(matches[-1][0]) if matches else 0)
    value={'status':status,'iteration':iteration,'target':50000,'latest_validation':h[-1] if h else None,
      'run':str(d),'updated_utc':engine.now(),'error':error}
    engine.write_json(REPORTS/'progress.json',value)
    page='<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="10"><title>Monotonic clock training</title><style>body{font:18px system-ui;background:#112035;color:#edf3fb;margin:30px}pre{white-space:pre-wrap}progress{width:90%}</style>'
    page+=f'<h1>H3 + 单调时钟 + 速度—轨迹一致性</h1><h2>{html.escape(status)}：{iteration:,} / 50,000</h2><progress value="{iteration}" max="50000"></progress><p>120 train / 30 validation / 50 test；固定数据时间轴；UTC {value["updated_utc"]}</p><pre>'+html.escape(json.dumps(value['latest_validation'],ensure_ascii=False,indent=2))+'</pre>'
    if error:page+='<pre>'+html.escape(error)+'</pre>'
    page+='<a href="assessment.md">结果报告</a>'
    engine.write_text_atomic(REPORTS/'monitor.html',page)
    return value


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--preflight-only',action='store_true');parser.add_argument('--hours',type=float,default=12);args=parser.parse_args()
    RUNS.mkdir(exist_ok=True);REPORTS.mkdir(exist_ok=True)
    repo=prepare();wire(repo);frozen=freeze(repo)
    if (REPORTS/'source_freeze.json').exists():assert_frozen(repo)
    else:engine.write_json(REPORTS/'source_freeze.json',frozen)
    deadline=datetime.now(timezone.utc)+timedelta(hours=args.hours)
    d=initialize();monitor(d,'preflight')
    try:
        preflight(repo,deadline)
        if args.preflight_only:monitor(d,'preflight_passed');return
        branch(d);monitor(d,'training')
        # Execute waits in its own coordinator process. Status is refreshed by the
        # desktop monitor or the lightweight reporting thread below.
        import threading
        stop=threading.Event()
        def refresh():
            while not stop.wait(10):monitor(d,'training')
        thread=threading.Thread(target=refresh,daemon=True);thread.start()
        try:
            if step(d)<50000 and base.train(d,repo,deadline)!=0:raise RuntimeError('Training saved before completion')
        finally:stop.set();thread.join()
        if step(d)!=50000:raise ValueError('Full training not completed')
        best=engine.read_json(d/'metric_best.json')['PSNR']
        engine.write_json(RUNS/'selection_lock.json',{'locked_utc':engine.now(),'selection_split':'validation',
          'best':best,'final_iteration':50000,'test_opened_before_lock':False})
        monitor(d,'rendering')
        base.render(d,repo,deadline,split='test',tags=['best','50000'])
        from .export import export
        export(d)
        monitor(d,'completed')
        print('COMPLETE '+str(REPORTS/'assessment.md'),flush=True)
    except Exception as e:
        monitor(d,'failed',repr(e));raise


if __name__=='__main__':main()
