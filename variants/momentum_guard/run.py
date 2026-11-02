"""One controlled gradient-policy experiment with complete recovery/export."""
import argparse
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import subprocess
import sys
import threading
from variants.metric_tuning import run as engine
from variants.kinematic_resplit import run as base
from variants.monotonic_training import run as previous
from variants.unified_flow import run as parent
from .prepare import prepare

ROOT=Path(__file__).resolve().parents[2]; HERE=Path(__file__).parent
REPORTS=ROOT/'experiments/momentum_repair'; RUNS=ROOT/'runs/pidg_momentum_guard'
NAME='M1_guarded_shared'


def config(smoke=False, off=False):
    c,text=parent.config(parent.NAMES[1],smoke,off)
    c.update(momentum_gradient_guard=not off,momentum_shared_ratio=.1)
    return c,text


def initialize(smoke=False,off=False):
    d=RUNS/('_smoke' if smoke else '')/('flags_off' if off else NAME)/'rubber_duck'
    c,text=config(smoke,off); d.mkdir(parents=True,exist_ok=True)
    if (d/'metric_config.json').exists():
        assert engine.read_json(d/'metric_config.json')==c
        assert (d/'model_config.py').read_text(encoding='utf8')==text
    else:
        engine.write_json(d/'metric_config.json',c); engine.write_text_atomic(d/'model_config.py',text)
        engine.write_json(d/'reproduction.json',{'id':NAME,'status':'initialized','commands':[],
            'train':120,'validation':30,'test':50,'seed':0,'selection':'validation only',
            'target_iterations':20 if off else 100 if smoke else 50000})
    return d


def freeze(repo):
    parent.assert_frozen(ROOT/'variants/unified_flow/code')
    files=[*HERE.glob('*.py'),ROOT/'tests/test_momentum_guard.py',REPORTS/'plan.json',
        REPORTS/'diagnosis.json',ROOT/'output/unified_flow/diagnose_repair.py',HERE/'source_manifest.json']
    return {'parent':parent.freeze(ROOT/'variants/unified_flow/code'),'code':engine.runtime_hashes(repo),
        'helpers':{str(p.relative_to(ROOT)):engine.sha256(p) for p in files}}


def assert_frozen(repo):
    if engine.read_json(REPORTS/'source_freeze.json')!=freeze(repo):
        raise ValueError('Frozen momentum experiment changed')


def wire(repo):
    engine.RUNS,engine.REPORTS,engine.repo_global,engine.assert_frozen=RUNS,REPORTS,repo,assert_frozen
    base.RUNS,base.REPORTS,base.assert_frozen=RUNS,REPORTS,assert_frozen


def preflight(repo,deadline):
    receipt=REPORTS/'preflight.json'
    if receipt.exists() and engine.read_json(receipt).get('status')=='passed': return
    env=engine.env_for(RUNS); env['PIDG_UNIFIED_GPU_TEST']='1'
    with (REPORTS/'tests.log').open('wb') as log:
        p=subprocess.run([sys.executable,'-m','pytest','tests/test_momentum_guard.py',
            'tests/test_unified_branch.py','tests/test_unified_flow_geometry.py','-q',
            '--basetemp','tmp/momentum_guard_gpu_pytest'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    if p.returncode: raise RuntimeError('Unit/CUDA checks failed')
    d=initialize(True)
    if previous.step(d)<70 and base.train(d,repo,deadline,True,limit=70)!=75: raise ValueError('70-step save failed')
    if previous.step(d)<100 and base.train(d,repo,deadline,True)!=0: raise ValueError('100-step smoke failed')
    assert b'FULL_CHECKPOINT_RESTORED completed=70 next=71' in (d/'train.log').read_bytes()
    guards=[__import__('json').loads(s) for s in (d/'momentum_gradient_guard.jsonl').read_text().splitlines()]
    assert len(guards)==1 and guards[0]['iteration']==100
    for row in guards[0]['groups'].values():
        assert row['guarded_momentum_norm']<=.1*row['image_norm']*(1+1e-5)
    off=initialize(True,True)
    if previous.step(off)<20 and base.train(off,repo,deadline,True,off=True)!=0: raise ValueError('Flags-off failed')
    assert not (off/'momentum_gradient_guard.jsonl').exists()
    with (REPORTS/'optimizer_update.log').open('wb') as log:
        p=subprocess.run([sys.executable,'-u','-m','variants.momentum_guard.verify_update'],
            cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    if p.returncode: raise ValueError('Actual optimizer / gradient parity check failed')
    engine.write_json(receipt,{'status':'passed','smoke_steps':100,'complete_resume':{'saved':70,'next':71},
        'flags_off_steps':20,'actual_optimizer_update':engine.read_json(REPORTS/'optimizer_update.json'),
        'guard':guards,'completed_utc':engine.now()})


def monitor(status,error=None):
    d=RUNS/NAME/'rubber_duck'; h=base.history(d) if d.exists() else []
    state={'status':status,'model':NAME,'iteration':base.progress(d) if d.exists() else 0,
        'latest_validation':h[-1] if h else None,'error':error,'updated_utc':engine.now()}
    engine.write_json(REPORTS/'progress.json',state)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--preflight-only',action='store_true')
    parser.add_argument('--hours',type=float,default=12);args=parser.parse_args()
    REPORTS.mkdir(exist_ok=True);RUNS.mkdir(exist_ok=True)
    # Must precede every architecture import, including evaluation in coordinator.
    os.environ['PIDG_HASH_SOURCE_DIR']=str(ROOT/'third_party/PIDG/hashencoder')
    os.environ['PIDG_HASH_BUILD_DIR']=str(ROOT/'third_party/PIDG/tmp_build')
    from variants.unified_flow.prepare_data import prepare as prepare_data
    prepare_data();repo=prepare();wire(repo)
    if (REPORTS/'source_freeze.json').exists(): assert_frozen(repo)
    else: engine.write_json(REPORTS/'source_freeze.json',freeze(repo))
    deadline=datetime.now(timezone.utc)+timedelta(hours=args.hours)
    try:
        monitor('preflight');preflight(repo,deadline)
        if args.preflight_only: monitor('preflight_passed');return
        d=initialize();parent.branch(d)
        assert engine.sha256(d/'checkpoints/resume.pt')==engine.read_json(d/'branch.json')['identical_full_checkpoint_sha256'] if previous.step(d)==3000 else True
        stop=threading.Event()
        def refresh():
            while not stop.wait(10): monitor('training')
        t=threading.Thread(target=refresh,daemon=True);t.start()
        try:
            monitor('training')
            if previous.step(d)<50000 and base.train(d,repo,deadline)!=0: raise RuntimeError('Saved before completion')
            if previous.step(d)!=50000: raise ValueError('Incomplete full run')
        finally: stop.set();t.join()
        engine.write_json(RUNS/'selection_lock.json',{'locked_utc':engine.now(),'selection_split':'validation',
            'models':{NAME:engine.read_json(d/'metric_best.json')['PSNR']},
            'test_opened_before_lock':False,'historical_test_exposure':True})
        from .export import export,report
        monitor('rendering');base.render(d,repo,deadline,split='test',tags=['best','50000']);export(d)
        monitor('diagnosing');report();assert_frozen(repo);monitor('completed')
    except Exception as error: monitor('failed',repr(error));raise


if __name__=='__main__':main()
