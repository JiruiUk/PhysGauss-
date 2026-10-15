"""Sequential, source-frozen matrix worker with an exclusive controller lock."""
import argparse
from contextlib import contextmanager
from datetime import datetime,timezone
import os
from pathlib import Path
import shutil
import sys
from pidg_benchmark import PROJECT,read_json,write_json,sha256,runtime_hashes
from run_constitutive_experiment import run,execute,RUNS
from report_constitutive import matrix,report,tag_for


def jobs(plan,mode):
    key='selected_long_configs' if mode=='long' else 'controlled_smoke_configs'
    ids=plan['execution'][key]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate matrix configurations')
    output=[]
    for name in ids:
        cfg=read_json(PROJECT/f'experiments/configs/{name}.json')
        if mode=='long' and cfg['scene'] not in plan['execution']['selected_long_scenes']:
            raise ValueError('Configuration outside selected benchmark scope')
        output.append({'configuration_id':name,'scene':cfg['scene'],'config_hash':sha256(PROJECT/f'experiments/configs/{name}.json')})
    return output


@contextmanager
def controller_lock():
    RUNS.mkdir(parents=True,exist_ok=True)
    with (RUNS/'controller.lock').open('a+b') as stream:
        stream.seek(0)
        if not stream.read(1):stream.write(b'0');stream.flush()
        stream.seek(0);locked=False
        try:
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
            locked=True;yield
        finally:
            if locked:
                stream.seek(0)
                if os.name=='nt':msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
                else:fcntl.flock(stream,fcntl.LOCK_UN)


def work(mode,resume=False):
    if (RUNS/'stop_queue.request').exists():raise RuntimeError('Saved stop request present; use start script -Resume to acknowledge')
    plan=matrix();queue=jobs(plan,mode)
    frozen={p.name:sha256(p) for p in (PROJECT/'pidg_ext').glob('*.py')}
    source=runtime_hashes(PROJECT/'variants/constitutive_pidg/code')
    controller={'pid':os.getpid(),'mode':mode,'status':'running','jobs':queue,'completed_jobs':[],
                'selected_scenes':plan['execution']['selected_long_scenes'],'matrix_hash':sha256(PROJECT/'experiments/pidg_experiment_matrix.yaml'),
                'started_utc':datetime.now(timezone.utc).isoformat()}
    write_json(RUNS/'controller.json',controller)
    try:
        for job in queue:
            if (RUNS/'stop_queue.request').exists():controller['status']='saved_stopped';return
            if shutil.disk_usage(PROJECT).free<15*1024**3:raise RuntimeError('Less than 15 GiB free')
            assert sha256(PROJECT/'experiments/pidg_experiment_matrix.yaml')==controller['matrix_hash'], 'Matrix changed during execution'
            assert {p.name:sha256(p) for p in (PROJECT/'pidg_ext').glob('*.py')}==frozen
            assert runtime_hashes(PROJECT/'variants/constitutive_pidg/code')==source
            assert sha256(PROJECT/f"experiments/configs/{job['configuration_id']}.json")==job['config_hash']
            controller['current_configuration']=job['configuration_id'];write_json(RUNS/'controller.json',controller);report(plan)
            run_path=run(job['configuration_id'],tag_for(job['configuration_id'],mode),50000 if mode=='long' else 300,mode=='long',resume)
            if mode=='long' and job['configuration_id']=='A4_single' and not (run_path/'identifiability.json').exists():
                info=read_json(run_path/'reproduction.json');info['status']='diagnosing'
                env=os.environ.copy();env['PYTHONPATH']=str(PROJECT)+os.pathsep+env.get('PYTHONPATH','')
                command=[sys.executable,'-u',str(PROJECT/'scripts/diagnose_constitutive.py'),'--run',str(run_path)]
                if execute(command,run_path,'diagnostics',PROJECT,env,info):raise RuntimeError('A5 post-fit diagnostics failed')
                info['status']='completed';info['a5_training_reuse']=True;write_json(run_path/'reproduction.json',info)
            controller['completed_jobs'].append(job['configuration_id']);write_json(RUNS/'controller.json',controller);report(plan)
        controller['status']='completed'
    except SystemExit as error:
        if error.code==75:controller['status']='saved_stopped'
        else:controller.update(status='failed',error=str(error));raise
    except Exception as error:
        controller.update(status='failed',error=str(error));raise
    finally:
        controller['updated_utc']=datetime.now(timezone.utc).isoformat();write_json(RUNS/'controller.json',controller);report(plan)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['smoke','long'],default='long');parser.add_argument('--resume',action='store_true');parser.add_argument('--report-only',action='store_true')
    args=parser.parse_args()
    if args.report_only:report()
    else:
        with controller_lock():work(args.mode,args.resume)
