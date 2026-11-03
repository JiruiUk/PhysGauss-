"""Continue the authorized current run through evaluation without scheduled jobs."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import sys
import time
import subprocess
import psutil
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from variants.metric_tuning.runtime import read_json,write_json
REPORT=ROOT/'experiments/scratch_integrity';RUNS=ROOT/'runs/pidg_scratch_integrity'


def main():
    deadline=datetime.now(timezone.utc)+timedelta(hours=12)
    state={'status':'waiting_training','pid':psutil.Process().pid,'started_utc':datetime.now(timezone.utc).isoformat()}
    write_json(RUNS/'pipeline.json',state)
    try:
        missing_since=None
        while True:
            training=read_json(RUNS/'controller.json')
            if training['status']=='trained_pending_evaluation':break
            if training['status']=='failed':raise RuntimeError(training.get('error','Training failed'))
            running=[]
            for p in psutil.process_iter(['pid','cmdline']):
                try:
                    if 'variants.scratch_integrity.run' in (p.info['cmdline'] or []):running.append(p.info['pid'])
                except (psutil.NoSuchProcess,psutil.AccessDenied):pass
            if not running:
                missing_since=missing_since or time.monotonic()
                if time.monotonic()-missing_since>60:raise RuntimeError('Training coordinator stopped before evaluation')
            else:missing_since=None
            assert datetime.now(timezone.utc)<deadline,'Pipeline deadline reached'
            time.sleep(5)
        for stage in ['evaluate','support','export']:
            log=REPORT/f'{stage}.log'
            with log.open('w',encoding='utf-8') as output:
                process=subprocess.Popen([sys.executable,'-u',str(REPORT/f'{stage}.py')],cwd=ROOT,stdout=output,stderr=subprocess.STDOUT)
                state.update(status=stage,child_pid=process.pid);write_json(RUNS/'pipeline.json',state)
                assert process.wait()==0,f'{stage} failed: {log}'
        state['status']='evaluated_pending_review'
    except Exception as error:
        state.update(status='failed',error=str(error));raise
    finally:
        state['updated_utc']=datetime.now(timezone.utc).isoformat();write_json(RUNS/'pipeline.json',state)


if __name__=='__main__':main()
