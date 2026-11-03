"""Finish a bounded R0 run and measurements; never silently launch another candidate."""
import os
from .run import ROOT, REPORT, RUNS, engine


def main():
    import argparse,subprocess,sys
    parser=argparse.ArgumentParser();parser.add_argument('--evaluate-only',action='store_true');args=parser.parse_args()
    state={'status':'training' if not args.evaluate_only else 'evaluating','pid':os.getpid(),'started_utc':engine.now()}
    engine.write_json(RUNS/'pipeline.json',state)
    modules=[] if args.evaluate_only else ['variants.static_pool.run']
    modules+=['variants.static_pool.evaluate','variants.static_pool.export','variants.static_pool.summarize']
    try:
        for module in modules:
            state.update(status=module,updated_utc=engine.now());engine.write_json(RUNS/'pipeline.json',state)
            name=module.rsplit('.',1)[-1]
            with (REPORT/(name+'_pipeline.log')).open('a',encoding='utf-8') as stream:
                process=subprocess.run([sys.executable,'-u','-m',module],cwd=ROOT,env=engine.env_for(RUNS),stdout=stream,stderr=subprocess.STDOUT)
            if process.returncode==75:
                state.update(status='saved_stopped');return
            assert process.returncode==0,f'{module} failed; inspect {name}_pipeline.log'
        state['status']='evaluated_pending_manual_review'
    except Exception as error:
        state.update(status='failed',error=str(error));raise
    finally:
        state['updated_utc']=engine.now();engine.write_json(RUNS/'pipeline.json',state)


if __name__=='__main__':
    # A resumed chat/heartbeat must not start a second training controller.
    import msvcrt
    RUNS.mkdir(parents=True,exist_ok=True)
    with (RUNS/'pipeline.lock').open('a+b') as lock:
        lock.seek(0)
        if not lock.read(1):lock.write(b'0');lock.flush()
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        try:main()
        finally:
            lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)
