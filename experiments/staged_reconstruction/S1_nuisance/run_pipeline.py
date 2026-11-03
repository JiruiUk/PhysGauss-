"""Serialize the four predeclared synthetic arms, audits and evaluation."""
import argparse
from datetime import datetime,timezone
import msvcrt
import os
from pathlib import Path
import subprocess
import sys
from variants.staged_reconstruction.nuisance_static.prepare import ROOT,REPORT,RUNS,verify
from variants.staged_reconstruction.prepare_data import read,write,sha


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--enable-nuisance',action='store_true'); args = parser.parse_args()
    if not args.enable_nuisance: raise SystemExit('Nuisance synthetic pipeline is opt-in')
    assert not (RUNS/'stop.request').exists(), 'Saved user stop; do not resume'
    verify(); assert read(REPORT/'smoke_verification.json')['passed']
    lock = (RUNS/'pipeline.lock').open('a+b'); lock.seek(0)
    if not lock.read(1): lock.write(b'0'); lock.flush()
    lock.seek(0); msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    attempt = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'); stage = 'preflight'
    def run(label,module,arguments):
        assert not (RUNS/'stop.request').exists(), 'Saved user stop; do not resume'
        path = REPORT/f'{attempt}_{label}.log'
        with path.open('wb') as log:
            child = subprocess.Popen([sys.executable,'-u','-m',module,*arguments],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW)
            write(RUNS/'pipeline.json',{'status':'running','stage':label,'pid':os.getpid(),'child_pid':child.pid,
                'log':path.relative_to(ROOT).as_posix(),'attempt':attempt,'orchestrator_sha256':sha(Path(__file__))})
            assert child.wait() == 0,f'{label} failed; preserved {path}'
    try:
        protocol = read(REPORT.parent/'protocol.json')
        for case in protocol['synthetic_cases']:
            for arm in protocol['arms']:
                stage = case+'_'+arm
                if not (REPORT/f'formal_{case}_{arm}_audit.json').exists():
                    run(stage,'variants.staged_reconstruction.nuisance_static.train',['--enable-nuisance','--case',case,'--arm',arm])
                run(stage+'_audit','variants.staged_reconstruction.nuisance_static.audit',['--case',case,'--arm',arm])
        stage = 'evaluation'
        if not (REPORT/'synthetic_evaluation.json').exists(): run(stage,'variants.staged_reconstruction.nuisance_static.evaluate',[])
        verify()
        write(RUNS/'pipeline.json',{'status':'synthetic_evaluated_pending_manual_review','stage':'complete',
            'attempt':attempt,'real_training_started':False,'orchestrator_sha256':sha(Path(__file__))})
    except Exception as error:
        write(RUNS/'pipeline.json',{'status':'saved_stopped' if (RUNS/'stop.request').exists() else 'failed',
            'stage':stage,'attempt':attempt,'error':str(error),'pid':os.getpid()}); raise
    finally:
        lock.seek(0); msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1); lock.close()


if __name__ == '__main__': main()
