"""Serialized opt-in orchestration; frozen training sources are never edited."""
import argparse
from datetime import datetime, timezone
import json
import msvcrt
import os
from pathlib import Path
import subprocess
import sys
from variants.staged_reconstruction.clean_static.prepare import ROOT, REPORT, RUNS, verify
from variants.staged_reconstruction.prepare_data import read, write, sha


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--enable-clean-static',action='store_true'); args = parser.parse_args()
    if not args.enable_clean_static: raise SystemExit('Clean static pipeline is opt-in')
    verify(); assert read(REPORT/'smoke_verification.json')['passed']
    assert not (RUNS/'stop.request').exists(), 'Saved user stop; do not resume'
    lock = (RUNS/'pipeline.lock').open('a+b'); lock.seek(0)
    if not lock.read(1): lock.write(b'0'); lock.flush()
    lock.seek(0); msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    attempt = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    stage = 'preflight'
    def run(label,module,arguments):
        assert not (RUNS/'stop.request').exists(), 'Saved user stop; do not resume'
        path = REPORT/f'{attempt}_{label}.log'
        with path.open('wb') as output:
            child = subprocess.Popen([sys.executable,'-u','-m',module,*arguments],cwd=ROOT,stdout=output,stderr=subprocess.STDOUT,
                                     creationflags=subprocess.CREATE_NO_WINDOW)
            write(RUNS/'pipeline.json',{'status':'running','stage':label,'pid':os.getpid(),'child_pid':child.pid,
                  'attempt':attempt,'log':path.relative_to(ROOT).as_posix(),'orchestrator_sha256':sha(Path(__file__))})
            code = child.wait()
        assert code == 0, f'{label} exit {code}; preserved {path}'
    try:
        for arm in ['control','visibility']:
            stage = arm
            audit_path = REPORT/f'formal_{arm}_audit.json'
            if not audit_path.exists():
                run(arm,'variants.staged_reconstruction.clean_static.train',['--enable-clean-static','--arm',arm])
            assert not (RUNS/'stop.request').exists(), 'Saved user stop; do not resume'
            run(arm+'_audit','variants.staged_reconstruction.clean_static.audit',['--arm',arm])
        stage = 'evaluation'
        if not (REPORT/'exports.json').exists(): run(stage,'variants.staged_reconstruction.clean_static.evaluate',[])
        verify(); assert read(REPORT/'exports.json')['passed']
        write(RUNS/'pipeline.json',{'status':'evaluated_pending_manual_review','stage':'complete','pid':os.getpid(),
              'attempt':attempt,'orchestrator_sha256':sha(Path(__file__))})
    except Exception as error:
        write(RUNS/'pipeline.json',{'status':'saved_stopped' if (RUNS/'stop.request').exists() else 'failed',
              'stage':stage,'attempt':attempt,'error':str(error),'pid':os.getpid()}); raise
    finally:
        lock.seek(0); msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1); lock.close()


if __name__ == '__main__': main()
