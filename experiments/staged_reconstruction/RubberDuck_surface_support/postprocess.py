"""Serial, lock-protected read-only stages after the fixed matched training."""
from pathlib import Path
import argparse
import subprocess
import sys
from variants.staged_reconstruction.i10_surface_support import runtime as a
from variants.staged_reconstruction.i10_tracking.io import ROOT,read,write,sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--enable-postprocess',action='store_true')
    if not p.parse_args().enable_postprocess:raise SystemExit('Postprocess default off')
    r=a.install(True);assert not r.stopped()
    assert read(a.RUNS/'pipeline.json')['status']=='trained_pending_evaluation'
    assert not (a.REPORT/'evaluation.json').exists()
    stages=[('evaluation','evaluate.py',['--enable-surface-evaluation','--phase','evaluate']),
            ('controller','controller_audit.py',['--enable-controller-audit']),
            ('structure','structural_diagnostic.py',['--enable-structural-diagnostic']),
            ('video','evaluate.py',['--enable-surface-evaluation','--phase','video']),
            ('artifacts','audit.py',['--enable-audit'])]
    with r.lock('postprocess.lock'):
        state=dict(status='running',completed=[],source_sha256=sha(Path(__file__)))
        write(a.REPORT/'postprocess.json',state)
        try:
            for stage,filename,flags in stages:
                assert not r.stopped()
                log=a.REPORT/f'{stage}_attempt1.log';assert not log.exists()
                with log.open('wb') as stream:
                    subprocess.run([sys.executable,str(a.BASE/filename),*flags],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
                state['completed'].append(dict(stage=stage,log=log.relative_to(ROOT).as_posix(),sha256=sha(log)))
                write(a.REPORT/'postprocess.json',state)
                print('L5_POSTPROCESS_STAGE_COMPLETE',stage,flush=True)
        except Exception as error:
            state.update(status='failed',error=repr(error));write(a.REPORT/'postprocess.json',state);raise
        state.update(status='completed_pending_manual_review');write(a.REPORT/'postprocess.json',state)

if __name__=='__main__':main()
