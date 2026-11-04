"""Wait for this one pipeline, then evaluate/export once; never launch training."""
import subprocess
import sys
import time
from variants.staged_reconstruction.rubber_joint import runtime as r
from variants.staged_reconstruction.prepare_data import ROOT,read,sha

def main():
    with r.lock('postprocess.lock'):
        while True:
            if r.stopped():return
            state=read(r.RUNS/'pipeline.json')
            if state['status']=='trained_pending_evaluation':break
            assert state['status']=='training',state
            time.sleep(5)
        for phase,name in [('evaluate','evaluation.json'),('video','videos.json')]:
            if r.stopped():return
            if (r.REPORT/name).exists():continue
            path=r.REPORT/f'postprocess_{phase}.log'
            with path.open('xb') as stream:
                code=subprocess.run([sys.executable,'-u',str(r.BASE/'evaluate.py'),
                    '--enable-rubber-joint-evaluation','--phase',phase],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT).returncode
            assert code==0,f'Preserve {path}'
        state=read(r.RUNS/'pipeline.json');state.update(status='evaluated_exported_pending_manual_review',pending_review=True)
        r.atomic_json(r.RUNS/'pipeline.json',state)
        r.atomic_json(r.REPORT/'postprocess.json',{'evaluation_and_video_complete':True,'manual_review_pending':True,
            'source_sha256':sha(r.BASE/'postprocess.py')})

if __name__=='__main__':main()
