"""Three independent processes: complete4 versus2+2 inference save/reload."""
import argparse
import subprocess
import sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from ..prepare_data import ROOT,read,write,sha
from .prepare import BASE,REPORT,RUNS,verify,stopped
from .controls import measure_controls


def main():
    p=argparse.ArgumentParser(); p.add_argument('--enable-plane-transport-preflight',action='store_true'); args=p.parse_args()
    assert args.enable_plane_transport_preflight and not stopped(); verify()
    assert not (REPORT/'preflight.json').exists()
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'); commands=[]
    for name,first,last in [('full',0,4),('split',0,2),('split',2,4)]:
        assert not stopped()
        command=[sys.executable,'-u','-m','variants.staged_reconstruction.plane_transport.infer',
                 '--enable-plane-transport','--preflight-directory',name,'--first',str(first),'--last',str(last)]
        log=REPORT/f'{stamp}_preflight_{name}_{first}_{last}.log'
        with log.open('x') as stream: result=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        assert result.returncode==0,str(log); commands.append({'command':command,'log':log.relative_to(ROOT).as_posix(),'sha256':sha(log)})
    checked={}
    for case in ['diffuse','glossy']:
        for i in range(4):
            full=RUNS/'preflight/full'/case/f'view_{i:03d}.npz'; split=RUNS/'preflight/split'/case/full.name
            with np.load(full) as a,np.load(split) as b:
                assert a.files==b.files
                for name in a.files: assert np.array_equal(a[name],b[name],equal_nan=True),(case,i,name)
                count=len(a['original'])
            assert read(full.with_suffix('.json'))['bindings']==read(split.with_suffix('.json'))['bindings']
            assert read(full.with_suffix('.json'))['counts']==read(split.with_suffix('.json'))['counts']
            checked[f'{case}/{i}']={'full_sha256':sha(full),'resumed_sha256':sha(split),'all_arrays_exact':True,'candidates':count}
    controls=measure_controls(read(BASE/'protocol.json'))
    verify(); write(REPORT/'analytic_controls.json',{'controls':controls,'diagnostic_only_no_candidate_selection':True})
    write(REPORT/'preflight.json',{'passed':True,'separate_processes':3,'both_conditions':True,'views_per_condition':4,
        'complete_slope_cost_acceptance_and_depth_recovery':True,'checked':checked,'commands':commands,
        'GT_read':False,'Gaussian_Adam_RNG_camera_recovery_claim':False,'source_freeze_sha256':sha(REPORT/'source_freeze.json'),
        'controls_sha256':sha(REPORT/'analytic_controls.json')})
    print('PLANE_TRANSPORT_PREFLIGHT_PASSED',flush=True)


if __name__=='__main__': main()
