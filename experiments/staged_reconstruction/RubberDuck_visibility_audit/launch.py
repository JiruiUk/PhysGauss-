"""Freeze and qualify one audit, then launch exactly one serial measurement."""
from pathlib import Path
import argparse
import subprocess
import sys
from variants.staged_reconstruction.rubber_visibility import worker as w
from variants.staged_reconstruction.prepare_data import ROOT,read,sha


def prepare():
    assert not w.stopped() and not (w.REPORT/'source_freeze.json').exists()
    assert read(w.REPORT/'unit_tests.json')['passed']
    files=read(ROOT/'experiments/staged_reconstruction/RubberDuck_joint_pilot/v1/source_freeze.json')
    for name,h in files.items():assert sha(ROOT/name)==h,name
    for row in read(w.BASE/'protocol.json')['models'].values():
        assert sha(ROOT/row['path'])==row['sha256'];files[row['path']]=row['sha256']
    paths=list((ROOT/'variants/staged_reconstruction/rubber_visibility').glob('*.py'))
    paths += [w.BASE/'launch.py',w.BASE/'plan.md',w.BASE/'protocol.json',w.BASE/'preregistration.json',
        ROOT/'tests/test_rubber_visibility.py',
        ROOT/'third_party/PIDG/submodules/flow-based-diff-gaussian-rasterization/cuda_rasterizer/forward.cu',
        ROOT/'third_party/PIDG/submodules/flow-based-diff-gaussian-rasterization/rasterize_points.cu']
    for path in paths:files[path.relative_to(ROOT).as_posix()]=sha(path)
    w.old.atomic_json(w.REPORT/'source_freeze.json',files);w.verify();print('J5_FREEZE',len(files),flush=True)


def qualify():
    assert not (w.REPORT/'qualification.json').exists()
    logs=[]
    for i in [0,1]:
        path=w.REPORT/f'qualification{i}.log';assert not path.exists()
        with path.open('xb') as stream:
            code=subprocess.run([sys.executable,'-u','-m','variants.staged_reconstruction.rubber_visibility.worker',
                '--enable-rubber-visibility-audit','--phase','qualify','--replicate',str(i)],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT).returncode
        assert code==0,f'Keep failed log {path}'
        logs.append({'path':path.relative_to(ROOT).as_posix(),'sha256':sha(path)})
    a,b=[read(w.REPORT/f'qualification{i}.json') for i in [0,1]]
    w.old.assert_equivalent(a,b,0,0)
    w.old.atomic_json(w.REPORT/'qualification.json',{'passed':True,'independent_processes':2,
        'group_hashes_targets_source0_fixture_exact':True,'logs':logs,'single_fixture':a['single_fixture']})


def main():
    p=argparse.ArgumentParser();p.add_argument('--enable-rubber-visibility-audit',action='store_true')
    p.add_argument('--phase',choices=['prepare','qualify'],default='prepare');args=p.parse_args()
    if not args.enable_rubber_visibility_audit:raise SystemExit('J5 default off')
    if args.phase=='prepare':prepare()
    else:qualify()

if __name__=='__main__':main()
