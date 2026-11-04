"""Independent evaluation-only repair; original failed worker/freeze stay intact."""
import argparse
import subprocess
import sys
from types import SimpleNamespace
import numpy as np
import torch
from variants.staged_reconstruction.rubber_visibility import worker as w


def centered_intrinsics(width, height):
    return np.array([[100., 0., (width-1)/2], [0., 100., (height-1)/2], [0., 0., 1.]])


def verify_supplement():
    w.verify()
    for name, value in w.read(w.REPORT/'repair_source_freeze.json').items():
        assert w.sha(w.ROOT/name) == value, name


@torch.no_grad()
def corrected_single_fixture(renderer):
    from variants.staged_reconstruction.nuisance_static.runtime import calibrated_camera
    p=w.read(w.BASE/'protocol.json')['native_single_control'];width,height=p['width'],p['height']
    cam=calibrated_camera(np.eye(4), centered_intrinsics(width,height), width,height)
    bank=SimpleNamespace(get_xyz=torch.tensor([[0.,0.,p['z']]],device='cuda'),
        get_scaling=torch.full((1,3),p['scale'],device='cuda'),get_rotation=torch.tensor([[1.,0.,0.,0.]],device='cuda'),
        get_opacity=torch.full((1,1),p['opacity'],device='cuda'),get_features=torch.zeros((1,16,3),device='cuda'),
        max_sh_degree=3,active_sh_degree=3)
    package=renderer(cam,bank,w.PIPE,torch.zeros(3,device='cuda'),0.,0.,0.,override_color=torch.ones((1,3),device='cuda'))
    a=package['alpha'].squeeze();weight=package['weight_per_gs_pixel'][0];valid=package['gs_per_pixel'][0]==0
    assert valid.any() and torch.equal(package['render'][0],a)
    def error(x):return float((weight[valid]-x[valid]).abs().max())
    return {'single_native_white_RGB_equals_alpha_exact':True,'valid_pixels':int(valid.sum()),
        'alpha_max':float(a.max()),'stored_weight_vs_compositing_alpha_max_error':error(a),
        'stored_weight_vs_alpha_times_one_minus_alpha_max_error':error(a*(1-a)),
        'stored_weights_used_in_J5':False,'control_not_a_training_asset':True}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--enable-rubber-visibility-audit',action='store_true')
    parser.add_argument('--phase',choices=['prepare','qualify','replicate'],default='prepare')
    parser.add_argument('--replicate',type=int,choices=[2,3],default=2);args=parser.parse_args()
    if not args.enable_rubber_visibility_audit:raise SystemExit('J5 repair default off; no data/model opened')
    assert not w.stopped()
    if args.phase=='prepare':
        assert not (w.REPORT/'repair_source_freeze.json').exists()
        w.verify()
        names=[__file__,str(w.ROOT/'tests/test_rubber_visibility_repair.py')]
        files={w.Path(x).relative_to(w.ROOT).as_posix():w.sha(w.Path(x)) for x in names}
        w.old.atomic_json(w.REPORT/'repair_source_freeze.json',files)
        w.old.atomic_json(w.REPORT/'repair_record.json',{'original_failed_log':'qualification0.log',
            'original_frozen_sources_preserved':True,'change':'Evaluation fixture principal point (w/2,h/2) corrected to native ((w-1)/2,(h-1)/2)',
            'protocol_or_gates_changed':False,'replicates':[2,3]})
        return
    verify_supplement();w.old.RUNS=w.RUNS
    if args.phase=='replicate':
        with w.old.lock('worker.lock'):
            w.single_fixture=corrected_single_fixture
            w.qualify(args.replicate)
            verify_supplement()
        return
    assert not (w.REPORT/'qualification.json').exists();logs=[]
    for i in [2,3]:
        path=w.REPORT/f'qualification{i}.log'
        with path.open('xb') as stream:
            code=subprocess.run([sys.executable,'-u','-m',__spec__.name,'--enable-rubber-visibility-audit',
                '--phase','replicate','--replicate',str(i)],cwd=w.ROOT,stdout=stream,stderr=subprocess.STDOUT).returncode
        assert code==0,f'Preserve failure {path}'
        logs.append({'path':path.relative_to(w.ROOT).as_posix(),'sha256':w.sha(path)})
    a,b=[w.read(w.REPORT/f'qualification{i}.json') for i in [2,3]];w.old.assert_equivalent(a,b,0,0)
    verify_supplement()
    w.old.atomic_json(w.REPORT/'qualification.json',{'passed':True,'independent_processes':2,
        'group_hashes_targets_source0_fixture_exact':True,'logs':logs,'single_fixture':a['single_fixture'],
        'evaluation_repair_receipt':'repair_record.json','failed_attempt_retained':True})
    print('J5_REPAIRED_QUALIFICATION_PASS',flush=True)


if __name__=='__main__':main()
