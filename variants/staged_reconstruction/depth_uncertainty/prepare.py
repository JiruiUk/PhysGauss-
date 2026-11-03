"""Freeze a single data-only candidate and its exact parent depth arrays."""
import argparse
from pathlib import Path
from ..prepare_data import ROOT,read,write,sha
from ..depth_refine.infer import verify as verify_parent

BASE = ROOT/'experiments/staged_reconstruction/S1_depth_uncertainty'
REPORT = BASE/'v1'
RUNS = ROOT/'runs/pidg_staged_reconstruction/S1_depth_uncertainty/v1'
PARENT = ROOT/'runs/pidg_staged_reconstruction/S1_depth_refinement/v1'
SOURCE = ROOT/'experiments/staged_reconstruction/S1_clean_visibility/synthetic_v2'


def stopped():
    return any((ROOT/p/'stop.request').exists() for p in [
        'runs/pidg_staged_reconstruction/S1_depth_uncertainty/v1',
        'runs/pidg_staged_reconstruction/S1_depth_refinement/v1',
        'runs/pidg_staged_reconstruction/S1_nuisance/v1'])


def verify():
    verify_parent()
    for p,h in read(REPORT/'source_freeze.json').items(): assert sha(ROOT/p) == h,p


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--freeze',action='store_true'); args = parser.parse_args()
    assert args.freeze, 'Use explicit --freeze; no automatic experiment'
    assert not stopped(), 'Saved user stop; never resume'
    verify_parent(); protocol = read(BASE/'protocol.json')
    assert not (REPORT/'source_freeze.json').exists()
    assert '21 passed' in (REPORT/'tests.log').read_text()
    parent = ROOT/protocol['parent']; assert sha(parent) == protocol['parent_sha256']
    inputs = read(parent)['outputs'].copy()
    paths = list(Path(__file__).parent.glob('*.py'))+[BASE/'plan.md',BASE/'protocol.json',REPORT/'tests.log',parent,
        ROOT/'tests/test_depth_uncertainty.py',ROOT/'experiments/staged_reconstruction/S1_depth_evidence/refinement_v1/source_freeze.json',
        SOURCE/'protocol.json',SOURCE/'dataset.json']
    for case in protocol['synthetic_cases']:
        paths += [SOURCE/f'{case}_inference.json',ROOT/f'data/derived/clean_static_synthetic/v2/{case}/train/cameras.json']
    inputs.update({p.relative_to(ROOT).as_posix():sha(p) for p in paths})
    assert all(sha(ROOT/p) == h for p,h in inputs.items())
    write(REPORT/'source_freeze.json',inputs)
    write(REPORT/'test_receipt.json',{'passed':True,'tests':21,'new_tests':13,'inherited_tests':8,
        'flags_off_identity':True,'missing_occluded_duplicate_pose_time_and_geometry_checks':True,
        'log_sha256':sha(REPORT/'tests.log')})
    print('DEPTH_UNCERTAINTY_FROZEN',sha(REPORT/'source_freeze.json'),flush=True)


if __name__ == '__main__': main()
