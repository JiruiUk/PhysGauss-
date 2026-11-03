"""Hash-bind the fixed evaluator and every parent/data asset before measuring."""
import argparse
from pathlib import Path
from ..prepare_data import ROOT,read,write,sha

BASE = ROOT/'experiments/staged_reconstruction/S1_photometric_oracle'
REPORT = BASE/'v1'
RUNS = ROOT/'runs/pidg_staged_reconstruction/S1_photometric_oracle/v1'
OUT = ROOT/'output/staged_reconstruction/S1_photometric_oracle/v1'
D1 = ROOT/'runs/pidg_staged_reconstruction/S1_depth_refinement/v1'
SOURCE = ROOT/'experiments/staged_reconstruction/S1_clean_visibility/synthetic_v2'
D1_REPORT = ROOT/'experiments/staged_reconstruction/S1_depth_evidence/refinement_v1'


def stopped():
    return any(p.exists() for p in [BASE/'stop.request',RUNS/'stop.request',
        ROOT/'runs/pidg_staged_reconstruction/stop.request',ROOT/'stop.request'])


def verify():
    for name,h in read(REPORT/'source_freeze.json').items():
        assert sha(ROOT/name) == h,name


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--freeze',action='store_true'); args = parser.parse_args()
    assert args.freeze and not stopped()
    assert not (REPORT/'source_freeze.json').exists()
    assert 'passed' in (REPORT/'tests.log').read_text()
    dataset = read(SOURCE/'dataset.json'); inputs = dict(dataset['files'])
    parent = read(D1_REPORT/'inference.json'); inputs.update(parent['outputs'])
    paths = list(Path(__file__).parent.glob('*.py')) + [BASE/'plan.md',BASE/'protocol.json',BASE/'literature.md',
        ROOT/'tests/test_photometric_oracle.py',REPORT/'tests.log',D1_REPORT/'inference.json',
        D1_REPORT/'source_freeze.json',SOURCE/'dataset.json',
        ROOT/'variants/staged_reconstruction/adaptive_static/geometry.py',
        ROOT/'variants/staged_reconstruction/rgb_prior_proposals/core.py',
        ROOT/'variants/staged_reconstruction/synthetic_audit/input.py',
        ROOT/'variants/staged_reconstruction/synthetic_audit/generate.py',
        ROOT/'variants/staged_reconstruction/synthetic_audit/generate_dense.py',
        ROOT/'variants/staged_reconstruction/pretrained_depth/model.py',
        ROOT/'variants/staged_reconstruction/rgb_prior_proposals/io.py']
    for case in ['diffuse','glossy']: paths.append(SOURCE/f'{case}_inference.json')
    inputs.update({p.relative_to(ROOT).as_posix():sha(p) for p in paths})
    # Verify inherited frozen hashes as well, without invoking/restarting parents.
    for receipt in [D1_REPORT/'source_freeze.json']:
        for name,h in read(receipt).items(): assert sha(ROOT/name) == h,name
    for name,h in inputs.items(): assert sha(ROOT/name) == h,name
    write(REPORT/'source_freeze.json',inputs)
    print('PHOTOMETRIC_ORACLE_FROZEN',sha(REPORT/'source_freeze.json'),len(inputs),flush=True)


if __name__ == '__main__': main()
