"""Freeze this candidate, its training-only parents and RGB input boundary."""
import argparse
from pathlib import Path
from ..prepare_data import ROOT, read, write, sha
from ..pretrained_depth.prepare import verify as parent_verify, stopped as parent_stopped

BASE = ROOT/'experiments/staged_reconstruction/S1_rgb_prior_proposals'
REPORT = BASE/'v1'
RUNS = ROOT/'runs/pidg_staged_reconstruction/S1_rgb_prior_proposals/v1'
D1 = ROOT/'runs/pidg_staged_reconstruction/S1_depth_refinement/v1'
D2 = ROOT/'runs/pidg_staged_reconstruction/S1_depth_uncertainty/v1'
D3 = ROOT/'runs/pidg_staged_reconstruction/S1_pretrained_geometry/v1'
SOURCE = ROOT/'experiments/staged_reconstruction/S1_clean_visibility/synthetic_v2'


def stopped():
    return parent_stopped() or (RUNS/'stop.request').exists() or (BASE/'stop.request').exists()


def verify():
    parent_verify()
    for path, h in read(REPORT/'source_freeze.json').items(): assert sha(ROOT/path) == h, path


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--freeze', action='store_true'); args = parser.parse_args()
    assert args.freeze and not stopped()
    parent_verify(); assert not (REPORT/'source_freeze.json').exists()
    assert '22 passed' in (REPORT/'tests.log').read_text(errors='replace')
    assert read(REPORT/'preflight.json')['passed']
    p = read(BASE/'protocol.json'); inputs = {}
    paths = list(Path(__file__).parent.glob('*.py')) + [
        BASE/'plan.md', BASE/'protocol.json', BASE/'run_pipeline.py', ROOT/'tests/test_rgb_prior_proposals.py',
        REPORT/'tests.log', REPORT/'tests_attempt1.log', REPORT/'preflight.json',
        ROOT/'variants/staged_reconstruction/adaptive_static/geometry.py',
        ROOT/'variants/staged_reconstruction/synthetic_audit/input.py']
    for name, h in p['parent_receipts_sha256'].items():
        assert sha(ROOT/name) == h, name
        paths.append(ROOT/name); inputs.update(read(ROOT/name)['outputs'])
    for case in p['synthetic_cases']:
        train = ROOT/f'data/derived/clean_static_synthetic/v2/{case}/train'
        paths += [train/'cameras.json', SOURCE/f'{case}_inference.json'] + list((train/'rgb').glob('*.png'))
    inputs.update({path.relative_to(ROOT).as_posix(): sha(path) for path in paths})
    for path, h in inputs.items(): assert sha(ROOT/path) == h, path
    write(REPORT/'source_freeze.json', inputs)
    write(REPORT/'test_receipt.json', {'passed': True, 'tests': 22, 'new': 14, 'inherited': 8,
        'preflight_sha256': sha(REPORT/'preflight.json'), 'flags_off_identity': True,
        'no_PIDG_or_Gaussian_optimization_change': True})
    print('RGB_PRIOR_PROPOSAL_FROZEN', sha(REPORT/'source_freeze.json'), flush=True)


if __name__ == '__main__': main()
