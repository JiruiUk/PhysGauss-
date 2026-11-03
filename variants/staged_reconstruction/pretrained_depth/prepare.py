"""Freeze pretrained evidence, RGB inputs and original train-inferred anchors."""
import argparse
from pathlib import Path
from ..prepare_data import ROOT, read, write, sha
from ..depth_uncertainty.prepare import verify as verify_parent, stopped as parent_stopped

BASE = ROOT / 'experiments/staged_reconstruction/S1_pretrained_geometry'
REPORT = BASE / 'v1'
RUNS = ROOT / 'runs/pidg_staged_reconstruction/S1_pretrained_geometry/v1'
PARENT = ROOT / 'runs/pidg_staged_reconstruction/S1_depth_uncertainty/v1'
SOURCE = ROOT / 'experiments/staged_reconstruction/S1_clean_visibility/synthetic_v2'


def stopped():
    return parent_stopped() or (RUNS / 'stop.request').exists() or (BASE / 'stop.request').exists()


def verify():
    verify_parent()
    for p, h in read(REPORT / 'source_freeze.json').items():
        assert sha(ROOT / p) == h, p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', action='store_true')
    args = parser.parse_args()
    assert args.freeze and not stopped()
    verify_parent()
    assert not (REPORT / 'source_freeze.json').exists()
    assert '27 passed' in (REPORT / 'tests.log').read_text(errors='replace')
    p = read(BASE / 'protocol.json'); parent = ROOT / p['parent']
    assert sha(parent) == p['parent_sha256']
    upstream = read(BASE / 'upstream_receipt.json')
    inputs = {**upstream['files'], **read(parent)['outputs']}
    paths = list(Path(__file__).parent.glob('*.py')) + [
        BASE / 'plan.md', BASE / 'protocol.json', BASE / 'evaluator_design.md', BASE / 'upstream_receipt.json',
        BASE / 'run_pipeline.py', REPORT / 'tests.log', parent,
        ROOT / 'tests/test_pretrained_depth.py',
        ROOT / 'variants/staged_reconstruction/synthetic_audit/input.py',
        ROOT / 'variants/staged_reconstruction/synthetic_audit/generate.py']
    for case in p['synthetic_cases']:
        train = ROOT / f'data/derived/clean_static_synthetic/v2/{case}/train'
        paths += [train / 'cameras.json', SOURCE / f'{case}_inference.json'] + list((train / 'rgb').glob('*.png'))
    inputs.update({path.relative_to(ROOT).as_posix(): sha(path) for path in paths})
    for path, h in inputs.items():
        assert sha(ROOT / path) == h, path
    write(REPORT / 'source_freeze.json', inputs)
    write(REPORT / 'test_receipt.json', {'passed': True, 'tests': 27, 'new_tests': 14,
          'flags_off_exact_identity': True, 'no_PIDG_training_or_renderer_changes': True,
          'log_sha256': sha(REPORT / 'tests.log')})
    print('PRETRAINED_DEPTH_FROZEN', sha(REPORT / 'source_freeze.json'), flush=True)


if __name__ == '__main__':
    main()
