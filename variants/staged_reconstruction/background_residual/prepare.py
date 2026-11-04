"""Fix a new controlled continuation before any optimizer update."""
import argparse
from pathlib import Path
from ..prepare_data import ROOT, DATA, read, write, sha
from ..motion_integrity.prepare import PARENT, PARENT_SHA, load_parent

EXP = ROOT / 'experiments/staged_reconstruction/background_residual'
WORK = ROOT / 'runs/pidg_staged_reconstruction/background_residual'


def verify():
    for name, expected in read(EXP/'source_freeze.json').items():
        assert sha(ROOT/name) == expected, name
    plan = read(EXP/'protocol.json')
    assert sha(PARENT) == PARENT_SHA == plan['parent_sha256']
    assert sha(DATA/'transforms_train.json') == plan['metadata_sha256']
    return plan


def main():
    p = argparse.ArgumentParser(); p.add_argument('--enable-background-residual', action='store_true'); args = p.parse_args()
    if not args.enable_background_residual:
        raise SystemExit('Background residual neutralization is opt-in')
    assert not (EXP/'protocol.json').exists()
    assert '8 passed' in (EXP/'unit_tests.log').read_text(encoding='utf-8', errors='replace')
    assert sha(PARENT) == PARENT_SHA
    inherited = read(ROOT/'experiments/staged_reconstruction/motion_integrity/protocol.json')
    plan = {'enabled_by_default': False, 'arms': ['I10_control', 'I10_static_residual'],
        'parent': PARENT.relative_to(ROOT).as_posix(), 'parent_sha256': PARENT_SHA,
        'actual_updates': 2000, 'smoke_updates': 100, 'width': 1600,
        'absolute_physics_step': '6000+local_update; original interval, weights and gradient guard',
        'intervention': 'Replace actual RGB target only at original observed blue static pixels by stop-gradient frozen S1 RGB',
        'same_initialization_Adam_RNG_sampler_physics_cache': True,
        'foreground_L1_and_empty_background_targets_unchanged': True,
        'static_mask': 'Original GT RGB blue rule, no instance or material segmentation',
        'static_render_is_model_conditioned_not_GT': True,
        'metadata_sha256': sha(DATA/'transforms_train.json'),
        'fixed_cameras': inherited['fixed_cameras'], 'screen_times': inherited['screen_times'],
        'full_acceptance_times': inherited['full_acceptance_times'], 'acceptance': inherited['acceptance'],
        'reference_dynamic_PSNR': 25.181221, 'static_reference_PSNR': 35.6438163,
        'static_PSNR_drop_max_dB': 1., 'test_policy': 'No test RGB for iterative training or selection',
        'scope_limit': 'Continuation-time residual coupling only, not static initialization or earlier damage'}
    write(EXP/'protocol.json', plan)
    files = [*sorted(Path(__file__).parent.glob('*.py')), ROOT/'tests/test_background_residual.py',
        EXP/'research.md', EXP/'protocol.json', ROOT/'variants/staged_reconstruction/union/train.py',
        ROOT/'variants/staged_reconstruction/union/model.py', ROOT/'variants/staged_reconstruction/train_static.py',
        ROOT/'variants/staged_reconstruction/stage2/train.py', ROOT/'variants/staged_reconstruction/motion_integrity/prepare.py']
    write(EXP/'source_freeze.json', {f.relative_to(ROOT).as_posix(): sha(f) for f in files})
    WORK.mkdir(parents=True, exist_ok=True)
    print('BACKGROUND_RESIDUAL_PROTOCOL_FIXED', plan, flush=True)


if __name__ == '__main__':
    main()
