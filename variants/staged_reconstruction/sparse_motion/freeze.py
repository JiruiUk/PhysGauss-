"""Bind sources and sparse node parameters without changing inherited freezes."""
import argparse
from ..prepare_data import ROOT, REPORT, read, write, sha
from ..union.prepare import verify_sources
from .prepare import EXP, WORK


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--verify', action='store_true')
    args = p.parse_args()
    verify_sources()
    # First iteration inputs/implementation remain immutable as well.
    for name, expected in read(REPORT/'motion_integrity/source_freeze.json').items():
        assert sha(ROOT/name) == expected, name
    path = EXP/'source_freeze.json'
    if args.verify:
        for name, expected in read(path).items():
            assert sha(ROOT/name) == expected, name
    else:
        assert not path.exists()
        preflight = read(EXP/'preflight.json')
        assert preflight['passed'] and preflight['fully_sparse_optimizer_updates'] >= 100
        files = [*sorted((ROOT/'variants/staged_reconstruction/sparse_motion').glob('*.py')),
                 ROOT/'tests/test_sparse_motion.py', EXP/'research.md', EXP/'protocol.json', WORK/'controls.npz']
        write(path, {f.relative_to(ROOT).as_posix(): sha(f) for f in files})
    print('SPARSE_SOURCE_FREEZE_VERIFIED', flush=True)


if __name__ == '__main__':
    main()
