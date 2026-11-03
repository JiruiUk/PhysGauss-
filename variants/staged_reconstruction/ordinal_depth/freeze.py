"""Bind every new source and input; preserve previous experiment freezes."""
import argparse
from ..prepare_data import ROOT, REPORT, read, write, sha
from ..union.prepare import verify_sources
from .prepare import EXP, WORK


def main():
    p = argparse.ArgumentParser(); p.add_argument('--verify', action='store_true'); args = p.parse_args()
    verify_sources()
    for branch in ['motion_integrity','sparse_motion']:
        for name, expected in read(REPORT/branch/'source_freeze.json').items():
            assert sha(ROOT/name) == expected, name
    path = EXP/'source_freeze.json'
    if args.verify:
        for name, expected in read(path).items():
            assert sha(ROOT/name) == expected, name
    else:
        assert not path.exists() and read(EXP/'preflight.json')['passed']
        assert read(EXP/'preflight.json')['full_weight_updates'] >= 100
        files = [*sorted((ROOT/'variants/staged_reconstruction/ordinal_depth').glob('*.py')),
                 ROOT/'tests/test_ordinal_depth.py', EXP/'research.md', EXP/'protocol.json', *sorted((WORK/'pairs').glob('*.npz'))]
        write(path, {f.relative_to(ROOT).as_posix(): sha(f) for f in files})
    print('ORDINAL_SOURCE_FREEZE_VERIFIED', flush=True)


if __name__ == '__main__':
    main()
