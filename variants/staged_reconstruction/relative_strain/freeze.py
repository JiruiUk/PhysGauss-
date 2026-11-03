"""Freeze I9 and verify every unchanged inherited experimental source."""
import argparse
from ..prepare_data import ROOT, read, write, sha
from ..covariance_transport.freeze import verify as verify_inherited
from .prepare import EXP, INITIAL, INITIAL_SHA


def verify():
    verify_inherited()
    for name, value in read(EXP/'source_freeze.json').items():
        assert sha(ROOT/name) == value, name
    assert sha(INITIAL) == INITIAL_SHA


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if args.verify:
        verify()
    else:
        path = EXP/'source_freeze.json'
        assert not path.exists() and read(EXP/'preflight.json')['passed']
        files = [*sorted((ROOT/'variants/staged_reconstruction/relative_strain').glob('*.py')),
                 ROOT/'tests/test_relative_strain.py', ROOT/'scripts/audit_invertible_strain.py',
                 EXP/'research.md', EXP/'protocol.json', INITIAL]
        write(path, {file.relative_to(ROOT).as_posix(): sha(file) for file in files})
        verify()
    print('RELATIVE_STRAIN_SOURCE_FREEZE_VERIFIED', flush=True)


if __name__ == '__main__': main()
