"""Bind early-depth code, protocol and inherited immutable sources."""
import argparse
from ..prepare_data import ROOT, REPORT, read, write, sha
from ..union.prepare import verify_sources
from .prepare import EXP, INITIAL, INITIAL_SHA


def main():
    p=argparse.ArgumentParser(); p.add_argument('--verify',action='store_true'); a=p.parse_args()
    verify_sources()
    for branch in ['motion_integrity','sparse_motion','ordinal_depth']:
        for name,value in read(REPORT/branch/'source_freeze.json').items(): assert sha(ROOT/name)==value,name
    path=EXP/'source_freeze.json'
    if a.verify:
        for name,value in read(path).items(): assert sha(ROOT/name)==value,name
    else:
        assert not path.exists() and read(EXP/'preflight.json')['passed']
        files=[*sorted((ROOT/'variants/staged_reconstruction/early_depth').glob('*.py')),
               ROOT/'tests/test_early_depth.py',EXP/'research.md',EXP/'protocol.json',INITIAL]
        write(path,{f.relative_to(ROOT).as_posix():sha(f) for f in files})
    assert sha(INITIAL)==INITIAL_SHA
    print('EARLY_SOURCE_FREEZE_VERIFIED',flush=True)


if __name__=='__main__': main()
