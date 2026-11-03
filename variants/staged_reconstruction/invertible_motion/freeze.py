"""Bind optional invertible-motion code and the unchanged inherited source chain."""
import argparse
from ..prepare_data import ROOT,REPORT,read,write,sha
from ..union.prepare import verify_sources
from .prepare import EXP,INITIAL,INITIAL_SHA


def verify():
    verify_sources()
    for branch in ['motion_integrity','sparse_motion','ordinal_depth','early_depth','local_rigidity','coarse_motion','invertible_motion']:
        for name,value in read(REPORT/branch/'source_freeze.json').items(): assert sha(ROOT/name)==value,name
    assert sha(INITIAL)==INITIAL_SHA


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--verify',action='store_true'); args=parser.parse_args()
    if args.verify: verify()
    else:
        path=EXP/'source_freeze.json'; assert not path.exists() and read(EXP/'preflight.json')['passed']
        files=[*sorted((ROOT/'variants/staged_reconstruction/invertible_motion').glob('*.py')),
               ROOT/'tests/test_invertible_motion.py',EXP/'research.md',EXP/'protocol.json',INITIAL]
        write(path,{file.relative_to(ROOT).as_posix():sha(file) for file in files}); verify()
    print('INVERTIBLE_MOTION_SOURCE_FREEZE_VERIFIED',flush=True)


if __name__=='__main__': main()
