"""Artifact-isolated adapter; frozen T0 implementation and outputs unchanged."""
import argparse
from pathlib import Path
from ..i10_tracking import io,extract,bind,audit


def install(enabled=False):
    if not enabled:return None
    base=io.ROOT/'experiments/staged_reconstruction/RubberDuck_I10_tracking/dense'
    run=io.ROOT/'runs/pidg_staged_reconstruction/RubberDuck_I10_tracking/dense/v1'
    out=io.ROOT/'output/staged_reconstruction/RubberDuck_I10_tracking/dense/v1'
    for module in [io,extract,bind,audit]:
        module.BASE=base;module.REPORT=base/'v1';module.RUNS=run;module.OUT=out
    return base


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--enable-dense-tracking',action='store_true')
    parser.add_argument('stage',choices=['freeze','extract','bind','replay','tracker-replay']);args=parser.parse_args()
    if not args.enable_dense_tracking:raise SystemExit('Dense tracking is opt-in')
    base=install(True)
    if args.stage=='freeze':
        io.freeze();receipt=io.read(base/'v1/source_freeze.json')
        paths=list(Path(__file__).parent.glob('*.py'))+[base/'preregistration.json']
        receipt['files'].update({p.relative_to(io.ROOT).as_posix():io.sha(p) for p in paths})
        io.write(base/'v1/source_freeze.json',receipt);io.verify()
    elif args.stage=='extract':extract.run()
    elif args.stage=='bind':bind.run()
    elif args.stage=='replay':bind.run(True)
    else:audit.run()

if __name__=='__main__':main()
