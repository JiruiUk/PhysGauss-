"""Explicit numerical-control worker; no native import or writes when disabled."""
import argparse
import runpy
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--enable-joint-rgb-stability', action='store_true')
    parser.add_argument('--native-check', action='store_true')
    parser.add_argument('--case', choices=['diffuse', 'glossy'], default='diffuse')
    parser.add_argument('--arm', choices=['frozen_static', 'revisable_static'], default='frozen_static')
    parser.add_argument('--preflight-tag', choices=['full', 'split'])
    parser.add_argument('--stop-after', type=int)
    args = parser.parse_args()
    if not args.enable_joint_rgb_stability:
        raise SystemExit('J1-R1 numerical control is disabled')
    assert args.native_check or args.preflight_tag, 'Preflight only: no formal training'
    from variants.staged_reconstruction.joint_rgb_stability.adapter import configure
    io = configure()
    if args.native_check:
        runpy.run_path(str(io.ROOT/'experiments/staged_reconstruction/S1_joint_rgb/native_check.py'), run_name='__main__')
    else:
        from variants.staged_reconstruction.joint_rgb.train import main as train
        sys.argv = [__file__, '--enable-joint-rgb', '--case', args.case, '--arm', args.arm,
            '--preflight-tag', args.preflight_tag]
        if args.stop_after:
            sys.argv += ['--stop-after', str(args.stop_after)]
        train()


if __name__ == '__main__':
    main()
