"""Independent numerical repair; retain the original failed audit untouched."""
import argparse
from pathlib import Path
from unittest.mock import patch
from variants.staged_reconstruction.i10_joint_scale import io,core
from variants.staged_reconstruction.i10_joint_scale.audit import calculate
from experiments.staged_reconstruction.RubberDuck_joint_scale.solver_v2 import fit


def main():
    p=argparse.ArgumentParser();p.add_argument('--enable-joint-scale-audit',action='store_true')
    p.add_argument('stage',choices=['freeze','measure','replay']);args=p.parse_args()
    if not args.enable_joint_scale_audit:raise SystemExit('Joint scale audit default off')
    with io.lock():
        io.verify();receipt=io.REPORT/'numerical_repair_freeze.json'
        if args.stage=='freeze':
            assert not receipt.exists()
            paths=[Path(__file__),io.BASE/'solver_v2.py',io.BASE/'numerical_repair.md',io.REPORT/'audit.json',
                   io.ROOT/'tests/test_i10_joint_scale_solver.py']
            io.write(receipt,dict(files={x.relative_to(io.ROOT).as_posix():io.sha(x) for x in paths},
                old_attempt_preserved=True,objective_bounds_initialization_tolerances_iteration_budget_unchanged=True));return
        for name,h in io.read(receipt)['files'].items():assert io.sha(io.ROOT/name)==h,name
        with patch.object(core,'joint_fit',fit):result=calculate()
        if args.stage=='measure':
            assert not (io.REPORT/'audit_v2.json').exists();io.write(io.REPORT/'audit_v2.json',result)
            io.write(io.RUNS/'state.json',dict(status='completed_audit_pending_review',pending_review=True,optimizer_updates=0))
            print('JOINT_SCALE_V2_RESULT',result['candidate']['summary'],flush=True)
            for q in result['joint_solutions']:print('SOLVER',q['success'],q['iterations'],q['optimality'],flush=True)
        else:
            assert result==io.read(io.REPORT/'audit_v2.json')
            io.write(io.REPORT/'independent_joint_replay.json',dict(passed=True,all120_decisions_exact=True,
                repaired_source_sha256=io.sha(io.BASE/'solver_v2.py'),optimizer_updates=0))
            print('JOINT_SCALE_V2_REPLAY_PASS',flush=True)
        io.verify()


if __name__=='__main__':main()
