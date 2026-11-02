"""Bounded launcher with a receipted diagnostics-path relocation; no trainer edits."""
from pathlib import Path
import sys
import os
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from variants.shape_coherence import run
from variants.metric_tuning import run as engine


def control_unchanged():
    from output.c1_flow_ablation.reuse_control import control_hashes
    expected=engine.read_json(ROOT/'experiments/c1_flow_ablation/reuse_control_plan.json')['historical_control_sha256']
    assert control_hashes()==expected,'Historical C1 changed'


def main():
    repo=run.prepare();run.wire(repo);run.assert_frozen(repo)
    assert engine.read_json(run.REPORTS/'preflight.json')['status']=='passed'
    control_unchanged()
    with engine.lock():
        candidate=run.initialize(run.NAME);run.branch(candidate)
        p=candidate/'branch.json';receipt=engine.read_json(p)
        digest=receipt['original_source_checkpoint_sha256']
        assert digest==engine.read_json(run.REPORTS/'source_freeze.json')['full_3k_sha256']
        extra={'destination':str(candidate),'identical_full_checkpoint_sha256':digest}
        for k,v in extra.items():
            if k in receipt:assert receipt[k]==v
        engine.write_json(p,{**receipt,**extra})
    run.main()
    control_unchanged()
    status=engine.read_json(run.RUNS/'controller.json')['status']
    if status=='trained_pending_result_verification':
        from output.shape_coherence.finalize import main as finish
        finish()


if __name__=='__main__':main()
