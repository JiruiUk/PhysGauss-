"""Close reviewed J8 trial without overwriting or auto-promoting retained J7."""
from pathlib import Path
from variants.staged_reconstruction.prepare_data import ROOT,read,sha,write

B=Path(__file__).resolve().parent
R=ROOT/'runs/pidg_staged_reconstruction/RubberDuck_J7_long_memory/v1'

def main():
    report=B/'v1';ev=read(report/'evaluation.json');manual=read(report/'manual_review.json')
    for name in ['final_validation.json','trajectory_audit.json','independent_input_audit.json','recovery.json','native_check.json','controller_audit.json']:
        assert read(report/name)['passed'],name
    assert manual['actually_viewed'] and manual['independent_geometry_GT'] is False
    for name,h in manual['images'].items():assert sha(ROOT/name)==h,name
    v=ev['validation'];numeric=bool(v['PSNR_nonregression_gate'] and v['parent_nonregression_gate'])
    geometry=bool(manual['static_structure_preserved'] and manual['dynamic_cleanup_and_completeness_improved'])
    status='completed_reviewed_candidate_not_auto_promoted' if numeric and geometry else 'completed_not_promoted'
    result=dict(status=status,pending_review=False,pending=False,formal_updates=4000,
        numeric_gates_passed=numeric,manual_structure_gate_passed=geometry,
        J7_retained=True,no_automatic_extension_or_sweep=True,independent_geometry_GT=False,
        original_L4_and_full40_failures_preserved=True,
        artifacts={name:sha(report/name) for name in ['evaluation.json','videos.json','final_validation.json','manual_review.json','trajectory_audit.json','controller_audit.json','assessment.md']},
        source_sha256=sha(Path(__file__)))
    write(report/'final_review.json',result);write(R/'final_status.json',result);write(R/'active.json',result)
    pipeline=read(R/'pipeline.json');pipeline.update(status=status,pending_review=False,pending=False);write(R/'pipeline.json',pipeline)
    for arm in read(B/'protocol.json')['arms']:
        write(R/'formal'/arm/'final_status.json',dict(status=status,iteration=2000,pending_review=False,pending=False))
    print('J8_REVIEW_CLOSED',status,flush=True)

if __name__=='__main__':main()
