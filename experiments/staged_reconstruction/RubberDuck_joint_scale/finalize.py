"""Close the reviewed read-only result; no optimizer or model mutation."""
from pathlib import Path
from variants.staged_reconstruction.i10_joint_scale import io


def main():
    io.verify();a=io.read(io.REPORT/'audit_v2.json');s=a['candidate']['summary']
    assert not s['local_evidence_gate_passed']
    assert io.read(io.REPORT/'independent_audit.json')['passed']
    assert io.read(io.REPORT/'independent_joint_replay.json')['passed']
    assert io.read(io.REPORT/'independent_replay.json')['passed']
    manual=io.read(io.REPORT/'manual_review.json');assert manual['actual_visual_review'] and manual['actual_reviewed_images']==7
    for path,h in manual['reviewed_images_sha256'].items():assert io.sha(io.ROOT/path)==h,path
    for name,receipt in [('independent_audit.py','independent_evaluation_freeze.json'),
                         ('independent_audit_v2.py','independent_evaluation_freeze_v2.json'),
                         ('review.py','figure_evaluation_freeze.json'),('track_review.py','track_figure_freeze.json')]:
        assert io.sha(io.BASE/name)==io.read(io.REPORT/receipt)['source_sha256'],name
    state=dict(status='completed_audit_not_admitted',pending=False,pending_review=False,optimizer_updates=0,
        current_model_replaced=False,native_training_integration_qualified=False,weak_depth_enabled=False,
        calibrated_views=s['calibrated_views'],calibrated_views_by_third=s['calibrated_views_by_third'],
        reliable_motion_tracks=s['reliable_motion_tracks'],distinct_rows=s['distinct_rows'],
        diagnostic_fit_observations=s['auxiliary_observations'],parent_preserved=True,
        original_T4_global_failure_preserved=True,independent_geometry_GT=False,historical_color_lineage_disclosed=True)
    for name in ['state.json','active.json','pipeline.json','final_status.json']:io.write(io.RUNS/name,state)
    io.write(io.REPORT/'final_status.json',state)
    artifacts=['assessment.md','audit_v2.json','independent_audit.json','independent_replay.json',
        'independent_joint_replay.json','manual_review.json','figures.json','track_figure.json','final_status.json',
        'source_freeze.json','extraction_freeze.json','numerical_repair_freeze.json',
        'independent_evaluation_freeze.json','independent_evaluation_freeze_v2.json',
        'figure_evaluation_freeze.json','track_figure_freeze.json','state_repair.json']
    io.write(io.REPORT/'final_review.json',dict(**state,independent_audit_passed=True,
        actual_visual_review=True,source_sha256=io.sha(Path(__file__)),
        artifacts_sha256={name:io.sha(io.REPORT/name) for name in artifacts},
        numerical_failures_preserved=['audit.json','measure_attempt1.log','independent_attempt1.log',
                                     'failure_attempt1.json','failure_attempt2.json','independent_attempt2.log'],
        audit_result_not_geometry_training_improvement=True))
    print('JOINT_SCALE_FINAL_REVIEW_CLOSED',s['calibrated_views'],s['reliable_motion_tracks'],flush=True)


if __name__=='__main__':main()
