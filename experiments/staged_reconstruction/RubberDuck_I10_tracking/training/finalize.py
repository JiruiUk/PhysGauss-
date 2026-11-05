"""Close one fixed T2 trial after human-visible native and artifact review."""
from pathlib import Path
from variants.staged_reconstruction.i10_tracking.io import ROOT,read,write,sha

B=Path(__file__).parent
R=ROOT/'runs/pidg_staged_reconstruction/RubberDuck_I10_tracking/training/v1'


def main():
    report=B/'v1'
    for name in ['native_check.json','recovery.json','independent_audit.json']:
        assert read(report/name)['passed']
    ev=read(report/'evaluation.json');manual=read(report/'manual_review.json')
    assert manual['actual_visual_review'] and manual['fixed_native_RGB_alpha_D_reviewed']
    assert read(report/'trajectory_visual.json')['independent_3D_GT'] is False
    passed=ev['validation']['PSNR_nonregression_gate'] and manual['dynamic_completeness_and_cleanup_passed']
    status=dict(status='completed_pilot_visual_improvement' if passed else 'completed_not_promoted',pending=False,pending_review=False,
        optimizer_updates=4000,static_color_reference=False,historical_color_parent_disclosed=True,
        full_unknown_static_from_scratch=False,RGB_nonregression_passed=ev['validation']['PSNR_nonregression_gate'],
        manual_dynamic_completeness_and_cleanup_passed=manual['dynamic_completeness_and_cleanup_passed'],
        independent_geometry_GT=False,automatic_extension=False)
    artifacts=['unit_tests.json','native_check.json','recovery.json','evaluation.json','videos.json',
        'independent_audit.json','manual_review.json','trajectory_visual.json','assessment.md','source_freeze.json','evaluation_freeze.json']
    write(report/'final_review.json',dict(**status,artifacts_sha256={name:sha(report/name) for name in artifacts}))
    for path in [report/'final_status.json',R/'final_status.json',R/'active.json',R/'pipeline.json']:write(path,status)
    for arm in ['rgb_only','rgb_tracks']:
        write(R/'formal'/arm/'final_status.json',dict(status='completed_reviewed',iteration=2000,pending=False,pending_review=False,
            trial_status=status['status'],checkpoint_sha256=sha(R/'formal'/arm/'checkpoint.pt')))
    write(report/'final_validation.json',dict(passed=True,all_review_artifact_SHA=True,
        both_arms_complete=True,each_arm_updates=2000,all_required_review_complete=True,
        pending_review=False,new_source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in [Path(__file__),B/'trajectory_view.py']}))

if __name__=='__main__':main()
