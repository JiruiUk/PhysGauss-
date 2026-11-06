"""Close only after actual training, native replay, artifacts and visual review."""
from pathlib import Path
from variants.staged_reconstruction.i10_tracking.io import ROOT,read,write,sha

B=Path(__file__).parent;V=B/'v1';R=ROOT/'runs/pidg_staged_reconstruction/RubberDuck_I10_revocable_surface/v1'


def main():
    for name in ['unit_tests.json','native_check.json','recovery.json','independent_audit.json','controller_audit.json']:
        assert read(V/name)['passed'],name
    ev=read(V/'evaluation.json');manual=read(V/'manual_review.json')
    assert manual['actual_visual_review'] and manual['fixed_native_RGB_alpha_D_reviewed']
    passed=ev['validation']['PSNR_nonregression_gate'] and ev['validation']['parent_nonregression_gate'] and manual['dynamic_completeness_and_cleanup_passed']
    status=dict(status='completed_pilot_visual_improvement' if passed else 'completed_not_promoted',pending=False,pending_review=False,
        optimizer_updates=4000,each_arm_updates=2000,current_model_replaced=False,
        static_color_reference=False,historical_color_parent_disclosed=True,
        original_T3_T4_L1_failure_preserved=True,all7_native_controller_refreshes_replayed=True,
        RGB_nonregression_passed=ev['validation']['PSNR_nonregression_gate'],parent_nonregression_passed=ev['validation']['parent_nonregression_gate'],
        manual_completeness_cleanup_passed=manual['dynamic_completeness_and_cleanup_passed'],independent_geometry_GT=False,automatic_extension=False)
    names=['unit_tests.json','native_check.json','recovery.json','evaluation.json','videos.json','independent_audit.json',
        'controller_audit.json','terminal_probe.json','manual_review.json','assessment.md','source_freeze.json','trace_freeze.json','evaluation_freeze.json','hypotheses.json','initial_health.json']
    write(V/'final_review.json',dict(**status,artifacts_sha256={n:sha(V/n) for n in names}))
    for path in [V/'final_status.json',R/'final_status.json',R/'active.json',R/'pipeline.json']:write(path,status)
    for arm in ['fixed_surface','revocable_surface']:
        write(R/'formal'/arm/'final_status.json',dict(status='completed_reviewed',iteration=2000,pending=False,pending_review=False,
            trial_status=status['status'],checkpoint_sha256=sha(R/'formal'/arm/'checkpoint.pt')))
    write(V/'final_validation.json',dict(passed=True,pending_review=False,actual_models_updated=True,all_required_review_complete=True,
        completion_sources_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in [Path(__file__),B/'assess.py']}))


if __name__=='__main__':main()
