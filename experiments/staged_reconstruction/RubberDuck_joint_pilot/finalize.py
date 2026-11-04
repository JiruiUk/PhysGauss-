"""Finalize only after an actual image review; do not manufacture geometry GT."""
from datetime import datetime,timezone
from pathlib import Path
from variants.staged_reconstruction.rubber_joint import runtime as r
from variants.staged_reconstruction.prepare_data import ROOT,read,sha


def main():
    assert not r.stopped();r.verify();assert not (r.REPORT/'final_review.json').exists()
    evaluation=read(r.REPORT/'evaluation.json');videos=read(r.REPORT/'videos.json')
    manual=read(r.REPORT/'manual_review.json');assert manual['actually_viewed']
    for name,h in {**evaluation['diagnostics']['images'],**manual['images']}.items():assert sha(ROOT/name)==h,name
    assert len(evaluation['plys'])==15
    for row in evaluation['plys'].values():assert sha(ROOT/row['path'])==row['sha256']
    assert len(videos['videos'])==2
    for row in videos['videos'].values():
        assert row['decoded_frames']==200 and sha(ROOT/row['path'])==row['sha256']
        assert sha(ROOT/row['decoded_contact'])==row['decoded_contact_sha256']
    numerical=evaluation['validation']['PSNR_nonregression_gate']
    admitted=numerical and manual['clear_alpha_depth_cleanup_without_visible_structure_loss']
    result={'status':'completed_passed_real_pilot_review' if admitted else 'completed_not_promoted',
        'pending_review':False,'formal_new_updates':4000,'qualification_updates_excluded':40,
        'numerical_PSNR_nonregression_passed':numerical,'manual_native_images_reviewed':True,
        'clear_alpha_depth_cleanup_without_visible_structure_loss':manual['clear_alpha_depth_cleanup_without_visible_structure_loss'],
        'pilot_review_admitted':admitted,'independent_geometry_GT_available':False,
        'not_a_claim_of_complete_or_correct_geometry':True,'legacy_color_conditioned_parent_disclosed':True,
        'physics_updated':False,'matched_initial_full_state_and2000_sampling_audited':True,
        'checkpoint_audits':evaluation['checkpoint_audits'],
        'evaluation_sha256':sha(r.REPORT/'evaluation.json'),'videos_sha256':sha(r.REPORT/'videos.json'),
        'manual_review_sha256':sha(r.REPORT/'manual_review.json'),'assessment_sha256':sha(r.REPORT/'assessment.md'),
        'source_sha256':sha(Path(__file__)),'completed_UTC':datetime.now(timezone.utc).isoformat()}
    r.atomic_json(r.REPORT/'final_review.json',result);r.atomic_json(r.RUNS/'final_status.json',result)
    for name in ['active.json','pipeline.json']:
        state=read(r.RUNS/name)
        state.update(status=result['status'],pending_review=False,worker_pids_are_historical=True,
            final_review_sha256=sha(r.REPORT/'final_review.json'));r.atomic_json(r.RUNS/name,state)
    for arm in ['frozen_static','joint_depth_trust']:
        r.atomic_json(r.RUNS/'formal'/arm/'final_status.json',{'status':result['status'],'iteration':2000,
            'pending_review':False,'final_review_sha256':sha(r.REPORT/'final_review.json')})
    print('J4_FINAL_REVIEW',result['status'],flush=True)

if __name__=='__main__':main()
