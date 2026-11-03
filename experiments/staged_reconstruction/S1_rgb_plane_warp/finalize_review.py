"""Conclude the measured data-only stage after actual image review."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
from variants.staged_reconstruction.prepare_data import ROOT,read,write,sha
from variants.staged_reconstruction.rgb_plane_warp.prepare import REPORT,RUNS,verify


def main():
    p=argparse.ArgumentParser(); p.add_argument('--images-reviewed',action='store_true'); args=p.parse_args()
    assert args.images_reviewed and not (REPORT/'final_review.json').exists()
    verify(); assert read(REPORT/'input_audit.json')['passed']
    assert read(RUNS/'pipeline.json')['status']=='measured_pending_review'
    gate=read(REPORT/'gate.json'); evaluation=read(REPORT/'evaluation.json'); exports=read(REPORT/'exports.json')
    assert exports['all_original_dataset_hashes_verified']
    images={**evaluation['images'],**exports['images']}
    for name,h in images.items(): assert sha(ROOT/name)==h,name
    if gate['both_cases_Gaussian_evidence']: status='complete_depth_evidence_passed_requires_Gaussian_protocol'
    elif gate['both_cases_numeric_improvement']: status='complete_numeric_improved_not_Gaussian_admitted'
    else: status='complete_rejected_geometry_gate'
    final={'status':status,'manual_review_completed':True,'reviewed_images':images,
        'actually_viewed':'both cases fixed0/24/72 raw RGB, same-pixel D1/D6 error and accepted change maps; all three analytic controls compared against original cost',
        'all192_views_and_parent_candidates_complete':True,'source_input_parent_slope_acceptance_audit_passed':True,
        'fixed0_24_72_full_RGB_replay_exact':True,'flags_off_identity_and_independent2plus2_recovery_passed':True,
        'Gaussian_or_real_training_started':False,'native86_188_PLY_video_or_Adam_recovery_claim':False,
        'unknown_identity_and_S3U_preserved':True,'gates_unchanged':True,
        'evidence_sha256':{name:sha(REPORT/name) for name in ['source_freeze.json','tests.log','preflight.json','inference.json','input_audit.json','evaluation.json','gate.json','exports.json','coverage_diagnostic.json','assessment.md']},
        'next_plan_sha256':sha(REPORT.parent/'next_plan.md'),'reviewed_at_utc':datetime.now(timezone.utc).isoformat(),'finalizer_sha256':sha(Path(__file__))}
    write(REPORT/'final_review.json',final)
    write(RUNS/'final_status.json',{'status':status,'pending_review':False,'views':192,'final_review_sha256':sha(REPORT/'final_review.json'),
        'Gaussian_or_real_training':False})
    pipeline=read(RUNS/'pipeline.json'); pipeline.update(status='complete',stage='complete',manual_review='complete',final_review_sha256=sha(REPORT/'final_review.json'))
    write(RUNS/'pipeline.json',pipeline)
    write(RUNS/'state.json',{'status':status,'pending_review':False,'GPU_process_required':False})
    print('D6_FINAL_REVIEW_COMPLETE',status,sha(REPORT/'final_review.json'),flush=True)


if __name__=='__main__': main()
