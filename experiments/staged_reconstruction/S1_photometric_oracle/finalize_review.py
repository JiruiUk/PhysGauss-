"""Record completed human-visible artifact review and reconcile final states."""
from pathlib import Path
from datetime import datetime,timezone
from variants.staged_reconstruction.prepare_data import ROOT,read,write,sha
from variants.staged_reconstruction.photometric_oracle.prepare import REPORT,RUNS,OUT,verify


def main():
    verify(); independent=read(REPORT/'independent_audit.json'); assert independent['passed']
    assert independent['measurement_sha256']==sha(REPORT/'measurement.json')
    assert read(RUNS/'state.json')['status']=='measured_pending_review'
    images=[OUT/f'{case}_view_{i:03d}.png' for case in ['diffuse','glossy'] for i in [0,24,72]]+[OUT/'analytic_control_curves.png']
    assert all(p.exists() for p in images)
    assert not [p for p in RUNS.rglob('*') if p.is_file() and p.suffix.lower() in ['.npz','.npy','.pt','.pth','.ply','.mp4']]
    status='complete_matching_measurement_audited'
    final={'status':status,'manual_review_completed':True,'reviewed_images':{p.relative_to(ROOT).as_posix():sha(p) for p in images},
        'actually_viewed':'two cases fixed0/24/72 RGB, face/visibility grouping, GT/grid-vs-D1 fit differences and acceptance; all six curves of the three fixed analytic controls',
        'finding':'Original measurement rejects true large-error correction even on same-face/all6-visible rays; fractional control biased; quantization/center occlusion alone insufficient, one causal mechanism not proven',
        'source_and_data_freeze_verified':True,'all_original_candidates_retained':True,
        'six_source_and_held_RGB_cost_statistics_exact_on_independent_fixed_view_replay':True,
        'GT_training_assets_or_pixel_cost_arrays_exported':False,'Gaussian_training_started':False,'real_training_authorized':False,
        'native86_188_PLY_video_or_optimizer_recovery_claim':False,
        'next':'Single preregistered S1-D6 RGB local-plane support warp, data-derived slopes, unchanged strict gates; no D5 rerun',
        'evidence_sha256':{name:sha(REPORT/name) for name in ['source_freeze.json','tests.log','measurement.json','analytic_controls.json','independent_audit.json','exports.json','assessment.md']},
        'finalizer_sha256':sha(Path(__file__)),'reviewed_at_utc':datetime.now(timezone.utc).isoformat()}
    assert not (REPORT/'final_review.json').exists(); write(REPORT/'final_review.json',final)
    write(RUNS/'final_status.json',{'status':status,'review_sha256':sha(REPORT/'final_review.json'),'pending_review':False,
        'views':192,'next_stage':'S1-D6','Gaussian_or_real_training':False})
    write(RUNS/'pipeline.json',{'status':'complete','measurement':'complete','independent_replay':'passed',
        'manual_review':'complete','final_review_sha256':sha(REPORT/'final_review.json')})
    write(RUNS/'state.json',{'status':status,'views':192,'GPU_process_required':False,'pending_review':False})
    print('D5_FINAL_REVIEW_COMPLETE',sha(REPORT/'final_review.json'),flush=True)


if __name__=='__main__': main()
