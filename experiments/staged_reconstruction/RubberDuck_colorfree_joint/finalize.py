"""Record the actually inspected J7 result; never promote or start another run."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
from variants.staged_reconstruction.colorfree_joint.runtime import BASE,REPORT,RUNS,OUT
from variants.staged_reconstruction.prepare_data import ROOT, read, sha, write


def main():
    p=argparse.ArgumentParser();p.add_argument('--enable-colorfree-finalize',action='store_true')
    if not p.parse_args().enable_colorfree_finalize:raise SystemExit('Finalization is opt-in')
    assert not (REPORT/'final_review.json').exists()
    ev=read(REPORT/'evaluation.json');v=read(REPORT/'final_validation.json');assert v['passed']
    assert sha(BASE/'final_verify.py')==v['verification_source_sha256']
    for receipt in [REPORT/'presentation.json']:
        q=read(receipt);assert sha(BASE/'present.py')==q['source_sha256']
        for key in ['inputs','outputs']:
            for name,h in q[key].items():assert sha(ROOT/name)==h
    viewed=[]
    for cam in [86,188]:
        for t in [2,100,200]:
            viewed += [OUT/f'c{cam:03d}_t{t:03d}_{kind}.png' for kind in ['joint_comparison','decomposition']]
            viewed += [OUT/f'c{cam:03d}_t{t:03d}_colorfree_static_jointRGB_native.png']
        viewed += [OUT/f'c{cam:03d}_t002_colorfree_static_{kind}_native.png' for kind in ['staticRGB','staticAlpha']]
        viewed += [ROOT/f'output/staged_reconstruction/S1_clean_visibility/v1/fixed{cam}_control18k_time0100.png']
    viewed += [OUT/'c086_t002_colorfree_static_staticD-alpha_native.png']
    viewed += [OUT/f'observed_train_index{i:03d}.png' for i in [0,24,48,72,96,119]]
    viewed += [OUT/f'{kind}_before_after.png' for kind in ['static','joint']]
    videos=read(REPORT/'videos.json')['videos']
    viewed += [ROOT/q['decoded_contact'] for q in videos.values()]
    assert len(viewed)==len(set(viewed))==35
    review=dict(review_method='Assistant actually viewed native PNGs, comparison/decomposition sheets and decoded contacts',
        images={f.relative_to(ROOT).as_posix():sha(f) for f in viewed},images_viewed=35,
        source_static_displayed_before_formal_training=True,
        source_static_no_blue_threshold=True,dynamic_historical_color_lineage=True,
        static_main_bodies_retained_in_reviewed_views=True,
        static_RGB_residual_reduction_observed=True,static_alpha_floating_support_remains=True,
        dynamic_fragmentation_and_dark_high_alpha_trails_remain=True,
        consistent_dynamic_cleanup=False,overall_completeness_improvement=False,
        manual_geometry_cleanup_gate=False,independent_fixed_view_geometry_GT=False,
        observations=['Static pillars/bases remain and most yellow/color halo RGB is reduced after joint training.',
            'Static alpha still has broad floating footprints/spikes; darker RGB does not certify geometry removal.',
            'c86/t200 retains dark high-alpha oval support and particle streaks; c188/t100 remains fragmented with horizontal tails.',
            'Some dynamic alpha footprints widen; train-trajectory RGB cannot replace independent same-time multiview GT.',
            'Both decoded video contacts were viewed; whole video human playback and browser playback are not claimed.'])
    write(REPORT/'manual_review.json',review)
    status='completed_rgb_improved_not_geometry_promoted'
    result=dict(stage='J7',status=status,pending=False,completed_utc=datetime.now(timezone.utc).isoformat(),
        human_requested_display_fusion_and_test_completed=True,formal_updates=4000,updates_per_arm=2000,
        RGB_gate=ev['validation']['PSNR_nonregression_gate'],manual_geometry_cleanup_gate=False,
        retained_model=read(BASE/'protocol.json')['parent'],retained_model_sha256=read(BASE/'protocol.json')['parent_sha256'],
        candidate_retained_as_appearance_result=True,automatic_promotion=False,automatic_extension=False,
        candidate_checkpoint=(RUNS/'formal/colorfree_static/checkpoint.pt').relative_to(ROOT).as_posix(),
        candidate_sha256=sha(RUNS/'formal/colorfree_static/checkpoint.pt'),training_process_exited=True,
        no_new_physics_or_object_identity=True,independent_geometry_GT=False,
        receipts={name:sha(REPORT/name) for name in ['evaluation.json','videos.json','final_validation.json','manual_review.json','assessment.md','presentation.json']},
        finalization_source_sha256=sha(Path(__file__)))
    assert result['RGB_gate']
    write(REPORT/'final_review.json',result);write(REPORT/'final_status.json',result)
    write(RUNS/'final_status.json',result);write(RUNS/'active.json',dict(status=status,pending=False,training_process_exited=True,formal_updates=4000))
    pipeline=read(RUNS/'pipeline.json');pipeline.update(status=status,pending=False,final_review_sha256=sha(REPORT/'final_review.json'))
    write(RUNS/'pipeline.json',pipeline)
    for arm in read(BASE/'protocol.json')['arms']:
        write(RUNS/'formal'/arm/'final_status.json',dict(status=status if arm=='colorfree_static' else 'completed_reviewed_control',iteration=2000,pending=False))
    print('J7_REVIEW_FINALIZED_RGB_IMPROVED_GEOMETRY_NOT_PROMOTED',flush=True)


if __name__=='__main__':main()
