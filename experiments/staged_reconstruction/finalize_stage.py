"""Finalize only after a separate actual image-review receipt exists."""
import argparse
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from variants.staged_reconstruction.prepare_data import REPORT,RUNS,read,write,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['S3','S4','S5','S6'],required=True);a=p.parse_args()
    stage=a.stage;report=REPORT/stage;checkpoint=RUNS/stage/'formal/checkpoint.pt'
    audit=read(report/'state_objective_audit.json');numeric=read(report/f'{stage}_review_numeric.json')
    exports=read(report/f'{stage}_exports.json');manual=read(report/'manual_visual_review.json')
    actual=sha(checkpoint)
    assert actual==audit['checkpoint_sha256']==numeric['checkpoint_sha256']==exports['checkpoint_sha256']==manual['checkpoint_sha256']
    assert manual['native_images_reviewed'] and manual['files_reviewed'] and manual['findings']
    for p,digest in manual['files_reviewed'].items():assert sha(ROOT/p)==digest
    assert len(exports['videos'])==3 and all(x['decoded_frames']==200 for x in exports['videos'])
    admitted=all(audit['numeric_gates'].values()) and audit['all_fixed_frame_physics_valid'] and manual['approve_conditional_continuation']
    result={**audit,'native_images_reviewed':True,'manual_visual_review_sha256':sha(report/'manual_visual_review.json'),
        'geometry_success_established':False,'physical_ID_after_ambiguity':'unknown','contact_surface_certified':False,
        'videos_independently_decoded':True,'admitted_for_conditional_continuation':admitted,
        'visual_findings':manual['findings'],'complete_physical_contact_inference':False}
    write(report/'final_verification.json',result)
    write(report/'admission.json',{'admitted':admitted,'checkpoint_sha256':actual,'native_images_reviewed':True,
        'scope':'Conditional sequential reconstruction corrections only; no certified complete geometry, physical re-ID or contact forces',
        'numeric_gates':audit['numeric_gates'],'independent_physics_GT':False,'all_fixed_frame_internal_physics_valid':audit['all_fixed_frame_physics_valid'],
        'geometry_success_established':False,'contact_surface_certified':False,'physical_ID_after_ambiguity':'unknown'})
    m=audit['native_validation'];test=audit['native_test']
    lines=[f'# {stage} completed measurements', '',
        f"Completed {audit['iteration']} actual updates; native1600x900 val/test, full-state/sampling audit, native image review and three200-frame video decode checks completed.", '',
        f"Validation full PSNR {m['PSNR']:.6f}dB, dynamic {m['dynamic_PSNR']:.6f}dB, dynamic-only alpha {m['dynamic_alpha']:.6f}, static {m['static_PSNR']:.6f}dB. Test dynamic {test['dynamic_PSNR']:.6f}dB.", '',
        f"Unchanged numeric gates: {audit['numeric_gates']}. Exact Python RNG/camera replay: {audit['sampler']['exact_python_RNG_and_final_camera_stack']}. Static geometry exactly matches S1: {audit['static_geometry_exact_S1']}. All fixed-frame internal physics stencils valid: {audit['all_fixed_frame_physics_valid']}.", '',
        'Actual visual findings:', '',*['- '+x for x in manual['findings']], '',
        'Manual review scope is the named native observed/group and fixed-view files in manual_visual_review.json. Every exported video frame was independently decoded, but not every video frame was manually watched.', '',
        f"Qualified track mean projection error at400px: {audit['track_observation_mean_px400']}. Conditional interior diagnostic: {audit['conditional_interior']}.", '',
        f"Whole-run conditional-interior activation: {audit.get('training_conditional_interior')}.", '',
        'These are training-image correspondences and internal model/conditional-static proxies. They are not independent geometry, material velocity, physical stress or contact GT. Acquisition seconds and late physical identities remain unknown. Novel-view geometry limitations are retained; a successful numeric gate is not geometry success.', '',
        f"Conditional continuation admission: {admitted}. Contact force/friction/restitution, constitutive inference and unknown forces remain off."]
    (report/'assessment.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print('STAGE_FINALIZED',stage,'conditional_admission',admitted)


if __name__=='__main__':main()
