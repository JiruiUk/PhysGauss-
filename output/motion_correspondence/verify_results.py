"""Independent checks of frozen inputs, association receipts and render support."""
from pathlib import Path
import sys,json
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from variants.motion_correspondence.run import REPORT,OUT,read,write,sha,observations,VARIANTS
from variants.motion_correspondence.model import MotionTracker


def main():
    lock=read(REPORT/'input_lock.json')
    assert all(sha(ROOT/p)==value for p,value in lock.items())
    initial=REPORT/'initial_probe';initial_lock=read(initial/'experiments/input_lock.json')
    for original,archived in [('experiments/motion_correspondence/plan.json','experiments/plan.json'),
        ('variants/motion_correspondence/model.py','variants/model.py'),('variants/motion_correspondence/run.py','variants/run.py'),
        ('tests/test_motion_correspondence.py','test_motion_correspondence.py')]:
        key=next(k for k in initial_lock if k.replace('\\','/')==original)
        assert sha(initial/archived)==initial_lock[key]
    assert read(REPORT/'manual_review.json')['review_complete']
    controlled=read(REPORT/'controlled.json');summary={}
    assert len(controlled['trials'])==36 and controlled['qualified_reference_frames']==25
    assert controlled['reference_is_pseudo_not_material_GT'] and controlled['forecast_computed_before_reappearance_observation']
    for name in VARIANTS:
        errors=[e for row in controlled['trials'] for e in row['variants'][name]['forecast_pixel_errors']]
        outcomes=[o for row in controlled['trials'] for o in row['variants'][name]['outcomes']]
        correct=sum(o['correct'] is True for o in outcomes);wrong=sum(o['correct'] is False for o in outcomes)
        assert len(outcomes)==72 and len(errors)==72
        result=controlled['summary'][name]
        assert result['correct']==correct and result['wrong']==wrong and result['abstained']==72-correct-wrong
        assert abs(result['forecast_error_mean_pixels']-np.mean(errors))<1e-10
        summary[name]={'correct':correct,'wrong':wrong,'unknown':72-correct-wrong,'forecast_mean_pixels':float(np.mean(errors))}
    synthetic=read(REPORT/'synthetic.json')
    assert synthetic['hidden_pass']['history']['accepted_correct']==2
    assert synthetic['hidden_bounce']['history']['accepted_wrong']==2
    for scene in ['hidden_pass','hidden_bounce']:
        guarded=synthetic[scene]['history_guarded']
        assert guarded['accepted']==[-1,-1]
        assert len(guarded['identical_observation_sequel'])==8
        assert all(row['accepted']==[-1,-1] for row in guarded['identical_observation_sequel'])
    rolling=read(REPORT/'rolling_results.json');rolling_lock=read(REPORT/'rolling_input_lock.json')
    assert all(sha(ROOT/p)==value for p,value in rolling_lock.items())
    assert len(rolling['trials'])+len(rolling['excluded_windows'])==38
    assert rolling['qualified_windows']==len(rolling['trials'])
    assert len({(r['start_index'],r['hidden_observations']) for r in rolling['trials']+rolling['excluded_windows']})==38
    for stratum in ['all','early','middle','late']:
        subset=[row for row in rolling['trials'] if stratum=='all' or row['stratum']==stratum]
        for name in VARIANTS:
            errors=[];outcomes=[]
            for row in subset:
                v=row['variants'][name]
                distances=np.linalg.norm(np.array(v['forecast_before_observation'])-np.array(v['reference_xy']),axis=1)
                assert np.allclose(distances,v['forecast_errors_pixels'],rtol=0,atol=1e-10)
                errors.extend(distances.tolist());outcomes.extend(v['outcomes'])
            result=rolling['summary'][stratum][name]
            assert result['windows']==len(subset) and result['cases']==len(outcomes)
            assert result['correct']==sum(o['correct'] is True for o in outcomes)
            assert result['wrong']==sum(o['correct'] is False for o in outcomes)
            assert result['unknown']==sum(o['correct'] is None for o in outcomes)
            assert not errors or abs(result['mean_forecast_pixels']-np.mean(errors))<1e-10
    z=np.load(ROOT/'output/surface_support/image_tracks.npz');frames=z['frames'];components=z['components']
    natural={}
    for name in VARIANTS:
        report=read(REPORT/(name+'.json'));labels=np.load(OUT/(name+'.npz'))
        assert len(report['frames'])==120 and np.array_equal(labels['frames'],frames)
        assert labels['labels'].shape==(120,225,400)
        prior_unresolved=set()
        for index,row in enumerate(report['frames']):
            assignment=row['accepted_assignment'];positive=[j for j in assignment if j>=0]
            assert len(positive)==len(set(positive)) and row['frame']==frames[index]
            expected=np.full((225,400),-1,dtype=np.int8)
            obs=observations(components[index])
            for identity,j in enumerate(assignment):
                if j>=0:expected[components[index]==obs[j].component]=identity
            assert np.array_equal(labels['labels'][index],expected)
            unresolved=set(row.get('unresolved_seed_identities',[]))
            if name=='history_guarded':
                assert prior_unresolved<=unresolved
                assert all(assignment[i]<0 for i in unresolved)
                prior_unresolved=unresolved
        natural[name]={'frames':120,'accepted_candidate_events':sum(j>=0 for row in report['frames'] for j in row['accepted_assignment']),
                       'final_unresolved_seed_IDs':sorted(prior_unresolved),'material_accuracy':None}
    # Real observation sequence, checkpoint just before first merge, followed
    # by JSON round-trip and exact all-hypothesis replay.
    tracker=MotionTracker(observations(components[0]),(int(frames[0])-1)/199,VARIANTS['history_guarded'])
    for index in range(1,25):tracker.step(observations(components[index]),(int(frames[index])-1)/199)
    state=tracker.state_dict();write(REPORT/'motion_state_frame0042.json',state)
    restored=MotionTracker.from_state_dict(read(REPORT/'motion_state_frame0042.json'))
    rows=read(REPORT/'history_guarded.json')['frames']
    for index in range(25,45):
        value=restored.step(observations(components[index]),(int(frames[index])-1)/199);value['frame']=int(frames[index])
        assert value==rows[index]
    render=read(REPORT/'render_support.json')
    assert render['read_only'] and render['new_Gaussian_optimizer_steps']==0 and render['full_compositor_not_first20']
    assert all(sha(ROOT/p)==value for p,value in render['source_sha256'].items())
    expected_frames=[2,42,43,55,63,74,86,108,134,172,200]
    assert [r['frame'] for r in render['frames']]==expected_frames
    for row in render['frames']:
        assert 0<=row['all_early_patch_full_alpha_share_on_yellow']<=1.00001
        if row['frame']>=86:assert row['all_early_patch_full_alpha_share_on_yellow']==0
        for candidate in row['per_candidate']:
            value=candidate['early_patch_full_alpha_share_on_candidate']
            assert value is None or 0<=value<=1.00001
        with Image.open(OUT/'render_support'/f"frame{row['frame']:04d}.png") as im:
            im.load();assert im.size==(2400,265)
    from output.c1_flow_ablation.reuse_control import control_hashes
    protected=read(ROOT/'experiments/c1_flow_ablation/reuse_control_plan.json')['historical_control_sha256']
    assert control_hashes()==protected
    artifacts={str(p.relative_to(ROOT)):sha(p) for base in [REPORT,OUT] for p in base.rglob('*')
        if p.is_file() and p.suffix in ['.json','.npz','.png','.html','.py','.md']
        and 'initial_probe' not in p.parts and p.name not in ['independent_verification.json','manual_review.json']}
    write(REPORT/'independent_verification.json',{'passed':True,'locked_input_files':len(lock),'input_SHA_unchanged':True,
        'historical_C1_protection_passed':True,'controlled_recalculated':summary,'synthetic_delayed_ambiguity_passed':True,
        'initial_failed_probe_source_archive_SHA_verified':True,'representative_manual_review_completed':True,
        'rolling_windows_independently_recalculated':len(rolling['trials']),'rolling_local_pseudo_cases':len(rolling['trials'])*4,
        'natural':natural,'real_JSON_checkpoint_replay_frames':20,'all_candidate_masks_one_to_one_checked':True,
        'full_compositor_frames':expected_frames,'early_patch_zero_yellow_support_from_reviewed_frame86':True,
        'training_optimizer_steps':0,'GPU_training_smoke':'Not applicable: non-neural read-only association module; real GPU compositor smoke completed',
        'flags_off':'Exact delegation unit test; no baseline training/inference source changed',
        'physical_identity_accuracy':None,'full_reconstruction_integration_gate_passed':False,
        'gate_reason':'Natural material identities unverified; early canonical patches still lose visible duck support. Motion history alone does not resolve hidden symmetric interactions.',
        'artifact_sha256':artifacts,'manual_review_receipt':'manual_review.json'})
    print(json.dumps({'passed':True,'controlled':summary,'natural':natural,'locked_inputs':len(lock)}))


if __name__=='__main__':main()
