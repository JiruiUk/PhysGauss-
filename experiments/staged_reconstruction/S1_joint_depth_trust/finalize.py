"""Independent completed-state audit; requires an actual native-image review record."""
from datetime import datetime, timezone
from pathlib import Path
import torch
from experiments.staged_reconstruction.S1_joint_depth_trust.launch import configure, BASE, REPORT, RUNS, ROOT
from variants.staged_reconstruction.prepare_data import read, sha
from variants.staged_reconstruction.joint_rgb.core import assert_equivalent


def main():
    io = configure(); io.verify(); assert not io.stopped()
    assert not (REPORT/'final_review.json').exists()
    torch.set_num_threads(4); protocol = read(BASE/'protocol.json')
    evaluation, gate, exports = [read(REPORT/name) for name in ['synthetic_evaluation.json', 'synthetic_gate.json', 'exports.json']]
    manual = read(REPORT/'manual_review.json'); assert manual['actually_viewed']
    for name, expected in {**manual['images'], **evaluation['images'], **exports['images']}.items():
        assert sha(ROOT/name) == expected, name
    assert len(exports['plys']) == 8 and len(exports['videos']) == 2
    for record in list(exports['plys'].values())+list(exports['videos'].values()):
        assert sha(ROOT/record['path']) == record['sha256']
    assert all(r['decoded_frames'] == 160 and r['all_160_frames_decoded'] for r in exports['videos'].values())
    audit = {}
    for case in protocol['cases']:
        traces, initials = [], []
        for arm in protocol['arms']:
            directory = RUNS/'formal'/case/arm; path = directory/'checkpoint.pt'
            assert sha(path) == evaluation['checkpoint_audits'][case+'/'+arm]['checkpoint_sha256']
            state = torch.load(path, map_location='cpu', weights_only=False)
            initial = torch.load(directory/'initial.pt', map_location='cpu', weights_only=False)
            assert state['iteration'] == 6000 and len(state['sample_trace']) == 6000 and initial['iteration'] == 0
            assert not any('teacher' in key or 'cache' in key for key in state['model'])
            steps = []
            for group in state['optimizers'][0]['param_groups']:
                step = int(state['optimizers'][0]['state'][group['params'][0]]['step'])
                expected = 12000 if arm == 'revisable_static' and group['name'] in ['_opacity', '_features_dc', '_features_rest'] else 6000
                assert step == expected, (case, arm, group['name'], step)
                steps.append(step)
            dynamic = [int(value['step']) for value in state['optimizers'][1]['state'].values()]
            assert len(dynamic) == 22 and all(value == 6000 for value in dynamic)
            traces.append(state['sample_trace']); initials.append(initial)
            if arm == 'frozen_static':
                receipt = read(directory/'inherited_control.json'); old = ROOT/receipt['exact_copy_from']
                assert sha(path) == sha(old/'checkpoint.pt')
                assert sha(directory/'initial.pt') == sha(old/'initial.pt')
            audit[case+'/'+arm] = {'static_Adam_steps': steps, 'dynamic_Adam_states': 22,
                'all_dynamic_Adam_steps': 6000, 'checkpoint_sha256': sha(path),
                'inherited_control_not_retrained': arm == 'frozen_static'}
        assert traces[0] == traces[1]
        for key in ['model', 'optimizers', 'rng_python', 'rng_numpy', 'rng_torch', 'rng_cuda', 'camera_stack', 'visits', 'sample_trace']:
            assert_equivalent(initials[0][key], initials[1][key], 0, 0, case+'/'+key)
    assert sha(ROOT/'experiments/staged_reconstruction/S1_joint_rgb/evaluate.py') == evaluation['evaluation_source_sha256']
    assert sha(ROOT/'experiments/staged_reconstruction/S1_joint_rgb/accepted_recovery/export.py') == exports['source_sha256']
    numerical = all(r['numerical_passed'] for r in gate['cases'].values())
    assert numerical == gate['numerical_passed']
    passed = numerical and manual['clear_artifact_reduction_without_structure_loss']
    result = {'status': 'completed_passed_synthetic_gates' if passed else 'completed_not_promoted',
        'pending_review': False, 'numerical_passed': numerical, 'manual_native_images_reviewed': True,
        'synthetic_joint_cleanup_admitted': passed, 'Rubber_Duck_training_automatically_admitted': False,
        'new_formal_training_updates': 12000, 'historical_matched_control_updates': 12000,
        'complete_initial_model_twoAdam_RNG_camera_and6000_sampling_match': True,
        'full_optimizer_and_sample_audit': audit, 'parent_self_reference_is_not_GT': True,
        'original_quality_gates_and_full40_failures_retained': True,
        'evaluation_sha256': sha(REPORT/'synthetic_evaluation.json'), 'exports_sha256': sha(REPORT/'exports.json'),
        'manual_review_sha256': sha(REPORT/'manual_review.json'), 'assessment_sha256': sha(REPORT/'assessment.md'),
        'completed_UTC': datetime.now(timezone.utc).isoformat(), 'source_sha256': sha(Path(__file__))}
    gate.update(passed=passed, manual_review_required=False, real_training_authorized=False)
    io.atomic_json(REPORT/'synthetic_gate.json', gate)
    io.atomic_json(REPORT/'final_review.json', result); io.atomic_json(RUNS/'final_status.json', result)
    for name in ['state.json', 'pipeline.json']:
        state = read(RUNS/name); state.update(status=result['status'], pending_review=False,
            final_review_sha256=sha(REPORT/'final_review.json'), worker_pids_are_historical=True)
        io.atomic_json(RUNS/name, state)
    print('J3_FINAL_REVIEW', result['status'], flush=True)


if __name__ == '__main__': main()
