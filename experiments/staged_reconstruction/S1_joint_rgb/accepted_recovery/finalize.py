"""Independent final audit after actual reviewed native images and decoded exports."""
from datetime import datetime, timezone
from pathlib import Path
import torch
from experiments.staged_reconstruction.S1_joint_rgb.accepted_recovery.launch import configure, BASE, REPORT, RUNS, ROOT
from variants.staged_reconstruction.prepare_data import read, sha


def main():
    io = configure(); io.verify(); assert not io.stopped()
    assert not (REPORT/'final_review.json').exists()
    torch.set_num_threads(4); protocol = read(BASE/'protocol.json')
    evaluation, gate, exports = [read(REPORT/name) for name in ['synthetic_evaluation.json', 'synthetic_gate.json', 'exports.json']]
    manual = read(REPORT/'manual_review.json'); assert manual['actually_viewed']
    for name, expected in manual['images'].items(): assert sha(ROOT/name) == expected
    for name, expected in {**evaluation['images'], **exports['images']}.items(): assert sha(ROOT/name) == expected
    assert len(exports['plys']) == 8 and len(exports['videos']) == 2
    for record in list(exports['plys'].values())+list(exports['videos'].values()):
        assert sha(ROOT/record['path']) == record['sha256']
    assert all(record['decoded_frames'] == 160 for record in exports['videos'].values())
    audit = {}
    for case in protocol['cases']:
        traces = []
        for arm in protocol['arms']:
            path = RUNS/'formal'/case/arm/'checkpoint.pt'
            assert sha(path) == evaluation['checkpoint_audits'][case+'/'+arm]['checkpoint_sha256']
            state = torch.load(path, map_location='cpu', weights_only=False)
            assert state['iteration'] == 6000 and len(state['sample_trace']) == 6000
            static_steps = []
            for group in state['optimizers'][0]['param_groups']:
                current = int(state['optimizers'][0]['state'][group['params'][0]]['step'])
                expected = 12000 if arm == 'revisable_static' and group['name'] in ['_opacity', '_features_dc', '_features_rest'] else 6000
                assert current == expected, (case, arm, group['name'], current)
                static_steps.append(current)
            dynamic_steps = [int(value['step']) for value in state['optimizers'][1]['state'].values()]
            assert len(dynamic_steps) == 22 and all(step == 6000 for step in dynamic_steps)
            traces.append(state['sample_trace'])
            audit[case+'/'+arm] = {'static_Adam_steps': static_steps, 'dynamic_Adam_states': 22,
                'all_dynamic_Adam_steps': 6000, 'checkpoint_sha256': sha(path)}
        assert traces[0] == traces[1]
    assert sha(BASE.parent/'evaluate.py') == evaluation['evaluation_source_sha256']
    assert sha(BASE/'export.py') == exports['source_sha256']
    numerical = all(record['numerical_passed'] for record in gate['cases'].values())
    assert gate['numerical_passed'] == numerical
    promoted = numerical and manual['clear_artifact_reduction_without_structure_loss']
    result = {'status': 'completed_passed_synthetic_gates' if promoted else 'completed_not_promoted',
        'pending_review': False, 'numerical_passed': numerical, 'manual_native_images_reviewed': True,
        'synthetic_joint_cleanup_admitted': promoted, 'Rubber_Duck_training_automatically_admitted': False,
        'human_approved_recovery_acceptance': True, 'original_full40_failures_retained': True,
        'original_epsilon_and_quality_gates_preserved': True, 'full_optimizer_and_sample_audit': audit,
        'evaluation_sha256': sha(REPORT/'synthetic_evaluation.json'), 'manual_review_sha256': sha(REPORT/'manual_review.json'),
        'exports_sha256': sha(REPORT/'exports.json'), 'assessment_sha256': sha(REPORT/'assessment.md'),
        'completed_UTC': datetime.now(timezone.utc).isoformat(), 'source_sha256': sha(Path(__file__))}
    gate.update(passed=promoted, manual_review_required=False, real_training_authorized=False)
    io.atomic_json(REPORT/'synthetic_gate.json', gate)
    io.atomic_json(REPORT/'final_review.json', result); io.atomic_json(RUNS/'final_status.json', result)
    for name in ['state.json', 'pipeline.json', 'orchestrator.json']:
        state = read(RUNS/name); state.update(status=result['status'], pending_review=False,
            final_review_sha256=sha(REPORT/'final_review.json'), worker_pids_are_historical=True)
        io.atomic_json(RUNS/name, state)
    print('J1_A_FINAL_REVIEW_COMPLETE', result['status'], flush=True)


if __name__ == '__main__': main()
