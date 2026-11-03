"""Verify the completed cold-start experiment and its reviewed evidence."""
from pathlib import Path
import json
import math
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from variants.metric_tuning.runtime import read_json, write_json, sha256
from variants.scratch_integrity.run import NAMES, freeze, prepare_code
from variants.scratch_integrity.prefix import REPORT, RUNS, plan, freeze as prefix_freeze

OUT = ROOT / 'output/scratch_integrity'


def check_hashes(values, base=ROOT):
    for name, digest in values.items():
        assert sha256(base / name) == digest, name


def main():
    freeze(prepare_code())
    prefix_freeze(ROOT / 'variants/color_repair/code')
    previous = read_json(REPORT / 'training_freeze_attempt1.json')
    current = read_json(REPORT / 'training_freeze.json')
    changed = [p for p, h in previous['own_and_inherited'].items() if h != current['own_and_inherited'][p]]
    assert changed == ['variants/scratch_integrity/run.py']
    assert all(previous[k] == current[k] for k in ['fork', 'C1_parent', 'image_evidence'])
    assert sha256(REPORT / 'preflight_attempt1/run.py') == previous['own_and_inherited'][changed[0]]
    assert '35 passed' in (REPORT / 'tests_final.log').read_text()
    assert read_json(REPORT / 'preflight.json')['status'] == 'passed'
    assert read_json(REPORT / 'optimizer_probe.json')['status'] == 'passed'
    check_hashes(read_json(REPORT / 'evaluation_source_lock.json'))
    check_hashes(read_json(REPORT / 'history_before.json'))
    historical = read_json(ROOT / 'experiments/fixed_camera_temporal/artifact_manifest.json')
    check_hashes(historical)
    provenance = read_json(REPORT / 'GT_provenance.json')
    assert sha256(ROOT / provenance['historical_provenance_path']) == provenance['historical_provenance_sha256']
    check_hashes(provenance['split_metadata_sha256'])
    gt = provenance['original_RGBA_sha256']
    assert len(gt) == 200
    check_hashes(gt)
    initial = read_json(REPORT / 'initialization.json')
    assert initial['iteration'] == initial['optimizer_updates'] == 0
    assert initial['pretrained_model_used'] is False
    assert sha256(RUNS / 'random_initial_state/resume.pt') == initial['random_initial_checkpoint_sha256']
    binding = read_json(REPORT / 'data_binding.json')
    assert sha256(Path(binding['path'])) == binding['sha256']
    assert sha256(Path(binding['source_directory']) / 'checkpoints/resume.pt') == binding['source_checkpoint_sha256']
    assert binding['source_iteration'] in plan()['common_prefix_candidates']
    assert binding['historical_Gaussian_row_ids_used'] is False
    training = read_json(REPORT / 'training_verification.json')
    assert training['actual_total_updates_each'] == 50000
    assert training['only_main_flag_differs'] and training['row_topology_preserved_after_binding']
    assert training['same_post_prefix_camera_samples'] == 50000 - binding['source_iteration']
    samples = []
    for name in NAMES:
        directory = RUNS / 'pilot' / name / 'rubber_duck'
        assert read_json(directory / 'checkpoints/resume.json')['iteration'] == 50000
        assert sha256(directory / 'checkpoints/resume.pt') == training['states'][name]['sha256']
        samples.append([json.loads(line) for line in (directory / 'camera_sampling.jsonl').read_text().splitlines()])
    assert samples[0] == samples[1]
    assert [row['iteration'] for row in samples[0]] == list(range(binding['source_iteration'] + 1, 50001))
    assert not (RUNS / 'pilot' / NAMES[0] / 'rubber_duck/transfer_loss.jsonl').exists()
    loss = RUNS / 'pilot' / NAMES[1] / 'rubber_duck/transfer_loss.jsonl'
    losses = [json.loads(line) for line in loss.read_text().splitlines()]
    assert [row['iteration'] for row in losses] == list(range(binding['source_iteration'] + plan()['interval'], 50001, plan()['interval']))
    assert all(math.isfinite(row['weighted_loss']) for row in losses)
    for name, record in read_json(REPORT / 'training_evidence_files.json').items():
        assert sha256(ROOT / name) == record['sha256'] == sha256(ROOT / record['source'])
    render = read_json(REPORT / 'render_verification.json')
    check_hashes(render['checkpoint_sha256_unchanged'])
    check_hashes(render['source_sha256_unchanged'])
    adapter = read_json(REPORT / 'evaluation_adapter.json')
    assert adapter['wrapper_sha256'] == sha256(REPORT / 'evaluate.py')
    assert adapter['measurement_helper_sha256'] == sha256(ROOT / 'experiments/transfer_integrity/evaluate.py')
    assert adapter['native_renderer_helper_sha256'] == sha256(ROOT / 'output/fixed_camera_temporal/render_sequences.py')
    export = read_json(REPORT / 'export_verification.json')
    assert export['adapter_sha256'] == sha256(REPORT / 'export.py')
    assert export['exporter_sha256'] == sha256(ROOT / 'experiments/transfer_integrity/export.py')
    for name in ['C1_original_50k', *NAMES]:
        for split, count in [('val', 30), ('test', 50)]:
            metric = read_json(REPORT / f'{name}_{split}.json')
            assert metric['count'] == count and metric['full_state']['iteration'] == 50000
            assert metric['adapter_sha256'] == sha256(REPORT / 'evaluate.py')
            assert metric['evaluator_sha256'] == adapter['measurement_helper_sha256']
        result = read_json(REPORT / f'{name}_results.json')
        assert result['fixed_camera_fingerprints_before'] == result['fixed_camera_fingerprints_after']
        for view in ['fixed_0002', 'fixed_0086', 'fixed_0188', 'moving_control']:
            for channel in ['rgb', 'alpha']:
                files = list((OUT / name / view / 'native').glob(f'{channel}*.png'))
                assert len(files) == 8, (name, view, channel)
                small = list((OUT / name / view / 'small').glob(f'{channel}*.png'))
                assert len(small) == 200, (name, view, channel)
    support = read_json(REPORT / 'support_summary.json')
    assert support['own_fresh_binding_sha256'] == binding['sha256']
    assert support['adapter_sha256'] == sha256(REPORT / 'support.py')
    for name in ['common_prefix', *NAMES]:
        assert read_json(REPORT / f'{name}_support.json')['evaluator_sha256'] == sha256(ROOT / 'variants/association_training/evaluate.py')
    videos = read_json(REPORT / 'video_verification.json')
    assert len(videos) == 16
    for name, value in videos.items():
        assert value['frames'] == 200 and sha256(OUT / name) == value['sha256']
    review = read_json(REPORT / 'manual_review.json')
    assert review['decoded_video_contact_inspected'] is True
    assert len(review['reviewed_artifacts']) >= 8
    check_hashes(review['reviewed_artifacts'], OUT)
    assert review['physical_ID_confirmed'] is False
    inspection = read_json(REPORT / 'inspection_provenance.json')
    check_hashes(inspection['source_sha256'], OUT)
    assert inspection['script_sha256'] == sha256(REPORT / 'inspection.py')
    from PIL import Image
    for frame in [2, 86, 200]:
        with Image.open(OUT / f'native_crop_fixed86_t{frame}.png') as panel:
            for column, name in enumerate(NAMES):
                with Image.open(OUT / name / 'fixed_0086/native' / f'rgb{frame:04d}.png') as image:
                    assert panel.crop((650 * column, 27, 650 * (column + 1), 607)).tobytes() == image.crop(tuple(inspection['crop_native_pixel_box'])).tobytes()
    assessment = REPORT / 'assessment.md'
    assert assessment.exists() and (REPORT / 'numeric_summary.json').exists()
    gallery = (OUT / 'index.html').read_text(encoding='utf-8')
    links = re.findall(r'(?:href|src)="([^"]+)"', gallery)
    assert all((OUT / name).resolve().exists() for name in links)
    check_hashes(read_json(REPORT / 'artifact_manifest.json'))
    evidence = [p for p in REPORT.iterdir() if p.is_file() and p.name not in ['final_verification.json', 'finalize.log']]
    write_json(REPORT / 'final_verification.json', {
        'status': 'completed_evaluated', 'conclusion': review['finding'],
        'activation_iteration': binding['source_iteration'], 'final_iteration_both': 50000,
        'fresh_initial_state_verified': True, 'same_post_prefix_training_camera_samples': len(samples[0]),
        'historical_render_files_verified_unchanged': len(historical),
        'historical_checkpoints_reports_unchanged': True, 'all_200_original_GT_files_unchanged': True,
        'training_source_and_binding_freeze_unchanged': True,
        'validation_count_per_model': 30, 'test_count_per_model': 50,
        'native_renders': 2400, 'videos_decoded': 16,
        'novel_camera_time_GT_available': False, 'physical_ID_after_occlusion_verified': False,
        'gallery_local_links_checked': len(links),
        'evidence_sha256': {p.name: sha256(p) for p in evidence}})
    state = read_json(RUNS / 'controller.json')
    state.update(status='completed_evaluated', assessment=str(assessment), conclusion=review['finding'])
    write_json(RUNS / 'controller.json', state)
    print('FINAL VERIFIED: cold initialization, matched 50k arms, historical files, GTs, renders, 16 videos and gallery links', flush=True)


if __name__ == '__main__':
    main()
