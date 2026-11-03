"""Independent final evidence audit; does not import training or change models."""
from pathlib import Path
import hashlib
import json
import math
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / 'experiments/static_pool'
RUNS = ROOT / 'runs/pidg_static_pool'
NAMES = ['H0_dynamic_control', 'H1_static_pool']


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def main():
    assert read(RUNS / 'pipeline.json')['status'] in {
        'evaluated_pending_manual_review', 'reviewed_not_promoted'}
    frozen = read(REPORT / 'training_freeze.json')
    source_count = 0
    for group, values in frozen.items():
        base = ROOT / {'fork': 'variants/static_pool/code',
                       'C1_parent': 'variants/color_repair/code'}.get(group, '')
        for path, expected in values.items():
            assert digest(base / path) == expected, path
            source_count += 1
    proof = read(REPORT / 'training_verification.json')
    binding = read(REPORT / 'binding.json')
    assert digest(Path(binding['path'])) == binding['sha256']
    with np.load(binding['path'], allow_pickle=False) as data:
        ids = torch.tensor(data['static_ids'].copy(), dtype=torch.long)
    cameras = []
    checkpoints = {}
    means = {}
    for name in NAMES:
        directory = RUNS / 'pilot' / name / 'rubber_duck'
        checkpoint = directory / 'checkpoints/resume.pt'
        assert digest(checkpoint) == proof['states'][name]['sha256']
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        opacity = state['gaussian_parameters']['_opacity'].sigmoid().flatten()
        static = opacity[ids]
        checkpoints[name] = {
            'sha256': digest(checkpoint),
            'selected_static_opacity_mean': float(static.mean()),
            'selected_static_opacity_median': float(static.median()),
            'selected_static_opacity_below_001_fraction': float((static < .01).float().mean()),
        }
        del state
        rows = [json.loads(line) for line in (directory / 'camera_sampling.jsonl').read_text().splitlines()]
        assert len(rows) == 35000
        assert [row['iteration'] for row in rows] == list(range(15001, 50001))
        cameras.append(rows)
        for split, count in [('val', 30), ('test', 50)]:
            report = read(REPORT / f'{name}_{split}.json')
            assert report['count'] == len(report['frames']) == count
            means[f'{name}/{split}'] = {}
            for key, expected in report['metrics'].items():
                actual = float(np.mean([row[key] for row in report['frames']]))
                assert math.isfinite(actual) and abs(actual - expected) < 1e-10, key
                means[f'{name}/{split}'][key] = actual
    assert cameras[0] == cameras[1]
    artifacts = read(REPORT / 'artifact_manifest.json')
    for path, expected in artifacts.items():
        assert digest(ROOT / path) == expected, path
    videos = read(REPORT / 'video_verification.json')
    assert len(videos) == 16 and all(row['frames'] == 200 for row in videos.values())
    review = read(REPORT / 'review_decision.json')
    crop_checks = 0
    for path, expected in review['review_image_sha256'].items():
        assert digest(ROOT / path) == expected
        view = Path(path).stem.removesuffix('_native_crops')
        with Image.open(ROOT / path) as sheet:
            for j, frame in enumerate(review['inspected_frames']):
                for k, name in enumerate(NAMES):
                    source = ROOT / 'output/static_pool' / name / view / 'native' / f'rgb{frame:04d}.png'
                    with Image.open(source) as native:
                        actual = np.asarray(sheet.crop((950*k, 630*j+30, 950*k+950, 630*j+630)))
                        assert np.array_equal(actual, np.asarray(native.crop(tuple(review['native_crop_box']))))
                        crop_checks += 1
    assert crop_checks == 18
    result = {
        'passed': True, 'frozen_source_input_hashes': source_count,
        'artifact_hashes': len(artifacts), 'matched_camera_updates': 35000,
        'full_checkpoints': checkpoints, 'recomputed_frame_means': means,
        'videos_fully_decoded_by_exporter': 16,
        'review_crop_hashes': len(review['review_image_sha256']),
        'review_native_crops_exact': crop_checks,
        'opacity_is_canonical_not_composited_contribution': True,
        'audit_script_sha256': digest(Path(__file__)),
    }
    (REPORT / 'final_audit.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
