"""Read-only evaluation of the existing fixed-camera temporal sequences."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from variants.view_completeness.metrics import evaluate, summarize

REPORT = ROOT/'experiments/view_completeness'
SOURCE = ROOT/'output/fixed_camera_temporal'
MODELS = ['C1_original_50k', 'T1_association_52k', 'O1_flow_on_50k']
VIEWS = ['fixed_0002', 'fixed_0086', 'fixed_0188', 'moving_control']


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--enabled', action='store_true')
    args = parser.parse_args()
    if not args.enabled:
        print('Fixed-view completeness metrics disabled; no input reads or writes')
        return
    plan = read(REPORT/'plan.json')
    render_proof_path = ROOT/'experiments/fixed_camera_temporal/render_verification.json'
    render_proof = read(render_proof_path)
    prior_artifacts = read(ROOT/'experiments/fixed_camera_temporal/artifact_manifest.json')
    def checked(path):
        h = sha(path)
        key = str(path.relative_to(ROOT)).replace('\\', '/')
        if key.startswith('output/fixed_camera_temporal/'):
            # The earlier Windows manifest uses native separators.
            original = prior_artifacts.get(key, prior_artifacts.get(str(path.relative_to(ROOT))))
            assert original == h, f'Frozen source artifact differs: {path}'
        inputs[str(path.relative_to(ROOT))] = h
        return path
    inputs = {}
    checkpoint_hashes = render_proof['checkpoint_sha256_unchanged']
    for path, digest in checkpoint_hashes.items():
        assert sha(Path(path)) == digest
    for path in [Path(__file__), Path(__file__).with_name('metrics.py'), REPORT/'plan.json', render_proof_path]:
        checked(path)
    val_meta_path = ROOT/'data/derived/kinematic_resplit/transforms_val.json'
    val = read(checked(val_meta_path))
    val_frames = {int(Path(f['file_path']).stem) for f in val['frames']}
    assert len(val_frames) == 30
    masks = {}
    for n in range(1, 201):
        path = checked(ROOT/f'data/PIDG/rubber_duck/train/{n:04d}.png')
        with Image.open(path) as image:
            masks[n] = np.asarray(image.convert('RGBA').getchannel('A').resize((400, 225), Image.Resampling.BOX))/255
    all_summary = {}
    observed_summary = {}
    for name in MODELS:
        rows = {}
        for view in VIEWS:
            samples = []
            for n in range(1, 201):
                path = checked(SOURCE/name/view/'small'/f'alpha{n:04d}.png')
                with Image.open(path) as image:
                    alpha = np.asarray(image.convert('L'), dtype=np.float64)/255
                truth = masks[n] if view == 'moving_control' or n == int(view[-4:]) else None
                record = evaluate(alpha, truth, enabled=True, **plan['proxy_settings'])
                assert record['full_object_completeness_score'] is None
                assert record['independent_alpha_target_available'] == (truth is not None)
                samples.append({'frame': n, **record})
            rows[view] = samples
            summaries = {'all200': summarize(samples),
                         'validation30_times': summarize([r for r in samples if r['frame'] in val_frames])}
            all_summary[f'{name}/{view}'] = summaries
            if view == 'moving_control':
                observed_summary[name] = {}
                for cohort, subset in [('all200', samples), ('validation30', [r for r in samples if r['frame'] in val_frames])]:
                    observed_summary[name][cohort] = {}
                    for metric in samples[0]['observed_coverage']:
                        values = [r['observed_coverage'][metric] for r in subset if r['observed_coverage'][metric] is not None]
                        observed_summary[name][cohort][metric] = float(np.mean(values)) if values else None
            print(name, view, '200 frames checked', flush=True)
        write(REPORT/f'{name}_metrics.json', rows)
    write(REPORT/'summary.json', all_summary)
    write(REPORT/'observed_coverage_summary.json', observed_summary)
    # Verify the consumed saved renders and GT remain unchanged. This audit
    # never opens a network/model for training and never writes to SOURCE.
    assert all(sha(ROOT/p) == h for p, h in inputs.items())
    assert all(sha(Path(p)) == h for p, h in checkpoint_hashes.items())
    write(REPORT/'input_sha256.json', inputs)
    write(REPORT/'verification.json', {
        'status': 'passed', 'model_count': 3, 'pose_count_with_moving_control': 4,
        'time_count': 200, 'validation_time_count': 30,
        'alpha_maps_evaluated': 2400, 'GT_RGBA_inputs': 200,
        'full_object_completeness_score_always_null': True,
        'grounded_metrics_only_at_actual_camera_time_pairs': True,
        'input_files_unchanged': len(inputs), 'optimizer_updates': 0,
        'checkpoint_sha256_unchanged': checkpoint_hashes,
        'training_runtime_or_physics_modified': False,
        'resolution': [400, 225], 'original_render_resolution': [1600, 900],
        'alpha_precision': 'Existing8-bit BOX-downsampled native alpha; not the original float-alpha arrays'})
    print('COMPLETED completeness metrics; all source artifacts/checkpoints unchanged', flush=True)


if __name__ == '__main__':
    main()
