"""Independent observation interpolation/threshold audit and fixed-support replay."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion, binary_closing, binary_fill_holes, binary_dilation
from variants.staged_reconstruction.prepare_data import read, write, sha, DATA
from variants.staged_reconstruction.surface_trajectory.prepare import OUT, PARENT, PARENT_SHA
from variants.staged_reconstruction.surface_trajectory.binding import choose_support
from variants.staged_reconstruction.surface_trajectory.split import group_tracks, split_groups
from variants.staged_reconstruction.surface_trajectory.admission import evidence_gates


def independent_sample(field, points):
    # OpenCV INTER_LINEAR uses 1/32-pixel interpolation tables. Implement its
    # four weights explicitly instead of reusing the tracking sampler.
    points = np.rint(np.asarray(points, np.float64)*32)/32
    base = np.floor(points).astype(int)
    fraction = points-base
    result = np.zeros((len(points),)+field.shape[2:], np.float64)
    for dy in [0, 1]:
        for dx in [0, 1]:
            x, y = base[:, 0]+dx, base[:, 1]+dy
            weight = (fraction[:, 0] if dx else 1-fraction[:, 0])*(fraction[:, 1] if dy else 1-fraction[:, 1])
            valid = (x >= 0) & (x < field.shape[1]) & (y >= 0) & (y < field.shape[0])
            weight = weight.reshape((-1,)+(1,)*(field.ndim-2))
            result[valid] += field[y[valid], x[valid]]*weight[valid]
    return result.astype(np.float32)


def main():
    assert sha(PARENT) == PARENT_SHA
    meta = read(DATA/'transforms_train.json')
    frames = {int(Path(f['file_path']).stem):f for f in meta['frames']}
    ordered = sorted(frames)
    flow_manifest = read(ROOT/'data/derived/flow_comparison_train_only/manifest.json')
    flow = {}
    for pair in flow_manifest['pairs']:
        path = ROOT/'data/derived/flow_comparison_train_only'/pair['file']
        assert sha(path) == flow_manifest['output_sha256'][pair['file']]
        with np.load(path, allow_pickle=False) as source:
            flow[pair['source'], pair['target']] = (source['forward'], source['backward'])
    images = {}
    gray = {}
    masks = {}
    for n, frame in frames.items():
        path = DATA/(frame['file_path']+'.png')
        assert sha(path) == flow_manifest['input_sha256'][str(path.relative_to(ROOT))]
        rgba = np.asarray(Image.open(path).convert('RGBA').resize((400, 225), Image.Resampling.BILINEAR))
        gray[n] = cv2.cvtColor(rgba[..., :3], cv2.COLOR_RGB2GRAY).astype(np.float32)
        a = rgba.astype(np.float32)/255
        r, g, b, alpha = a.transpose(2, 0, 1)
        original = (alpha > .95) & (r > .25) & (g > .18) & (b < .8*np.minimum(r, g))
        body = binary_fill_holes(binary_closing((alpha > .95) & (r > .25) & (g > .05)
                  & (b < .85*np.minimum(r, g)), structure=np.ones((3, 3)))) & (alpha > .95)
        highlights = (np.minimum(r, g) > .65) & (b > .5) & (b > .65*np.minimum(r, g))
        masks[1, n] = binary_erosion(original, iterations=2).astype(np.float32)
        masks[3, n] = (binary_erosion(body, iterations=2) & ~binary_dilation(highlights, structure=np.ones((5, 5)))).astype(np.float32)
    audited = {}
    for version in [1, 2, 3]:
        base = OUT if version == 1 else OUT/f'observation_v{version}'
        source = read(base/'image_tracks.json')
        config = read(base/'observation_protocol.json')
        assert source['consumed_cache_fields'] == ['forward', 'backward'] and source['optimizer_updates'] == 0
        assert source['observation_protocol_sha256'] == sha(base/'observation_protocol.json')
        maximum_difference = 0.
        checked_edges = 0
        for item in source['tracks']:
            assert len(item['frames']) >= 8 and item['bootstrap_count'] == 3
            assert not item['occlusion_reidentified'] and not item['physical_material_ID_GT']
            assert all(n in frames for n in item['frames'])
            assert all(abs(t-frames[n]['time']) < 1e-8 for n, t in zip(item['frames'], item['times']))
            points = np.asarray(item['uv_400'], np.float32)
            for index, (first, second) in enumerate(zip(item['frames'][:-1], item['frames'][1:])):
                f, b = flow[first, second]
                p, q = points[index:index+1], points[index+1:index+2]
                predicted = p+independent_sample(f, p)
                difference = float(np.max(np.abs(predicted-q)))
                maximum_difference = max(maximum_difference, difference)
                assert difference < 2e-4, (version, item['track_id'], first, difference)
                fb = np.linalg.norm(q+independent_sample(b, q)-p)
                assert fb <= .7502
                policy = 3 if version == 3 else 1
                assert independent_sample(masks[policy, first], p)[0] >= .9899
                assert independent_sample(masks[policy, second], q)[0] >= .9899
                offsets = np.array([(x, y) for y in range(-2, 3) for x in range(-2, 3)], np.float32)
                a = independent_sample(gray[first], p+offsets)
                z = independent_sample(gray[second], q+offsets)
                a -= a.mean(); z -= z.mean()
                ncc = float((a*z).sum()/(np.linalg.norm(a)*np.linalg.norm(z)))
                assert ncc >= .5999 and np.sqrt((a*a).mean()) >= 1.9999 and np.sqrt((z*z).mean()) >= 1.9999
                checked_edges += 1
            returned = points[-1:].copy()
            for first, second in reversed(list(zip(item['frames'][:-1], item['frames'][1:]))):
                returned = (returned+independent_sample(flow[first, second][1], returned)).astype(np.float32)
            closure = float(np.linalg.norm(returned-points[:1]))
            assert closure <= .7502 and abs(closure-item['reverse_closure_px']) < 2e-4
        audited[f'observation_v{version}'] = dict(tracks=len(source['tracks']), checked_edges=checked_edges,
            maximum_coordinate_replay_error_px=maximum_difference, independent_interpolation_threshold_checks=True)
    for version, base in [(2, OUT), (3, OUT/'landmark_v3')]:
        data = read(base/'bindings.json')
        config = read(base/'binding_protocol.json')
        source = read(OUT/f'observation_v{version}/image_tracks.json')
        evidence = read(base/'bootstrap_evidence.json')
        accepted = {x['track_id']:x for x in data['bindings']}
        rejected = {x['track_id']:x for x in data['rejected']}
        for item in source['tracks']:
            records = [{int(r):v for r, v in evidence[str(item['track_id'])][str(n)].items()} for n in item['frames'][:3]]
            best, bad = choose_support(records, item['uv_400'][:3], config)
            if best is None:
                assert item['track_id'] in rejected and rejected[item['track_id']]['reason'] == bad['reason']
            else:
                saved = accepted[item['track_id']]
                assert saved['rows'] == best['rows'] and np.allclose(saved['weights'], best['weights'], atol=1e-6)
                assert np.allclose(saved['bootstrap_errors_px'], best['bootstrap_errors_px'], atol=1e-6)
                assert abs(sum(saved['weights'])-1) < 1e-8 and min(saved['weights']) >= 0
                assert max(saved['depth_range_by_bootstrap']) <= config['depth_tolerance']+1e-7
        assert all(c['forward_conservation_error'] < 5e-4 for c in data['complete_forward_mass_checks'])
        saved = read(base/'split_manifest.json')
        groups, edges = group_tracks(source['tracks'], data['bindings'], saved['grouping_protocol'])
        replay = split_groups(groups, data['bindings'])
        assert json.loads(json.dumps(replay['groups'])) == saved['groups']
        assert replay['admitted'] == saved['admitted']
        for key in ['training_track_ids', 'holdout_track_ids', 'training_groups', 'holdout_groups']:
            assert replay.get(key) == saved.get(key)
        audited[f'binding_v{version}'] = dict(accepted=len(accepted), rejected=len(rejected),
            bootstrap_only_replay=True, all_fixed_weights_and_rejections_match=True,
            grouped_split_replayed=True, maximum_forward_conservation_error=max(c['forward_conservation_error'] for c in data['complete_forward_mass_checks']))
    # Verify all historical S3-U frozen sources, inputs and checkpoints.
    protected = read(ROOT/'experiments/staged_reconstruction/S3_union/source_freeze.json')
    for name, expected in protected.items():
        assert sha(ROOT/name) == expected, name
    assert sha(PARENT) == PARENT_SHA
    parent_audit = read(ROOT/'experiments/staged_reconstruction/S3_union/physics_state_audit.json')
    assert parent_audit['checkpoint_sha256'] == PARENT_SHA
    source = read(OUT/'landmark_v3/bindings.json')
    split = read(OUT/'landmark_v3/split_manifest.json')
    visual = read(OUT/'landmark_v3/visual_review.json')
    admission = evidence_gates(source['bindings'], split, visual)
    admission.update(user_authorized_research_execution=True, data_quality_gates_not_waived=True,
                     independent_geometry_or_material_GT_available=False)
    write(OUT/'admission.json', admission)
    receipt = dict(checks=audited, historical_protected_files_checked=len(protected),
        parent_sha256=PARENT_SHA, raw_flow_pairs_checked=len(flow), training_images_checked=len(frames),
        optimizer_updates=0, all_checks_passed=True,
        evidence_admitted=admission['evidence_admitted'], testing_and_inference_do_not_establish_material_identity=True)
    write(OUT/'verification.json', receipt)
    source_files = list((ROOT/'variants/staged_reconstruction/surface_trajectory').glob('*.py'))
    source_files += list((ROOT/'tests').glob('test_surface_trajectory_*.py'))+[Path(__file__)]
    write(OUT/'implementation_source_freeze.json', {p.relative_to(ROOT).as_posix():sha(p) for p in source_files})
    artifacts = [p for p in OUT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name != 'artifact_sha256.json']
    write(OUT/'artifact_sha256.json', {p.relative_to(ROOT).as_posix():sha(p) for p in artifacts})
    print(json.dumps(receipt, indent=2), flush=True)
    print('EVIDENCE_ADMISSION', admission['evidence_admitted'], admission['failed_gates'], flush=True)


if __name__ == '__main__':
    main()
