"""Train-only raw optical flow; no reconstructed depth or auxiliary intrinsics."""
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from variants.flow_comparison.geometry import forward_backward_valid, sample, pixel_grid

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT/'data/derived/unified_flow_train_only'
OLD = ROOT/'data/derived/flow_comparison_train_only'
SOURCE = ROOT/'data/derived/kinematic_resplit/training'
REPORT = ROOT/'experiments/unified_flow'
read = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def prepare():
    torch.set_num_threads(4)
    frames = read(SOURCE/'transforms_train.json')['frames']
    ids = {int(Path(f['file_path']).stem): f for f in frames}
    old = read(OLD/'manifest.json')
    assert sorted(ids) == old['train_frame_ids'] and len(ids) == 120
    inputs = {str(p.relative_to(ROOT)): sha(p) for p in [SOURCE/'transforms_train.json',
        ROOT/'experiments/flow_comparison/flow_model_lock.json', Path(__file__),
        ROOT/'variants/flow_comparison/geometry.py',
        *[SOURCE/(f['file_path']+'.png') for f in frames], *[OLD/p['file'] for p in old['pairs']]]}
    lock = {'inputs': inputs, 'purpose': 'Reuse only raw train UniMatch forward/backward fields; no teacher depth/masks or camera compensation',
            'resolution': [400, 225], 'mask': 'source/warped-target true training alpha >.95/.99, FB, bounds',
            'model': old['model_lock']}
    CACHE.mkdir(parents=True, exist_ok=True); REPORT.mkdir(parents=True, exist_ok=True)
    if (CACHE/'input_lock.json').exists():
        assert read(CACHE/'input_lock.json') == lock, 'Raw flow input lock changed'
    else:
        (CACHE/'input_lock.json').write_text(json.dumps(lock, indent=2)+'\n')
    if (CACHE/'manifest.json').exists():
        manifest = read(CACHE/'manifest.json')
        assert all(sha(CACHE/p) == value for p, value in manifest['output_sha256'].items())
        print('RAW TRAIN FLOW VERIFIED', flush=True); return manifest
    masks = {}
    for i, f in ids.items():
        with Image.open(SOURCE/(f['file_path']+'.png')) as im:
            a = torch.from_numpy(np.asarray(im.getchannel('A'), np.float32).copy())[None, None]/255
        masks[i] = F.interpolate(a, size=(225, 400), mode='bilinear', align_corners=False)[0, 0]
    pairs = []
    for p in old['pairs']:
        a, b = p['source'], p['target']
        with np.load(OLD/p['file']) as raw:
            forward, backward = [torch.from_numpy(raw[k].copy()) for k in ['forward', 'backward']]
        valid = forward_backward_valid(forward, backward)&(masks[a]>.95)&(sample(masks[b], pixel_grid(225, 400, forward)+forward)>.99)
        assert valid.sum()>100
        path = CACHE/f'{a:04d}_{b:04d}.npz'
        np.savez_compressed(path, observed=forward.numpy(), valid=valid.numpy())
        pairs.append({'source': a, 'target': b, 'dt': (b-a)/199, 'file': path.name, 'valid_pixels': int(valid.sum())})
    manifest = {'train_frame_ids': sorted(ids), 'pairs': pairs, 'input_lock_sha256': sha(CACHE/'input_lock.json'),
        'output_sha256': {p.name: sha(p) for p in CACHE.glob('*.npz')}, 'heldout_RGB_used': False,
        'estimated_flow_not_ground_truth': True, 'depth_teacher_used': False, 'camera_compensation': False}
    (CACHE/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    (REPORT/'flow_data_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print('RAW TRAIN FLOW CREATED', len(pairs), flush=True)
    return manifest


if __name__ == '__main__': prepare()
