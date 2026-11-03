"""Official frozen dynamic pointmap prediction; preserve both times' outputs."""
import argparse
from pathlib import Path
import sys
import numpy as np
from PIL import Image
import torch
from safetensors import safe_open
from safetensors.torch import load_file
from ..prepare_data import ROOT, DATA, read, write, sha
from .prepare import EXP, WORK, PREVIOUS


def verify():
    for name, expected in read(EXP / 'source_freeze.json').items():
        assert sha(ROOT / name) == expected, name
    plan = read(EXP / 'protocol.json')
    assert sha(DATA / 'transforms_train.json') == plan['data_metadata_sha256']
    assert sha(PREVIOUS / 'asset_weights.json') == plan['weights_manifest_sha256']
    assert sha(PREVIOUS / 'pair_vendor.json') == plan['vendor_manifest_sha256']
    assert sha(PREVIOUS / 'pilot_protocol.json') == plan['old_protocol_sha256']
    assert sha(PREVIOUS / 'source_camera_audit.json') == plan['known_camera_conversion_audit_sha256']
    return plan


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--enable-calibrated-pointmap-pilot', action='store_true'); args = parser.parse_args()
    if not args.enable_calibrated_pointmap_pilot:
        raise SystemExit('Calibrated pointmaps are opt-in')
    assert not (EXP / 'inference.json').exists() and not WORK.exists()
    plan = verify(); torch.set_num_threads(4)
    lock = read(PREVIOUS / 'pair_vendor.json'); vendor = ROOT / lock['vendor_root']
    assert all(sha(vendor / p) == expected for p, expected in lock['files_sha256'].items())
    weights = read(PREVIOUS / 'asset_weights.json'); path = ROOT / weights['weights_path']
    assert sha(path) == weights['weights_sha256'] and sha(path.parent / 'config.json') == weights['config_sha256']
    sys.path.insert(0, str(vendor))
    from dust3r.model import AsymmetricCroCo3DStereo
    model = AsymmetricCroCo3DStereo(**read(path.parent / 'config.json'))
    state = load_file(path, device='cpu'); live = model.state_dict()
    with safe_open(path, framework='pt', device='cpu') as stream:
        aliases = stream.metadata()
    assert len(aliases) == 8 and set(live) - set(state) == set(aliases)
    for alias, canonical in aliases.items():
        assert live[alias].data_ptr() == live[canonical].data_ptr()
        state[alias] = state[canonical]
    model.load_state_dict(state, strict=True); del state, live
    model = model.eval().requires_grad_(False).cuda()
    metadata = read(DATA / 'transforms_train.json')
    frames = {int(Path(f['file_path']).stem): f for f in metadata['frames']}
    def view(n):
        path = DATA / (frames[n]['file_path'] + '.png')
        with Image.open(path) as im:
            rgba = np.asarray(im.convert('RGBA'))
        rgb = np.rint(rgba[..., :3] * (rgba[..., 3:4].astype(np.float32)/255)).astype(np.uint8)
        im = Image.fromarray(rgb).resize((512, 288), Image.Resampling.LANCZOS)
        value = torch.tensor(np.asarray(im).copy(), dtype=torch.float32, device='cuda').permute(2, 0, 1)[None]/127.5-1
        return {'img': value, 'true_shape': torch.tensor([[288, 512]], device='cuda'), 'idx': n, 'instance': str(n)}, sha(path)
    WORK.mkdir(parents=True); rows = []
    with torch.inference_mode():
        for anchor in plan['anchors']:
            for partner in plan['partners'][str(anchor)]:
                first, a_hash = view(anchor); second, b_hash = view(partner)
                torch.cuda.reset_peak_memory_stats()
                a, b = model(first, second)
                arrays = {'first_points': a['pts3d'][0].cpu().numpy(), 'second_points_in_first': b['pts3d_in_other_view'][0].cpu().numpy(),
                          'first_confidence': a['conf'][0].cpu().numpy(), 'second_confidence': b['conf'][0].cpu().numpy()}
                assert all(np.isfinite(v).all() for v in arrays.values())
                assert arrays['first_points'].shape == arrays['second_points_in_first'].shape == (288, 512, 3)
                assert arrays['first_confidence'].shape == arrays['second_confidence'].shape == (288, 512)
                destination = WORK / f'anchor{anchor:04d}_partner{partner:04d}.npz'
                np.savez_compressed(destination, **arrays)
                rows.append({'anchor': anchor, 'partner': partner, 'path': destination.relative_to(ROOT).as_posix(), 'sha256': sha(destination),
                    'source_RGB_sha256': {str(anchor): a_hash, str(partner): b_hash}, 'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated()})
                print('CALIBRATED_POINTMAP_INFERRED', anchor, partner, float(arrays['first_confidence'].max()), float(arrays['second_confidence'].max()), flush=True)
                del first, second, a, b
    verify()
    write(EXP / 'inference.json', {'pairs': rows, 'protocol_sha256': sha(EXP / 'protocol.json'),
        'source_freeze_sha256': sha(EXP / 'source_freeze.json'), 'actual_frozen_official_float32_pair_forward': True,
        'strict_safetensors_tied_weight_loading': True, 'both_times_pointmaps_preserved': True,
        'cameras_optimized': False, 'PIDG_optimizer_updates': 0, 'test_or_val_RGB_used': False})


if __name__ == '__main__':
    main()
