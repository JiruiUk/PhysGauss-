"""Opt-in native no-update gradient worker on one pinned legacy S3-U copy."""
import argparse
import gc
import msvcrt
import os
from pathlib import Path
import sys
import numpy as np
from PIL import Image, ImageDraw
import torch
from ..prepare_data import ROOT, DATA, read, write, sha
from ..train_static import setup, camera, PIPE
from ..stage2.train import image, objective
from .core import audit_scope, state_hash, preserve_rng, rng_state, rng_equal, gradient_stats
from .prepare import BASE, REPORT, RUNS, OUT, stopped, verify


def pixels(value):
    return (value.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()*255).round().astype(np.uint8)


def load_scene(protocol):
    from ..union.prepare import verify_sources
    verify_sources()
    renderer = setup()
    from ..union.model import UnionScene
    parent = read(ROOT/protocol['parent_protocol'])
    scene = UnionScene(parent)
    path = ROOT/protocol['checkpoint']
    assert sha(path) == protocol['checkpoint_sha256']
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    assert checkpoint['iteration'] == 6000 and checkpoint['stage'] == 'S3-U'
    assert checkpoint['phase'] == 'physics'
    scene.load_state_dict(checkpoint['model'], strict=True)
    lineage = {'iteration': checkpoint['iteration'], 'stage': checkpoint['stage'],
        'phase': checkpoint['phase'], 'checkpoint_sha256': sha(path),
        'optimizer_loaded_or_updated': False, 'camera_stack_loaded_or_updated': False,
        'legacy_training_used_conditional_color_masks': True}
    del checkpoint
    return renderer, scene, lineage


def measure(renderer, scene, frame, meta, width):
    before, flags, rng = state_hash(scene), [p.requires_grad for p in scene.parameters()], rng_state()
    cam = camera(frame, meta, width)
    rgba = image(frame, width)
    # Original native target contract only, no alpha mask/opacity supervision.
    gt = torch.tensor(rgba[..., :3]*rgba[..., 3:4], device='cuda').permute(2, 0, 1)
    with torch.no_grad():
        baseline = scene.render(renderer, cam, frame['time'])
        old_rgb, old_alpha = baseline['render'].detach().clone(), baseline['alpha'].detach().clone()
        del baseline
    with audit_scope(scene, enabled=True):
        package = scene.render(renderer, cam, frame['time'])
        assert torch.equal(old_rgb, package['render']) and torch.equal(old_alpha, package['alpha'])
        loss = objective(package['render'], gt)
        groups = {'static_opacity': [scene.static._opacity], 'static_SH_dc': [scene.static._features_dc],
            'static_SH_rest': [scene.static._features_rest], 'canonical_xyz': [scene.pools[0]._xyz],
            'grids': list(scene.grids.parameters()), 'deformation_network': list(scene.networks.parameters())}
        all_parameters = [p for values in groups.values() for p in values]
        gradients = torch.autograd.grad(loss, all_parameters, allow_unused=True)
        stats, offset = {}, 0
        for name, parameters in groups.items():
            stats[name] = gradient_stats(parameters, gradients[offset:offset+len(parameters)])
            assert stats[name]['finite'] and stats[name]['l2'] > 0, name
            offset += len(parameters)
        value = float(loss.detach())
        del package, loss, gradients, all_parameters, groups
    assert state_hash(scene) == before and flags == [p.requires_grad for p in scene.parameters()]
    assert rng_equal(rng, rng_state())
    assert all(p.grad is None for p in scene.parameters())
    return {'frame': frame['file_path'], 'time': frame['time'], 'width': width,
        'full_RGB_loss': value, 'gradient_groups': stats, 'RGB_alpha_flags_on_off_exact': True,
        'model_hash': before, 'caller_flags_RNG_and_leaf_grads_preserved': True}, {
        'rgb': old_rgb.cpu().numpy(), 'alpha': old_alpha.cpu().numpy()}


def native_parity(renderer, scene, frame, meta, width):
    cam = camera(frame, meta, width)
    with torch.no_grad():
        a = scene.render(renderer, cam, frame['time'])
        rgb, alpha = a['render'].clone(), a['alpha'].clone()
        del a
        with audit_scope(scene, True):
            b = scene.render(renderer, cam, frame['time'])
            assert torch.equal(rgb, b['render']) and torch.equal(alpha, b['alpha'])
            del b
    return {'rgb': rgb.cpu().numpy(), 'alpha': alpha.cpu().numpy()}


def render_sheet(renderer, scene, cam, time, path, label, observed=None):
    with torch.no_grad():
        static = renderer(cam, scene.static, PIPE, scene.centers.new_zeros(3), 0., 0., 0.)
        dynamic = renderer(cam, scene.dynamic_at(time), PIPE, scene.centers.new_zeros(3), 0., 0., 0.)
        combined = scene.render(renderer, cam, time)
        values = [('legacy static only', static['render']), ('legacy dynamic only', dynamic['render']),
            ('legacy depth-sorted union', combined['render'])]
        images = [(name, pixels(value)) for name, value in values]
        if observed is not None:
            images.insert(0, ('observed RGB includes motion', pixels(observed)))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas = Image.new('RGB', (640*len(images), 410), '#171923'); draw = ImageDraw.Draw(canvas)
    for col, (name, array) in enumerate(images):
        native = path.with_name(path.stem+'_'+str(col)+'_native.png')
        assert not native.exists(); Image.fromarray(array).save(native)
        canvas.paste(Image.fromarray(array).resize((640, 360), Image.Resampling.LANCZOS), (col*640, 50))
        draw.text((col*640+5, 5), label, fill='white'); draw.text((col*640+5, 25), name, fill='white')
    assert not path.exists(); canvas.save(path)


def runtime_sources():
    result = {}
    for module in list(sys.modules.values()):
        name = getattr(module, '__file__', None)
        if name is None:
            continue
        path = Path(name).resolve()
        if path.is_relative_to(ROOT) and path.suffix == '.py' and 'envs' not in path.parts:
            result[path.relative_to(ROOT).as_posix()] = sha(path)
    return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--enable-joint-revocation-audit', action='store_true')
    parser.add_argument('--phase', choices=['preflight', 'formal'], default='formal')
    parser.add_argument('--replicate', choices=['0', '1'])
    args = parser.parse_args()
    if not args.enable_joint_revocation_audit:
        raise SystemExit('J0 is opt-in; no model or audit outputs opened')
    assert not stopped(); verify()
    protocol = read(BASE/'protocol.json')
    if args.phase == 'formal':
        assert args.replicate is None and read(REPORT/'preflight.json')['passed']
        for name, h in read(REPORT/'preflight.json')['imported_sources'].items():
            assert sha(ROOT/name) == h, name
    else:
        assert args.replicate is not None
    tag = 'formal' if args.phase == 'formal' else 'preflight_'+args.replicate
    receipt = REPORT/(tag+'.json'); assert not receipt.exists(), 'Do not repeat a completed worker'
    RUNS.mkdir(parents=True, exist_ok=True)
    lock = (RUNS/'audit.lock').open('a+b'); lock.seek(0)
    if not lock.read(1):
        lock.write(b'0'); lock.flush()
    lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    try:
        torch.set_num_threads(4); torch.cuda.init()
        caller_rng = rng_state()
        with preserve_rng():
            renderer, scene, lineage = load_scene(protocol)
            initial_hash = state_hash(scene)
            meta = read(DATA/'transforms_train.json'); assert len(meta['frames']) == protocol['expected_train_views']
            indices = [0] if args.phase == 'preflight' else protocol['fixed_train_list_indices']
            rows = []
            for index in indices:
                assert not stopped(), 'User stop: no further measurement'
                frame = meta['frames'][index]
                row, arrays = measure(renderer, scene, frame, meta, protocol['gradient_width'])
                row['train_list_index'] = index; rows.append(row)
                if args.phase == 'preflight':
                    directory = RUNS/tag; directory.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(directory/'gradient_render.npz', **arrays)
                    native = native_parity(renderer, scene, frame, meta, protocol['native_image_width'])
                    np.savez_compressed(directory/'native_render.npz', **native)
                    del native
                else:
                    width = protocol['native_image_width']; rgba = image(frame, width)
                    gt = torch.tensor(rgba[..., :3]*rgba[..., 3:4], device='cuda').permute(2, 0, 1)
                    render_sheet(renderer, scene, camera(frame, meta, width), frame['time'],
                        OUT/f'observed_index{index:03d}.png', 'OLD S3-U final6k; NO UPDATE; '+Path(frame['file_path']).stem, gt)
                    del gt
                del arrays
                write(RUNS/'state.json', {'status': args.phase, 'pid': os.getpid(), 'completed': len(rows), 'total': len(indices)})
                print('J0_GRADIENT', tag, index, row['full_RGB_loss'],
                    {k: v['l2'] for k, v in row['gradient_groups'].items()}, flush=True)
            if args.phase == 'formal':
                byid = {int(Path(frame['file_path']).stem): frame for frame in meta['frames']}
                for camid in protocol['fixed_native_cameras']:
                    cam = camera(byid[camid], meta, protocol['native_image_width'])
                    for number in protocol['fixed_native_frame_numbers']:
                        assert not stopped()
                        render_sheet(renderer, scene, cam, (number-1)/199,
                            OUT/f'fixed{camid}_t{number:04d}.png', f'OLD S3-U final6k; NO GT; c{camid}/t{number}')
            assert state_hash(scene) == initial_hash
            imported = runtime_sources()
            result = {'phase': args.phase, 'rows': rows, 'model_hash_before_and_after': initial_hash,
                'optimizer_updates': 0, 'parameter_values_unchanged': True, 'lineage': lineage,
                'static_geometry_frozen_and_flags_restored': True, 'imported_sources': imported,
                'source_freeze_sha256': sha(REPORT/'source_freeze.json'),
                'no_static_dynamic_mask_loss_or_depth_or_GT_inputs': True,
                'RGBA_alpha_only_original_black_RGB_compositing': True,
                'geometric_ownership_or_cleanup_success_claim': False}
            del scene, renderer; gc.collect(); torch.cuda.empty_cache()
        assert rng_equal(caller_rng, rng_state())
        result['caller_RNG_unchanged_including_construction'] = True
        verify(); assert sha(ROOT/protocol['checkpoint']) == protocol['checkpoint_sha256']
        if args.phase == 'formal':
            result['images'] = {p.relative_to(ROOT).as_posix(): sha(p) for p in OUT.glob('*.png')}
        write(receipt, result)
        write(RUNS/'state.json', {'status': 'measured_pending_manual_review' if args.phase == 'formal' else 'preflight_worker_complete',
            'optimizer_updates': 0, 'tag': tag})
        print('J0_WORKER_COMPLETE', tag, flush=True)
    except Exception as error:
        write(RUNS/'state.json', {'status': 'saved_stopped' if stopped() else 'failed', 'error': repr(error)})
        raise
    finally:
        lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1); lock.close()


if __name__ == '__main__':
    main()
